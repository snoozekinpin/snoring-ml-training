/* Usage: test_doa <doa_cases.bin> <doa_tracker.bin>. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "doa_gccphat.h"

static void to_float(const int16_t *in, float *out) { for (int n = 0; n < 512; n++) out[n] = (float)in[n] / 32768.0f; }

static int run_cases(const char *path) {
    FILE *fh = fopen(path, "rb");
    if (!fh) { perror("open cases"); return 1; }
    int32_t hdr[3];
    if (fread(hdr, sizeof(int32_t), 3, fh) != 3 || hdr[1] != 512) { fprintf(stderr, "bad cases header\n"); return 1; }
    int failures = 0;
    int16_t li[512], ri[512];
    float l[512], r[512], expect[2];
    for (int c = 0; c < hdr[0]; c++) {
        if (fread(li, sizeof(int16_t), 512, fh) != 512 || fread(ri, sizeof(int16_t), 512, fh) != 512 || fread(expect, sizeof(float), 2, fh) != 2) return 1;
        to_float(li, l);
        to_float(ri, r);
        float lag, ratio, lag16, ratio16;
        doa_gcc_phat(l, r, hdr[2], 60.0f, 3000.0f, &lag, &ratio);
        doa_gcc_phat_i16(li, ri, hdr[2], 60.0f, 3000.0f, &lag16, &ratio16); /* firmware entry point must match the float path */
        int ok = fabsf(lag - expect[0]) <= 0.05f && ((ratio >= 1.5f) == (expect[1] >= 1.5f)) && fabsf(lag16 - lag) <= 1e-6f && fabsf(ratio16 - ratio) <= 1e-6f * (1.0f + fabsf(ratio));
        printf("case %d: lag %.3f (ref %.3f) ratio %.2f (ref %.2f) %s\n", c, lag, expect[0], ratio, expect[1], ok ? "ok" : "MISMATCH");
        if (!ok) failures++;
    }
    fclose(fh);
    return failures;
}

static int run_long_episode(const char *cases_path) {
    /* 3000 frames with the source on the left (case delay +2) then 2500 on the right (delay -2):
     * counts must be unbounded and the lower median must stay with the majority side. */
    FILE *fh = fopen(cases_path, "rb");
    if (!fh) { perror("open cases"); return 1; }
    int32_t hdr[3];
    if (fread(hdr, sizeof(int32_t), 3, fh) != 3) return 1;
    int16_t li[512], ri[512];
    float expect[2], left_l[512], left_r[512], right_l[512], right_r[512];
    for (int c = 0; c < hdr[0] && c <= 7; c++) {
        if (fread(li, sizeof(int16_t), 512, fh) != 512 || fread(ri, sizeof(int16_t), 512, fh) != 512 || fread(expect, sizeof(float), 2, fh) != 2) return 1;
        if (c == 1) { to_float(li, right_l); to_float(ri, right_r); }  /* delay -2: reaches R first */
        if (c == 7) { to_float(li, left_l); to_float(ri, left_r); }    /* delay +2: reaches L first */
    }
    fclose(fh);
    doa_tracker_t t;
    doa_tracker_init(&t, 0.06f);
    float lag_left = 0.0f, lag_right = 0.0f;
    for (int k = 0; k < 3000; k++) doa_tracker_frame(&t, left_l, left_r, &lag_left);
    for (int k = 0; k < 2500; k++) doa_tracker_frame(&t, right_l, right_r, &lag_right);
    doa_episode_t ep;
    doa_tracker_episode(&t, &ep);
    int ok = ep.n_valid == 5500 && ep.side == 1 && fabsf(ep.lag_samples - lag_left) <= 0.05f && fabsf(ep.conf - 3000.0f / 5500.0f) <= 1e-6f;
    printf("long episode: n_valid %d side %d lag %.3f (left frame lag %.3f, right %.3f) conf %.4f %s\n", ep.n_valid, ep.side, ep.lag_samples, lag_left, lag_right, ep.conf, ok ? "ok" : "MISMATCH");
    return ok ? 0 : 1;
}

static int run_tracker(const char *path) {
    FILE *fh = fopen(path, "rb");
    if (!fh) { perror("open tracker"); return 1; }
    int32_t hdr[2];
    float spacing;
    if (fread(hdr, sizeof(int32_t), 2, fh) != 2 || fread(&spacing, sizeof(float), 1, fh) != 1 || hdr[1] != 512) { fprintf(stderr, "bad tracker header\n"); return 1; }
    doa_tracker_t t, t16;
    doa_tracker_init(&t, spacing);
    doa_tracker_init(&t16, spacing);
    int16_t *frames = (int16_t *)malloc((size_t)hdr[0] * 1024 * sizeof(int16_t)); /* kept for the second, mirrored episode */
    if (!frames) return 1;
    float l[512], r[512], exp_lag;
    int32_t exp_valid;
    int failures = 0;
    for (int k = 0; k < hdr[0]; k++) {
        int16_t *li = frames + (size_t)k * 1024, *ri = li + 512;
        if (fread(li, sizeof(int16_t), 512, fh) != 512 || fread(ri, sizeof(int16_t), 512, fh) != 512 || fread(&exp_valid, sizeof(int32_t), 1, fh) != 1 || fread(&exp_lag, sizeof(float), 1, fh) != 1) return 1;
        to_float(li, l);
        to_float(ri, r);
        float lag, lag16;
        int valid = doa_tracker_frame(&t, l, r, &lag);
        int valid16 = doa_tracker_frame_i16(&t16, li, ri, &lag16); /* firmware entry point, same frames */
        if (valid != exp_valid || fabsf(lag - exp_lag) > 0.05f || valid16 != valid || fabsf(lag16 - lag) > 1e-6f) {
            if (failures < 10) printf("frame %d: valid %d/%d (ref %d) lag %.3f/%.3f (ref %.3f) MISMATCH\n", k, valid, valid16, exp_valid, lag, lag16, exp_lag);
            failures++;
        }
    }
    int32_t side, n_valid;
    float ep_ref[3];
    if (fread(&side, sizeof(int32_t), 1, fh) != 1 || fread(ep_ref, sizeof(float), 3, fh) != 3 || fread(&n_valid, sizeof(int32_t), 1, fh) != 1) return 1;
    fclose(fh);
    doa_episode_t ep, ep16;
    doa_tracker_episode(&t, &ep);
    doa_tracker_episode(&t16, &ep16);
    int ok = ep.side == side && ep.n_valid == n_valid && fabsf(ep.lag_samples - ep_ref[0]) <= 0.05f && fabsf(ep.lag_ms - ep_ref[1]) <= 0.05f / 16.0f && fabsf(ep.conf - ep_ref[2]) <= 1e-6f
             && ep16.side == ep.side && ep16.n_valid == ep.n_valid && fabsf(ep16.lag_samples - ep.lag_samples) <= 1e-6f && fabsf(ep16.conf - ep.conf) <= 1e-6f;
    printf("tracker episode: side %d (ref %d) lag %.3f (ref %.3f) conf %.4f (ref %.4f) n_valid %d (ref %d) %s\n", ep.side, side, ep.lag_samples, ep_ref[0], ep.conf, ep_ref[2], ep.n_valid, n_valid, ok ? "ok" : "MISMATCH");

    /* Episode reset: votes cleared, adapted floor kept. A mirrored second episode (channels swapped) on the reset
     * trackers must equal a fresh tracker given the same floor (no carry-over) and land on the opposite side. */
    float floor_before = t.floor_db;
    doa_tracker_reset_episode(&t);
    doa_tracker_reset_episode(&t16);
    int reset_ok = t.n_valid == 0 && t.floor_db == floor_before && t16.n_valid == 0 && t16.floor_db == floor_before;
    doa_tracker_t fresh;
    doa_tracker_init(&fresh, spacing);
    fresh.floor_db = t.floor_db;
    int mismatch2 = 0;
    for (int k = 0; k < hdr[0]; k++) {
        const int16_t *li = frames + (size_t)k * 1024, *ri = li + 512;
        to_float(li, l);
        to_float(ri, r);
        float lag_a, lag_b, lag_c;
        int va = doa_tracker_frame(&t, r, l, &lag_a);
        int vb = doa_tracker_frame_i16(&t16, ri, li, &lag_b);
        int vc = doa_tracker_frame(&fresh, r, l, &lag_c);
        if (va != vb || va != vc || fabsf(lag_a - lag_b) > 1e-6f || fabsf(lag_a - lag_c) > 1e-6f) mismatch2++;
    }
    free(frames);
    doa_episode_t ep2, ep2_16, epf;
    doa_tracker_episode(&t, &ep2);
    doa_tracker_episode(&t16, &ep2_16);
    doa_tracker_episode(&fresh, &epf);
    int mirrored = ep.side == 1 ? 2 : (ep.side == 2 ? 1 : 0);
    int ok2 = reset_ok && mismatch2 == 0 && ep2.n_valid > 0 && ep2.side == mirrored && fabsf(ep2.lag_samples + ep.lag_samples) <= 0.2f
              && ep2.side == epf.side && ep2.n_valid == epf.n_valid && fabsf(ep2.lag_samples - epf.lag_samples) <= 1e-6f && fabsf(ep2.conf - epf.conf) <= 1e-6f
              && ep2_16.side == ep2.side && ep2_16.n_valid == ep2.n_valid && fabsf(ep2_16.lag_samples - ep2.lag_samples) <= 1e-6f && fabsf(ep2_16.conf - ep2.conf) <= 1e-6f;
    printf("tracker reset + mirrored episode: side %d (expected %d) lag %.3f (first %.3f) n_valid %d (fresh %d) frame mismatches %d %s\n",
           ep2.side, mirrored, ep2.lag_samples, ep.lag_samples, ep2.n_valid, epf.n_valid, mismatch2, ok2 ? "ok" : "MISMATCH");
    return failures + (ok ? 0 : 1) + (ok2 ? 0 : 1);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s doa_cases.bin doa_tracker.bin\n", argv[0]); return 2; }
    int failures = run_cases(argv[1]) + run_tracker(argv[2]) + run_long_episode(argv[1]);
    printf("doa parity: %d failing checks\n", failures);
    return failures ? 1 : 0;
}
