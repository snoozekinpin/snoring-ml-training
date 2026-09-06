#include "doa_gccphat.h"
#include "fft512.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define N 512
#define PI_F 3.14159265358979323846f
#define DOA_BAND_MIN_FRACTION 0.05  /* same rule as v5/doa.py BAND_MIN_FRACTION */

static float s_win[N];
static int s_ready = 0;
/* static work buffers: keeps ~12 KB off the task stack (not reentrant) */
static float s_lr[N], s_li[N], s_rr[N], s_ri[N], s_gr[N], s_gi[N];
static float s_lf[N], s_rf[N];

static void ensure_init(void) {
    if (s_ready) return;
    fft512_init();
    for (int n = 0; n < N; n++) s_win[n] = 0.5f - 0.5f * cosf(2.0f * PI_F * (float)n / (float)N);
    s_ready = 1;
}

void doa_gcc_phat(const float *l, const float *r, int max_lag, float band_lo, float band_hi, float *lag_out, float *ratio_out) {
    float *lr = s_lr, *li = s_li, *rr = s_rr, *ri = s_ri, *gr = s_gr, *gi = s_gi;
    ensure_init();
    if (max_lag > DOA_MAX_LAG_CAP) max_lag = DOA_MAX_LAG_CAP;
    for (int n = 0; n < N; n++) { lr[n] = l[n] * s_win[n]; li[n] = 0.0f; rr[n] = r[n] * s_win[n]; ri[n] = 0.0f; }
    fft512_forward(lr, li);
    fft512_forward(rr, ri);
    /* share of the L channel's power inside the band: below BAND_MIN_FRACTION there is no usable estimate */
    double total = 0.0, inband = 0.0;
    for (int k = 0; k <= N / 2; k++) {
        double pw = (double)lr[k] * lr[k] + (double)li[k] * li[k];
        float f = (float)k * 16000.0f / (float)N;
        total += pw;
        if (f >= band_lo && f <= band_hi) inband += pw;
    }
    if (total <= 1e-18 || inband < DOA_BAND_MIN_FRACTION * total) { *lag_out = 0.0f; *ratio_out = 0.0f; return; }
    memset(gr, 0, N * sizeof(float));
    memset(gi, 0, N * sizeof(float));
    for (int k = 0; k <= N / 2; k++) {
        float re = rr[k] * lr[k] + ri[k] * li[k];  /* R * conj(L) */
        float im = ri[k] * lr[k] - rr[k] * li[k];
        float mag = sqrtf(re * re + im * im) + 1e-12f;
        float f = (float)k * 16000.0f / (float)N;
        if (f < band_lo || f > band_hi) { re = 0.0f; im = 0.0f; } else { re /= mag; im /= mag; }
        gr[k] = re; gi[k] = im;
        if (k > 0 && k < N / 2) { gr[N - k] = re; gi[N - k] = -im; }
    }
    fft512_inverse(gr, gi);
    static float vals[2 * DOA_MAX_LAG_CAP + 1]; /* off the task stack, like the other work buffers */
    int count = 2 * max_lag + 1, kbest = 0;
    for (int m = 0; m < count; m++) {
        int idx = ((m - max_lag) % N + N) % N;
        vals[m] = gr[idx];
        if (vals[m] > vals[kbest]) kbest = m;
    }
    float lag = (float)(kbest - max_lag);
    if (kbest > 0 && kbest < count - 1) {
        float y0 = vals[kbest - 1], y1 = vals[kbest], y2 = vals[kbest + 1];
        float denom = y0 - 2.0f * y1 + y2;
        if (denom < 0.0f) lag += 0.5f * (y0 - y2) / denom;
    }
    /* peak ratio against the highest correlation anywhere else (all N lags except the peak and its neighbours) */
    int idx = ((kbest - max_lag) % N + N) % N;
    float second = 0.0f;
    int any = 0;
    for (int m = 0; m < N; m++) {
        if (m == idx || m == (idx + 1) % N || m == (idx - 1 + N) % N) continue;
        if (!any || gr[m] > second) { second = gr[m]; any = 1; }
    }
    if (!(any && vals[kbest] > 1e-9f && second > 1e-9f)) { *lag_out = 0.0f; *ratio_out = 0.0f; return; } /* degenerate: never valid */
    *lag_out = lag;
    *ratio_out = vals[kbest] / second;
}

void doa_gcc_phat_i16(const int16_t *l, const int16_t *r, int max_lag, float band_lo, float band_hi, float *lag_out, float *ratio_out) {
    for (int n = 0; n < N; n++) { s_lf[n] = (float)l[n] / 32768.0f; s_rf[n] = (float)r[n] / 32768.0f; }
    doa_gcc_phat(s_lf, s_rf, max_lag, band_lo, band_hi, lag_out, ratio_out);
}

void doa_tracker_init(doa_tracker_t *t, float spacing_m) {
    memset(t, 0, sizeof(*t));
    t->spacing_m = spacing_m; t->c = 343.0f; t->fs = 16000;
    t->band_lo = 60.0f; t->band_hi = 3000.0f; t->deadzone = 0.2f; t->min_ratio = 1.5f; t->floor_margin_db = 6.0f;
    t->max_lag = (int)ceilf(spacing_m / t->c * (float)t->fs) + 1;
    if (t->max_lag > DOA_MAX_LAG_CAP) t->max_lag = DOA_MAX_LAG_CAP;
    t->n_bins = (int)(2.0f * (float)t->max_lag / DOA_LAG_BIN + 0.5f) + 1;
    t->floor_db = -60.0f;
}

void doa_tracker_reset_episode(doa_tracker_t *t) {
    memset(t->hist, 0, sizeof(t->hist));
    memset(t->side_counts, 0, sizeof(t->side_counts));
    t->n_valid = 0; /* floor_db keeps adapting across episodes, as DoaTracker.reset() in Python */
}

static int lag_bin(const doa_tracker_t *t, float lag) {
    int b = (int)floorf((lag + (float)t->max_lag) / DOA_LAG_BIN + 0.5f);
    if (b < 0) b = 0;
    if (b > t->n_bins - 1) b = t->n_bins - 1;
    return b;
}

static int side_of(float lag, int max_lag, float deadzone) {
    float thr = deadzone * (float)max_lag;
    if (lag > thr) return 1;
    if (lag < -thr) return 2;
    return 0;
}

int doa_tracker_frame(doa_tracker_t *t, const float *l, const float *r, float *lag_out) {
    double sq = 0.0;
    for (int n = 0; n < N; n++) sq += (double)l[n] * (double)l[n];
    float level = 20.0f * log10f(sqrtf((float)(sq / N)) + 1e-9f);
    float lag, ratio;
    doa_gcc_phat(l, r, t->max_lag, t->band_lo, t->band_hi, &lag, &ratio);
    int coherent = ratio >= t->min_ratio;
    int valid = level >= t->floor_db + t->floor_margin_db && coherent; /* judged against the floor before this frame */
    if (level < t->floor_db + 5.0f) t->floor_db += 0.02f * (level - t->floor_db);          /* same rule as v5/doa.py NoiseFloor */
    else if (!coherent) t->floor_db += 0.002f * (level - t->floor_db);                    /* loud coherent frames never raise it */
    if (valid) {
        t->hist[lag_bin(t, lag)]++;
        t->side_counts[side_of(lag, t->max_lag, t->deadzone)]++;
        t->n_valid++;
    }
    if (lag_out) *lag_out = lag;
    return valid;
}

int doa_tracker_frame_i16(doa_tracker_t *t, const int16_t *l, const int16_t *r, float *lag_out) {
    for (int n = 0; n < N; n++) { s_lf[n] = (float)l[n] / 32768.0f; s_rf[n] = (float)r[n] / 32768.0f; }
    return doa_tracker_frame(t, s_lf, s_rf, lag_out);
}

void doa_tracker_episode(const doa_tracker_t *t, doa_episode_t *out) {
    memset(out, 0, sizeof(*out));
    if (t->n_valid == 0) return;
    uint32_t target = (t->n_valid - 1) / 2 + 1, cum = 0; /* lower median, same as the Python reference */
    int b = 0;
    for (; b < t->n_bins; b++) {
        cum += t->hist[b];
        if (cum >= target) break;
    }
    float med = (float)b * DOA_LAG_BIN - (float)t->max_lag;
    int side = side_of(med, t->max_lag, t->deadzone);
    out->side = side;
    out->lag_samples = med;
    out->lag_ms = med / (float)t->fs * 1000.0f;
    out->conf = (float)t->side_counts[side] / (float)t->n_valid;
    out->n_valid = (int)t->n_valid;
}
