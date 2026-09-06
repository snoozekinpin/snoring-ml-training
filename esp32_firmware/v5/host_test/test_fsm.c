/* Usage: test_fsm <fsm_trace.txt>. Replays the Python reference trace. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "snore_episode_fsm.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s fsm_trace.txt\n", argv[0]); return 2; }
    FILE *fh = fopen(argv[1], "r");
    if (!fh) { perror("open"); return 2; }
    fsm_params_t p;
    if (fscanf(fh, "%f %d %d %d %d %d %d %d", &p.tau, &p.tick_ms, &p.hold_ticks, &p.confirm_ticks, &p.verify_ticks, &p.min_bursts, &p.period_min_ticks, &p.period_max_ticks) != 8) {
        fprintf(stderr, "bad header\n");
        return 2;
    }
    fsm_t f;
    if (fsm_init(&f, &p) != 0) { fprintf(stderr, "fsm_init rejected the golden parameters\n"); return 1; }
    {   /* capacity validation: oversized timing must be refused instead of silently truncated */
        fsm_t g;
        fsm_params_t big = p;
        big.hold_ticks = FSM_MAX_HIST;
        if (fsm_init(&g, &big) == 0) { fprintf(stderr, "fsm_init accepted hold_ticks beyond FSM_MAX_HIST\n"); return 1; }
        big = p;
        big.confirm_ticks = FSM_MAX_BURSTS;
        if (fsm_init(&g, &big) == 0) { fprintf(stderr, "fsm_init accepted a confirm window beyond FSM_MAX_BURSTS\n"); return 1; }
        printf("fsm capacity validation ok\n");
    }
    float prob, dur, mean_p, level;
    int state, active, event, n_bursts, n_hits, tick = 0, mismatches = 0, ends = 0, retained_mismatches = 0;
    /* Oracle for the retained hit/burst sets: every hit (burst start) of the last confirm+hold+1 ticks after the
     * last episode end must be in the buffers, so a window equal to the capacity must not lose the current tick. */
    enum { RING = 4096 };
    static unsigned char hit_ring[RING], burst_ring[RING];
    int max_age = p.confirm_ticks + p.hold_ticks, prev_hit = 0, last_end = -1;
    while (fscanf(fh, "%f %d %d %d %f %f %d %d %f", &prob, &state, &active, &event, &dur, &mean_p, &n_bursts, &n_hits, &level) == 9) {
        float lvl = -40.0f + 5.0f * sinf((float)tick / 7.0f);
        int hit = prob >= p.tau;
        hit_ring[tick % RING] = (unsigned char)hit;
        burst_ring[tick % RING] = (unsigned char)(hit && !prev_hit);
        prev_hit = hit;
        fsm_event_t ev = fsm_tick(&f, prob, lvl);
        if (ev == FSM_EV_EPISODE_END) { last_end = tick; prev_hit = 0; }
        int exp_hits = 0, exp_bursts = 0;
        for (int a = 0; a <= max_age && tick - a > last_end; a++) { exp_hits += hit_ring[(tick - a) % RING]; exp_bursts += burst_ring[(tick - a) % RING]; }
        if (f.n_hits_buf != exp_hits || f.n_bursts != exp_bursts) {
            if (retained_mismatches < 10) printf("tick %d: retained hits %d (expected %d) bursts %d (expected %d) MISMATCH\n", tick, f.n_hits_buf, exp_hits, f.n_bursts, exp_bursts);
            retained_mismatches++;
        }
        int ok = (int)f.state == state && f.active == active && (int)ev == event;
        if (ok && event == 2) {
            ends++;
            ok = fabsf(f.last_duration_s - dur) <= 1e-3f && fabsf(f.last_mean_p - mean_p) <= 1e-4f && f.last_n_bursts == n_bursts && f.last_n_hits == n_hits && fabsf(f.last_level_dbfs - level) <= 1e-3f;
            if (!ok && mismatches < 10)
                printf("tick %d: end fields expected dur %.3f mean %.6f bursts %d hits %d level %.4f, got %.3f %.6f %d %d %.4f\n", tick, dur, mean_p, n_bursts, n_hits, level, f.last_duration_s, f.last_mean_p, f.last_n_bursts, f.last_n_hits, f.last_level_dbfs);
        } else if (!ok && mismatches < 10) {
            printf("tick %d: p=%.3f expected state %d active %d event %d, got %d %d %d\n", tick, prob, state, active, event, (int)f.state, f.active, (int)ev);
        }
        if (!ok) mismatches++;
        tick++;
    }
    fclose(fh);
    printf("fsm parity: %d ticks (window %d, capacity %d), %d end events checked, %d mismatches, %d retained-set mismatches\n", tick, max_age + 1, FSM_MAX_HITS, ends, mismatches, retained_mismatches);
    return (mismatches || retained_mismatches || ends == 0) ? 1 : 0;
}
