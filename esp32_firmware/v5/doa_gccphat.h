#pragma once
#include <stdint.h>

#define DOA_MAX_LAG_CAP 16

typedef struct { int side; float lag_samples, lag_ms, conf; int n_valid; } doa_episode_t;

#define DOA_LAG_BIN 0.01f
#define DOA_HIST_BINS (2 * DOA_MAX_LAG_CAP * 100 + 1)

typedef struct {
    float spacing_m, c, band_lo, band_hi, deadzone, min_ratio, floor_margin_db;
    int fs, max_lag, n_bins;
    float floor_db;
    uint32_t hist[DOA_HIST_BINS];   /* lag histogram, bin 0.01 samples over [-max_lag, +max_lag] */
    uint32_t side_counts[3];        /* per-frame side: 0 unknown, 1 left, 2 right */
    uint32_t n_valid;
} doa_tracker_t;

/* Positive lag: the signal reaches L first (source on the left).
 * Work buffers are static (about 12 KB in .bss, nothing on the stack): call these from one task only. */
void doa_gcc_phat(const float *l, const float *r, int max_lag, float band_lo, float band_hi, float *lag, float *ratio);
void doa_gcc_phat_i16(const int16_t *l, const int16_t *r, int max_lag, float band_lo, float band_hi, float *lag, float *ratio);
void doa_tracker_init(doa_tracker_t *t, float spacing_m);
void doa_tracker_reset_episode(doa_tracker_t *t); /* call at every episode start: clears the votes, keeps the adapted noise floor */
int doa_tracker_frame(doa_tracker_t *t, const float *l, const float *r, float *lag_out);
int doa_tracker_frame_i16(doa_tracker_t *t, const int16_t *l, const int16_t *r, float *lag_out); /* the firmware entry point */
void doa_tracker_episode(const doa_tracker_t *t, doa_episode_t *out);
