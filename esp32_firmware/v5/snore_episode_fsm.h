#pragma once
#include <stdint.h>

#define FSM_MAX_HIST 64
#define FSM_MAX_BURSTS 64
#define FSM_MAX_HITS 64

typedef struct {
    float tau;
    int tick_ms, hold_ticks, confirm_ticks, verify_ticks, min_bursts, period_min_ticks, period_max_ticks;
} fsm_params_t;

typedef enum { FSM_IDLE = 0, FSM_ACTIVE = 1, FSM_CONFIRMED = 2 } fsm_state_t;
typedef enum { FSM_EV_NONE = 0, FSM_EV_EPISODE_START = 1, FSM_EV_EPISODE_END = 2 } fsm_event_t;

typedef struct {
    fsm_params_t p;
    fsm_state_t state;
    uint32_t tick;
    float hist[FSM_MAX_HIST];
    int hist_len, hist_pos;
    int streak_q, in_burst;
    uint32_t bursts[FSM_MAX_BURSTS];
    int n_bursts;
    uint32_t hit_tick[FSM_MAX_HITS];
    float hit_p[FSM_MAX_HITS], hit_level[FSM_MAX_HITS];
    int n_hits_buf;
    int active;
    float activity;
    int below_ticks;
    uint32_t ep_first_burst, ep_last_hit;
    int ep_n_bursts, ep_n_hits;
    float ep_sum_p, ep_sum_level;
    float last_duration_s, last_mean_p, last_level_dbfs;
    int last_n_bursts, last_n_hits;
    uint32_t last_start_tick;
} fsm_t;

/* Returns 0, or -1 when the parameters exceed the fixed capacities
 * (hold_ticks + 1 <= FSM_MAX_HIST; confirm_ticks + hold_ticks + 1 <= FSM_MAX_BURSTS and <= FSM_MAX_HITS). */
int fsm_init(fsm_t *f, const fsm_params_t *p);
fsm_event_t fsm_tick(fsm_t *f, float p, float level_dbfs);
