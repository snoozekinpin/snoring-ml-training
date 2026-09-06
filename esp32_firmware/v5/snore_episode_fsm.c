#include "snore_episode_fsm.h"
#include <string.h>
#include <stdlib.h>

int fsm_init(fsm_t *f, const fsm_params_t *p) {
    int window = p->confirm_ticks + p->hold_ticks + 1; /* one burst or hit per tick, worst case */
    memset(f, 0, sizeof(*f));
    f->p = *p;
    f->state = FSM_IDLE;
    if (p->hold_ticks < 0 || p->confirm_ticks < 0 || p->verify_ticks < 0 || p->min_bursts < 1) return -1;
    if (p->hold_ticks + 1 > FSM_MAX_HIST || window > FSM_MAX_BURSTS || window > FSM_MAX_HITS) return -1;
    return 0;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int periodic(const fsm_t *f) {
    static uint32_t gaps[FSM_MAX_BURSTS]; /* work buffer off the task stack; the FSM is driven from one task */
    int n = f->n_bursts - 1;
    if (f->n_bursts < f->p.min_bursts) return 0;
    for (int i = 0; i < n; i++) gaps[i] = f->bursts[i + 1] - f->bursts[i];
    qsort(gaps, (size_t)n, sizeof(uint32_t), cmp_u32);
    uint32_t med = gaps[(n - 1) / 2]; /* lower median, same as the Python reference */
    return med >= (uint32_t)f->p.period_min_ticks && med <= (uint32_t)f->p.period_max_ticks;
}

static void drop_front_hit(fsm_t *f) {
    memmove(f->hit_tick, f->hit_tick + 1, (size_t)(f->n_hits_buf - 1) * sizeof(uint32_t));
    memmove(f->hit_p, f->hit_p + 1, (size_t)(f->n_hits_buf - 1) * sizeof(float));
    memmove(f->hit_level, f->hit_level + 1, (size_t)(f->n_hits_buf - 1) * sizeof(float));
    f->n_hits_buf--;
}

fsm_event_t fsm_tick(fsm_t *f, float p, float level_dbfs) {
    const fsm_params_t *P = &f->p;
    uint32_t i = f->tick++;
    int cap = P->hold_ticks + 1;
    uint32_t max_age = (uint32_t)(P->confirm_ticks + P->hold_ticks);
    fsm_event_t ev = FSM_EV_NONE;
    f->hist[f->hist_pos] = p;
    f->hist_pos = (f->hist_pos + 1) % cap;
    if (f->hist_len < cap) f->hist_len++;
    float act = f->hist[0];
    for (int k = 1; k < f->hist_len; k++) if (f->hist[k] > act) act = f->hist[k];
    f->activity = act;
    int hit = p >= P->tau;
    f->active = act >= P->tau;
    int burst_started = hit && !f->in_burst;
    /* Expire first, then append: after expiry at most max_age entries remain, so a window of
     * confirm + hold + 1 == capacity (accepted by fsm_init) never drops the current tick. The retained
     * set equals the Python reference, which appends first but never expires the current tick. */
    while (f->n_bursts > 0 && i - f->bursts[0] > max_age) {
        memmove(f->bursts, f->bursts + 1, (size_t)(f->n_bursts - 1) * sizeof(uint32_t));
        f->n_bursts--;
    }
    while (f->n_hits_buf > 0 && i - f->hit_tick[0] > max_age) drop_front_hit(f);
    if (burst_started && f->n_bursts < FSM_MAX_BURSTS) f->bursts[f->n_bursts++] = i;
    if (hit && f->n_hits_buf < FSM_MAX_HITS) {
        f->hit_tick[f->n_hits_buf] = i;
        f->hit_p[f->n_hits_buf] = p;
        f->hit_level[f->n_hits_buf] = level_dbfs;
        f->n_hits_buf++;
    }
    f->in_burst = hit;
    if (f->active) {
        f->streak_q += 2;
        if (f->streak_q > 4 * P->confirm_ticks) f->streak_q = 4 * P->confirm_ticks;
    } else if (f->streak_q > 0) {
        f->streak_q--;
    }
    if (f->state == FSM_IDLE && f->active) f->state = FSM_ACTIVE;
    if (f->state == FSM_ACTIVE) {
        if (f->streak_q == 0) {
            f->state = FSM_IDLE;
        } else if (f->streak_q >= 2 * P->confirm_ticks && periodic(f)) {
            uint32_t first = f->bursts[0];
            f->state = FSM_CONFIRMED;
            f->below_ticks = 0;
            f->ep_first_burst = first;
            f->ep_last_hit = i;
            f->ep_n_bursts = f->n_bursts;
            f->ep_sum_p = 0.0f;
            f->ep_sum_level = 0.0f;
            f->ep_n_hits = 0;
            for (int k = 0; k < f->n_hits_buf; k++) {
                if (f->hit_tick[k] >= first) {
                    f->ep_sum_p += f->hit_p[k];
                    f->ep_sum_level += f->hit_level[k];
                    f->ep_n_hits++;
                    f->ep_last_hit = f->hit_tick[k];
                }
            }
            ev = FSM_EV_EPISODE_START;
        }
    } else if (f->state == FSM_CONFIRMED) {
        if (hit) {
            f->ep_last_hit = i;
            f->ep_sum_p += p;
            f->ep_n_hits++;
            f->ep_sum_level += level_dbfs;
            if (burst_started) f->ep_n_bursts++;
        }
        if (!f->active) {
            f->below_ticks++;
            if (f->below_ticks >= P->verify_ticks) {
                int n = f->ep_n_hits > 0 ? f->ep_n_hits : 1;
                f->last_duration_s = (float)(f->ep_last_hit - f->ep_first_burst + 2) * (float)P->tick_ms / 1000.0f;
                f->last_mean_p = f->ep_sum_p / (float)n;
                f->last_level_dbfs = f->ep_sum_level / (float)n;
                f->last_n_bursts = f->ep_n_bursts;
                f->last_n_hits = f->ep_n_hits;
                f->last_start_tick = f->ep_first_burst;
                f->state = FSM_IDLE;
                f->streak_q = 0;
                f->n_bursts = 0;
                f->n_hits_buf = 0;
                f->in_burst = 0;
                ev = FSM_EV_EPISODE_END;
            }
        } else {
            f->below_ticks = 0;
        }
    }
    return ev;
}
