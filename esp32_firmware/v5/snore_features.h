#pragma once
#include <stdint.h>
#include "feature_spec.h"

/* Feature spec v1 (see v5/features.py). X is row-major [SF_N_FRAMES][SF_N_MELS].
 * Work buffers are static (nothing on the task stack); call from one task only. */
void sf_init(void);
void sf_compute(const int16_t *win, float *X);
void sf_quantize(const float *X, int8_t *q, float scale, int zero_point);
