#include "snore_features.h"
#include "mel_filterbank.h"
#include "fft512.h"
#include <math.h>

static float s_win[SF_N_FFT];
static int s_ready = 0;
/* static work buffers (about 5 KB in .bss, nothing on the task stack); not reentrant */
static float s_re[SF_N_FFT], s_im[SF_N_FFT], s_power[SF_N_BINS];

void sf_init(void) {
    fft512_init();
    for (int n = 0; n < SF_N_FFT; n++)
        s_win[n] = 0.5f - 0.5f * cosf(2.0f * SF_PI * (float)n / (float)SF_N_FFT); /* periodic Hann */
    s_ready = 1;
}

void sf_compute(const int16_t *win, float *X) {
    float *re = s_re, *im = s_im, *power = s_power;
    double sum = 0.0;
    if (!s_ready) sf_init();
    for (int t = 0; t < SF_N_FRAMES; t++) {
        const int16_t *frame = win + t * SF_HOP;
        for (int n = 0; n < SF_N_FFT; n++) {
            re[n] = ((float)frame[n] / 32768.0f) * s_win[n];
            im[n] = 0.0f;
        }
        fft512_forward(re, im);
        for (int k = 0; k < SF_N_BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];
        for (int m = 0; m < SF_N_MELS; m++) {
            const float *w = sf_mel_w + sf_mel_off[m];
            int start = sf_mel_start[m], len = sf_mel_len[m];
            float acc = 0.0f;
            for (int j = 0; j < len; j++) acc += w[j] * power[start + j];
            float L = 10.0f * log10f(acc + SF_LOG_EPS);
            X[t * SF_N_MELS + m] = L;
            sum += (double)L;
        }
    }
    float mean = (float)(sum / (double)SF_FEATURE_DIM);
    for (int i = 0; i < SF_FEATURE_DIM; i++) {
        float v = (X[i] - mean) / SF_NORM_DIV;
        X[i] = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
    }
}

void sf_quantize(const float *X, int8_t *q, float scale, int zero_point) {
    for (int i = 0; i < SF_FEATURE_DIM; i++) {
        long r = (long)rintf(X[i] / scale) + zero_point; /* rintf: half-to-even like numpy.round */
        if (r < -128) r = -128;
        if (r > 127) r = 127;
        q[i] = (int8_t)r;
    }
}
