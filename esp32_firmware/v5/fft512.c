#include "fft512.h"
#include <math.h>

#define FFT_N 512
#define FFT_PI 3.14159265358979323846f

static float s_cos[FFT_N / 2];
static float s_sin[FFT_N / 2];
static unsigned short s_rev[FFT_N];
static int s_ready = 0;

void fft512_init(void) {
    for (int k = 0; k < FFT_N / 2; k++) {
        s_cos[k] = cosf(2.0f * FFT_PI * (float)k / (float)FFT_N);
        s_sin[k] = sinf(2.0f * FFT_PI * (float)k / (float)FFT_N);
    }
    for (int i = 0; i < FFT_N; i++) {
        unsigned r = 0;
        for (int b = 0; b < 9; b++) r |= ((unsigned)(i >> b) & 1u) << (8 - b);
        s_rev[i] = (unsigned short)r;
    }
    s_ready = 1;
}

static void fft_core(float *re, float *im, int inverse) {
    if (!s_ready) fft512_init();
    for (int i = 0; i < FFT_N; i++) {
        int j = s_rev[i];
        if (j > i) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= FFT_N; len <<= 1) {
        int half = len >> 1, step = FFT_N / len;
        for (int i = 0; i < FFT_N; i += len) {
            for (int j = 0; j < half; j++) {
                float wr = s_cos[j * step];
                float wi = inverse ? s_sin[j * step] : -s_sin[j * step];
                float xr = re[i + j + half], xi = im[i + j + half];
                float tr = xr * wr - xi * wi, ti = xr * wi + xi * wr;
                re[i + j + half] = re[i + j] - tr;
                im[i + j + half] = im[i + j] - ti;
                re[i + j] += tr;
                im[i + j] += ti;
            }
        }
    }
    if (inverse) {
        for (int i = 0; i < FFT_N; i++) { re[i] /= (float)FFT_N; im[i] /= (float)FFT_N; }
    }
}

void fft512_forward(float *re, float *im) { fft_core(re, im, 0); }
void fft512_inverse(float *re, float *im) { fft_core(re, im, 1); }
