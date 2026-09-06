/* Usage: test_features <features.bin>.
 * Exit 0 when every case has max |dX| <= 5e-3, no quantised cell differs by more than 1 LSB,
 * and at most 1 % of quantised cells differ by exactly 1 LSB. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "snore_features.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s features.bin\n", argv[0]); return 2; }
    FILE *fh = fopen(argv[1], "rb");
    if (!fh) { perror("open"); return 2; }
    int32_t hdr[3], zp;
    float scale;
    if (fread(hdr, sizeof(int32_t), 3, fh) != 3 || fread(&scale, sizeof(float), 1, fh) != 1 || fread(&zp, sizeof(int32_t), 1, fh) != 1) { fprintf(stderr, "bad header\n"); return 2; }
    if (hdr[1] != SF_WIN || hdr[2] != SF_FEATURE_DIM) { fprintf(stderr, "golden shape mismatch\n"); return 2; }
    int16_t *win = malloc(sizeof(int16_t) * SF_WIN);
    float *ref = malloc(sizeof(float) * SF_FEATURE_DIM);
    float *X = malloc(sizeof(float) * SF_FEATURE_DIM);
    int8_t *qref = malloc(SF_FEATURE_DIM), *q = malloc(SF_FEATURE_DIM);
    double worst = 0.0;
    int failures = 0;
    sf_init();
    for (int c = 0; c < hdr[0]; c++) {
        if (fread(win, sizeof(int16_t), SF_WIN, fh) != (size_t)SF_WIN) return 2;
        if (fread(ref, sizeof(float), SF_FEATURE_DIM, fh) != (size_t)SF_FEATURE_DIM) return 2;
        if (fread(qref, 1, SF_FEATURE_DIM, fh) != (size_t)SF_FEATURE_DIM) return 2;
        sf_compute(win, X);
        sf_quantize(X, q, scale, zp);
        double maxd = 0.0;
        int bad = 0, one_lsb = 0, over = 0;
        for (int i = 0; i < SF_FEATURE_DIM; i++) {
            double d = fabs((double)X[i] - (double)ref[i]);
            if (d > maxd) maxd = d;
            if (d > 5e-3) bad++;
            int dq = abs((int)q[i] - (int)qref[i]);
            if (dq == 1) one_lsb++;
            if (dq > 1) over++;
        }
        if (maxd > worst) worst = maxd;
        int ok = bad == 0 && over == 0 && one_lsb * 100 <= SF_FEATURE_DIM;
        printf("case %2d: max|dX| = %.5f, cells over tolerance = %d, q off by 1 LSB = %d, q off by >1 = %d %s\n", c, maxd, bad, one_lsb, over, ok ? "ok" : "MISMATCH");
        if (!ok) failures++;
    }
    fclose(fh);
    printf("features parity: worst %.5f over %d cases, %d failing\n", worst, hdr[0], failures);
    return failures ? 1 : 0;
}
