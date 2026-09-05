/*
 * 酣眠 SnoozMate · 手写 CNN 推理 (ESP32-S3)
 * ==========================================
 * 不依赖 TFLite，直接用手写卷积实现
 * 权重从 snore_model_weights.h 加载
 *
 * 架构:
 *   Input(63,30) → Reshape(63,30,1) → Resize(32,30) 
 *   → Conv2D(32,3x3) → MaxPool(2x2) → Conv2D(64,3x3) → MaxPool(2x2)
 *   → Conv2D(128,3x3) → MaxPool(2x2) → GAP(4,3,128) → Dense(128) → Dense(1)
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "snore_model_weights.h"

// 输入尺寸
#define INPUT_HEIGHT 63
#define INPUT_WIDTH 30
#define MEL_BINS 30
#define TIME_FRAMES 63

// 中间层尺寸
#define RESIZE_H 32
#define RESIZE_W 30
#define CONv32_H 32
#define CONv32_W 30
#define CONv32_OUT 32
#define POOL1_H 16
#define POOL1_W 15
#define POOL1_OUT 32

#define CONV2_H 16
#define CONV2_W 15
#define CONV2_OUT 64
#define POOL2_H 8
#define POOL2_W 7
#define POOL2_OUT 64

#define CONV3_H 8
#define CONV3_W 7
#define CONV3_OUT 128
#define POOL3_H 4
#define POOL3_W 3
#define POOL3_OUT 128

#define FC1_DIM 128
#define FC2_DIM 1

// 工作缓冲区
static float input_buf[INPUT_HEIGHT * INPUT_WIDTH];
static float pool3_buf[POOL3_H * POOL3_W * POOL3_OUT];
static float fc1_buf[FC1_DIM];
static float fc2_buf[FC2_DIM];

/**
 * 手动 Conv2D + Relu
 * in: [H, W, C_in], kernel: [C_out, C_in, kh, kw], bias: [C_out]
 * out: [H_out, W_out, C_out]
 */
static void conv2d_relu(const float* in, const float* kernel, const float* bias,
                        int H, int W, int C_in, int C_out,
                        int kh, int kw, int H_out, int W_out, float* out) {
    for (int c = 0; c < C_out; c++) {
        for (int y = 0; y < H_out; y++) {
            for (int x = 0; x < W_out; x++) {
                float sum = bias[c];
                for (int dy = 0; dy < kh; dy++) {
                    for (int dx = 0; dx < kw; dx++) {
                        int iy = y + dy;
                        int ix = x + dx;
                        int inp_idx = (iy * W + ix) * C_in;
                        for (int ci = 0; ci < C_in; ci++) {
                            sum += in[inp_idx + ci] * kernel[(c * C_in + ci) * kh * kw + dy * kw + dx];
                        }
                    }
                }
                // ReLU
                out[(c * H_out + y) * W_out + x] = fmaxf(sum, 0.0f);
            }
        }
    }
}

/**
 * 手动 MaxPooling2D
 */
static void maxpool2d(const float* in, float* out,
                      int H, int W, int C, int pool_h, int pool_w) {
    int H_out = H / pool_h;
    int W_out = W / pool_w;
    for (int c = 0; c < C; c++) {
        for (int y = 0; y < H_out; y++) {
            for (int x = 0; x < W_out; x++) {
                float max_val = -1e30f;
                for (int dy = 0; dy < pool_h; dy++) {
                    for (int dx = 0; dx < pool_w; dx++) {
                        float val = in[((y * pool_h + dy) * W + x * pool_w + dx) * C + c];
                        if (val > max_val) max_val = val;
                    }
                }
                out[(y * W_out + x) * C + c] = max_val;
            }
        }
    }
}

/**
 * Global Average Pooling
 * in: [H, W, C] → out: [C]
 */
static void global_avg_pool(const float* in, float* out, int H, int W, int C) {
    for (int c = 0; c < C; c++) {
        float sum = 0;
        for (int i = 0; i < H * W; i++) {
            sum += in[i * C + c];
        }
        out[c] = sum / (H * W);
    }
}

/**
 * Dense 层
 */
static void dense(const float* in, const float* kernel, const float* bias,
                  int in_dim, int out_dim, float* out, int use_relu) {
    for (int i = 0; i < out_dim; i++) {
        float sum = bias[i];
        for (int j = 0; j < in_dim; j++) {
            sum += in[j] * kernel[i * in_dim + j];
        }
        out[i] = use_relu ? fmaxf(sum, 0.0f) : sum;
    }
}

/**
 * 主推理函数
 * input: log-mel spectrogram [TIME_FRAMES, MEL_BINS] (归一化到 0~1)
 * output: 鼾声概率 0~1
 */
float snore_infer(const float* input) {
    // 1. 复制到输入缓冲区
    memcpy(input_buf, input, TIME_FRAMES * MEL_BINS * sizeof(float));
    
    // 2. Conv1: [63,30,1] → [32,30,32]
    float conv1_buf[CONv32_H * CONv32_W * CONv32_OUT];
    conv2d_relu(input_buf, conv1_kernel, conv1_bias,
                63, 30, 1, CONv32_OUT, 3, 3, 32, 30, conv1_buf);
    
    // 3. MaxPool1: [32,30,32] → [16,15,32]
    float pool1_buf[POOL1_H * POOL1_W * POOL1_OUT];
    maxpool2d(conv1_buf, pool1_buf, 32, 30, 32, 2, 2);
    
    // 4. Conv2: [16,15,32] → [16,15,64]
    float conv2_buf[CONV2_H * CONV2_W * CONV2_OUT];
    conv2d_relu(pool1_buf, conv2_kernel, conv2_bias,
                16, 15, 32, CONV2_OUT, 3, 3, 16, 15, conv2_buf);
    
    // 5. MaxPool2: [16,15,64] → [8,7,64]
    float pool2_buf[POOL2_H * POOL2_W * POOL2_OUT];
    maxpool2d(conv2_buf, pool2_buf, 16, 15, 64, 2, 2);
    
    // 6. Conv3: [8,7,64] → [8,7,128]
    float conv3_buf[CONV3_H * CONV3_W * CONV3_OUT];
    conv2d_relu(pool2_buf, conv3_kernel, conv3_bias,
                8, 7, 64, CONV3_OUT, 3, 3, 8, 7, conv3_buf);
    
    // 7. MaxPool3: [8,7,128] → [4,3,128]
    maxpool2d(conv3_buf, pool3_buf, 8, 7, 128, 2, 2);
    
    // 8. Global Average Pooling: [4,3,128] → [128]
    float gap_buf[FC1_DIM];
    global_avg_pool(pool3_buf, gap_buf, 4, 3, 128);
    
    // 9. Dense1: [128] → [128] + ReLU
    dense(gap_buf, fc1_kernel, fc1_bias, FC1_DIM, FC1_DIM, fc1_buf, 1);
    
    // 10. Dense2: [128] → [1] (sigmoid 在调用方做)
    dense(fc1_buf, fc2_kernel, fc2_bias, FC1_DIM, FC2_DIM, fc2_buf, 0);
    
    return fc2_buf[0];
}

/**
 * Sigmoid 函数
 */
float sigmoid(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/**
 * 带 sigmoid 的版本
 */
float snore_infer_prob(const float* input) {
    float logit = snore_infer(input);
    return sigmoid(logit);
}

// 模型统计
int snore_get_param_count() {
    return 109313;
}