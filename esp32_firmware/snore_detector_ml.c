/*
 * 酣眠 SnoozMate · TinyML 鼾声检测（ESP32-S3 TFLite Micro）
 * ======================================================
 * 
 * 依赖:
 *   - tensorflow/lite/micro/all_ops_resolver.h
 *   - tensorflow/lite/micro/micro_interpreter.h
 *   - tensorflow/lite/micro/micro_error_reporter.h
 *   - snore_model.h （训练脚本自动生成的模型头文件）
 *
 * 与规则版配合使用（混合模式）：
 *   sd_ml_init()        → 初始化模型
 *   sd_ml_infer()       → 推理一帧（返回鼾声概率）
 *
 * 内存需求:
 *   - 模型权重: ~80 KB (int8)
 *   - 推理张量 arena: ~30-50 KB
 *   - 总计: ~150 KB
 *   ESP32-S3 有 512KB SRAM，够用
 */

#include "snore_detector.h"
#include "snore_model.h"  // 自动生成的模型

// TFLite Micro 头文件（根据实际项目路径调整）
// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h"

// ============================================================
// 配置
// ============================================================

#define SD_ML_INPUT_FRAMES    61    // 时间帧数（1秒 @32ms 帧长 × 滑窗 = 约61帧）
#define SD_MEL_BINS          30    // mel 频带数
#define SD_ML_ARENA_SIZE     (60 * 1024)  // arena 大小（根据模型调整）

// ============================================================
// 全局状态
// ============================================================

typedef struct {
    bool initialized;
    
    // TFLite 对象
    // tflite::MicroErrorReporter* error_reporter;
    // tflite::AllOpsResolver* ops_resolver;
    // tflite::MicroInterpreter* interpreter;
    
    // 输入输出张量指针
    // TfLiteTensor* input;
    // TfLiteTensor* output;
    
    // arena 内存（静态分配，避免 malloc）
    // uint8_t tensor_arena[SD_ML_ARENA_SIZE];
    
    // 特征缓存（累积多帧频谱）
    float feature_buffer[SD_ML_INPUT_FRAMES * SD_MEL_BINS];
    int feature_idx;
    int feature_count;
    
    // 滑窗
    float mel_frame[SD_MEL_BINS];  // 当前帧的 mel
    
    // 统计
    uint32_t total_inferences;
    uint32_t total_time_us;
    
} sd_ml_state_t;

static sd_ml_state_t g_ml_state = {0};


// ============================================================
// 特征提取（log-mel）
// ============================================================

/*
 * 在 ESP32 上计算 mel 频谱图。
 * 
 * 选项 1: 使用 esp-dsp 的 FFT + 自己算 mel 滤波器组（推荐）
 * 选项 2: 使用 TFLite Micro 的 microfrontend（alek6kun 的方案）
 * 
 * 这里给出选项 1 的参考实现。
 * 实际部署时建议用 esp-dsp 库的 dsps_fft2r_fc32 + dsps_mul_f32 做硬件加速。
 */

static void compute_mel_from_fft(float *fft_mag, float *mel_out, int n_fft_bins, int sr) {
    /*
     * 将 FFT 幅度谱转换为 mel 频谱
     * mel 滤波器组权重在初始化时预计算
     * 
     * 简化示意：
     *   for (int m = 0; m < SD_MEL_BINS; m++) {
     *       mel_out[m] = 0;
     *       for (int k = mel_start[m]; k < mel_end[m]; k++) {
     *           mel_out[m] += fft_mag[k] * mel_weight[m][k];
     *       }
     *   }
     * 
     * 实际部署时预计算 mel_weight 矩阵（只算一次）。
     */
    
    // 占位实现：直接取对应频段的能量平均
    float hz_per_bin = (float)sr / 2.0f / n_fft_bins;
    float min_hz = 40.0f;
    float max_hz = 6000.0f;
    
    int min_bin = (int)(min_hz / hz_per_bin);
    int max_bin = (int)(max_hz / hz_per_bin);
    if (max_bin > n_fft_bins) max_bin = n_fft_bins;
    
    int bins_per_mel = (max_bin - min_bin) / SD_MEL_BINS;
    if (bins_per_mel < 1) bins_per_mel = 1;
    
    for (int m = 0; m < SD_MEL_BINS; m++) {
        float sum = 0;
        int start = min_bin + m * bins_per_mel;
        int end = start + bins_per_mel;
        if (end > max_bin) end = max_bin;
        for (int k = start; k < end; k++) {
            sum += fft_mag[k];
        }
        mel_out[m] = sum / bins_per_mel;
    }
}

static void apply_log_and_normalize(float *mel) {
    /* Log + 归一化到 [0, 1] */
    float min_val = 1e10f;
    float max_val = -1e10f;
    
    for (int i = 0; i < SD_MEL_BINS; i++) {
        mel[i] = logf(mel[i] + 1e-6f);
        if (mel[i] < min_val) min_val = mel[i];
        if (mel[i] > max_val) max_val = mel[i];
    }
    
    float range = max_val - min_val;
    if (range < 0.01f) range = 1.0f;
    
    for (int i = 0; i < SD_MEL_BINS; i++) {
        mel[i] = (mel[i] - min_val) / range;
    }
}


// ============================================================
// 公共 API
// ============================================================

bool sd_ml_init(void) {
    if (g_ml_state.initialized) return true;
    
    /*
     * TFLite Micro 初始化代码：
     * 
     *   static tflite::MicroErrorReporter error_reporter;
     *   static tflite::AllOpsResolver ops_resolver;
     *   
     *   const tflite::Model* model = tflite::GetModel(g_snore_model);
     *   if (model->version() != TFLITE_SCHEMA_VERSION) {
     *       TF_LITE_REPORT_ERROR(&error_reporter, "Schema mismatch");
     *       return false;
     *   }
     *   
     *   static tflite::MicroInterpreter interpreter(
     *       model, ops_resolver, 
     *       g_ml_state.tensor_arena, SD_ML_ARENA_SIZE,
     *       &error_reporter
     *   );
     *   
     *   interpreter.AllocateTensors();
     *   
     *   g_ml_state.input = interpreter.input(0);
     *   g_ml_state.output = interpreter.output(0);
     *   g_ml_state.interpreter = &interpreter;
     *   g_ml_state.error_reporter = &error_reporter;
     */
    
    // 初始化特征缓存
    g_ml_state.feature_idx = 0;
    g_ml_state.feature_count = 0;
    g_ml_state.total_inferences = 0;
    g_ml_state.total_time_us = 0;
    
    g_ml_state.initialized = true;
    return true;
}

bool sd_ml_is_initialized(void) {
    return g_ml_state.initialized;
}

void sd_ml_reset(void) {
    g_ml_state.feature_idx = 0;
    g_ml_state.feature_count = 0;
}

/*
 * 喂入一帧的 mel 特征。
 * 当累积到 SD_ML_INPUT_FRAMES 帧时，自动执行推理。
 * 返回: true = 完成了一次推理，结果在 prob 中
 *       false = 还在累积
 */
bool sd_ml_feed_frame(const float *mel_frame, float *prob) {
    if (!g_ml_state.initialized) return false;
    
    // 存入特征缓冲（循环队列方式）
    memcpy(&g_ml_state.feature_buffer[g_ml_state.feature_idx * SD_MEL_BINS], 
           mel_frame, SD_MEL_BINS * sizeof(float));
    
    g_ml_state.feature_idx = (g_ml_state.feature_idx + 1) % SD_ML_INPUT_FRAMES;
    g_ml_state.feature_count++;
    
    // 累积足够帧数后推理
    if (g_ml_state.feature_count < SD_ML_INPUT_FRAMES) {
        return false;
    }
    
    // 执行推理
    float snore_prob = 0;
    /*
     * 推理代码：
     * 
     *   // 准备输入（int8 量化模型需要量化）
     *   int8_t* input_data = g_ml_state.input->data.int8;
     *   
     *   for (int i = 0; i < SD_ML_INPUT_FRAMES * SD_MEL_BINS; i++) {
     *       float val = g_ml_state.feature_buffer[i];
     *       // 量化：real = (quant - zero_point) * scale
     *       int quant_val = (int)(val / input_scale + input_zero_point);
     *       input_data[i] = (int8_t)quant_val;
     *   }
     *   
     *   uint32_t t_start = esp_timer_get_time();
     *   TfLiteStatus invoke_status = interpreter.Invoke();
     *   uint32_t t_end = esp_timer_get_time();
     *   
     *   g_ml_state.total_time_us += (t_end - t_start);
     *   g_ml_state.total_inferences++;
     *   
     *   // 读取输出
     *   int8_t output_val = g_ml_state.output->data.int8[0];
     *   float output_scale = g_ml_state.output->params.scale;
     *   int output_zero = g_ml_state.output->params.zero_point;
     *   snore_prob = (output_val - output_zero) * output_scale;
     */
    
    // 占位：返回 0（实际推理后替换）
    *prob = 0.0f;
    
    return true;
}

/*
 * 直接对一段音频做推理（完整 1 秒窗口）。
 * 输入: 16kHz、1秒的音频（16000 采样点）
 * 输出: 鼾声概率 0~1
 */
float sd_ml_infer_full(int16_t *audio_samples) {
    if (!g_ml_state.initialized) return 0.0f;
    
    /*
     * 1. 音频 → 分帧 → FFT → mel 频谱图
     * 2. 填入输入张量
     * 3. Invoke
     * 4. 读取输出概率
     * 
     * 详细实现参考 alek6kun/snore-recognition 的 feature_provider.cc
     */
    
    return 0.0f;
}

/*
 * 获取平均推理时间（微秒）
 */
uint32_t sd_ml_get_avg_inference_us(void) {
    if (g_ml_state.total_inferences == 0) return 0;
    return g_ml_state.total_time_us / g_ml_state.total_inferences;
}

uint32_t sd_ml_get_total_inferences(void) {
    return g_ml_state.total_inferences;
}
