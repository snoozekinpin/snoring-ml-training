/*
 * 酣眠 SnoozMate · 鼾声检测引擎 (ESP32-S3)
 * ========================================
 * 双模式架构：
 *   - MODE_RULE: 规则版（P0，默认启用，功耗最低）
 *   - MODE_TINYML: TinyML CNN 版（P1，更高精度）
 *
 * 规则版算法（三级漏斗）：
 *   Level 1: 能量闸门（<底噪×3 → 跳过）
 *   Level 2: 频谱特征（60-600Hz 能量占比 > 60%）
 *   Level 3: 时间模式（持续 0.3-4s + 周期性出现）
 *
 * 作者：酣眠团队
 */

#ifndef SNORE_DETECTOR_H
#define SNORE_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// 配置参数
// ============================================================

#define SD_SAMPLE_RATE       16000   // 采样率
#define SD_FRAME_MS         32      // 每帧 32ms = 512 采样 @16kHz
#define SD_FRAME_SIZE       (SD_SAMPLE_RATE * SD_FRAME_MS / 1000)  // 512
#define SD_N_FFT           SD_FRAME_SIZE
#define SD_FREQ_RES        (SD_SAMPLE_RATE / SD_N_FFT)  // ~31.25 Hz

// 鼾声频率范围
#define SD_SNORE_FREQ_MIN  60      // Hz
#define SD_SNORE_FREQ_MAX  600     // Hz

// 检测参数
#define SD_ENERGY_THRESHOLD  3.0f   // 能量 > 底噪 × 倍数
#define SD_SPECTRAL_RATIO   0.55f  // 鼾声频段能量占比
#define SD_MIN_DURATION_MS  300    // 最小持续时间
#define SD_MAX_DURATION_MS  5000   // 最大持续时间
#define SD_MIN_PERIOD_MS    1500   // 最小呼吸周期
#define SD_MAX_PERIOD_MS    7000   // 最大呼吸周期

// 模式
typedef enum {
    SD_MODE_RULE = 0,       // 规则版（低功耗）
    SD_MODE_TINYML,         // TinyML CNN（高精度）
    SD_MODE_HYBRID,         // 混合（规则过滤 + ML确认）
} sd_mode_t;

// 检测结果
typedef struct {
    bool is_snore;          // 是否鼾声
    float confidence;       // 置信度 0~1
    float energy_db;        // 能量(dB)
    float spectral_ratio;   // 鼾声频段占比
    float fundamental_hz;   // 基频估计
    uint32_t duration_ms;   // 已持续时间
} sd_result_t;

// 检测器状态
typedef struct {
    sd_mode_t mode;
    bool in_snore_event;
    uint32_t snore_start_ms;
    uint32_t current_ms;
    
    // 底噪估计（指数平均）
    float noise_floor_db;
    float noise_alpha;
    
    // 状态机
    int state;  // 0=idle, 1=candidate, 2=confirmed
    
    // 帧缓存
    int16_t frame_buffer[SD_FRAME_SIZE];
    int frame_idx;
    
    // FFT 工作区（预分配）
    float fft_buffer[SD_N_FFT];
    float mag_buffer[SD_N_FFT / 2 + 1];
    
    // 历史帧统计（用于周期性判断）
    float energy_history[64];  // 64帧 × 32ms = ~2秒
    int history_idx;
    
    // 事件计数
    uint32_t total_snore_events;
    uint32_t total_false_alarms;
    
    // TinyML 推理计数（混合模式用）
    uint32_t ml_invoke_count;
    
} snore_detector_t;

// ============================================================
// API
// ============================================================

/**
 * 初始化检测器
 */
void sd_init(snore_detector_t *det, sd_mode_t mode);

/**
 * 处理一批音频采样（int16, mono, 16kHz）
 * 返回这一批内检测到的鼾声事件数
 */
int sd_process(snore_detector_t *det, const int16_t *samples, int n_samples);

/**
 * 处理单帧（512采样），返回检测结果
 */
sd_result_t sd_process_frame(snore_detector_t *det, const int16_t *frame);

/**
 * 设置检测模式
 */
void sd_set_mode(snore_detector_t *det, sd_mode_t mode);

/**
 * 重置检测器状态（新的一夜开始时调用）
*/
void sd_reset(snore_detector_t *det);

/**
 * 获取当前底噪估计 (dB)
 */
float sd_get_noise_floor(snore_detector_t *det);

/**
 * 检测器统计信息字符串（调试用）
 */
void sd_get_stats_str(snore_detector_t *det, char *buf, int buf_len);

#ifdef __cplusplus
}
#endif

#endif // SNORE_DETECTOR_H
