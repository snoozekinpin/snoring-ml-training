/*
 * 酣眠 SnoozMate · 鼾声检测引擎实现（规则版）
 * ==========================================
 * 三级漏斗 + 状态机 + 周期性验证
 *
 * 优化点：
 *   - FFT 用 esp-dsp (ESP32-S3 有硬件加速)
 *   - 全部使用定点/浮点混合，避免 malloc
 *   - 底噪自适应（夜间环境会变化）
 */

#include "snore_detector.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

// ============================================================
// 工具函数
// ============================================================

static float int16_to_float_db(const int16_t *samples, int n) {
    // 计算 RMS 能量，转 dB
    float sum_sq = 0;
    for (int i = 0; i < n; i++) {
        float s = samples[i] / 32768.0f;
        sum_sq += s * s;
    }
    float rms = sqrtf(sum_sq / n);
    return 20.0f * log10f(rms + 1e-10f);
}

static float linear_to_db(float linear) {
    return 20.0f * log10f(linear + 1e-10f);
}

// 简化 FFT（实际部署用 esp-dsp 的 dsps_fft2r_fc32）
// 这里提供参考实现，性能要求高时换硬件加速版本
static void simple_fft(float *buffer, int n, bool inverse) {
    // Cooley-Tukey radix-2 FFT
    // 输入输出都在 buffer 里，格式: [re0, im0, re1, im1, ...]
    // 注意：实际项目请用 esp-dsp 库的硬件加速 FFT
    
    // 位反转排序
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = buffer[2*i];
            float ti = buffer[2*i + 1];
            buffer[2*i] = buffer[2*j];
            buffer[2*i + 1] = buffer[2*j + 1];
            buffer[2*j] = tr;
            buffer[2*j + 1] = ti;
        }
    }
    
    // 蝶形运算
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * 3.14159265358979323846f / len * (inverse ? 1 : -1);
        float wlen_r = cosf(ang);
        float wlen_i = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                float u_r = buffer[2*(i+j)];
                float u_i = buffer[2*(i+j) + 1];
                float v_r = buffer[2*(i+j+len/2)] * w_r - buffer[2*(i+j+len/2)+1] * w_i;
                float v_i = buffer[2*(i+j+len/2)] * w_i + buffer[2*(i+j+len/2)+1] * w_r;
                buffer[2*(i+j)] = u_r + v_r;
                buffer[2*(i+j) + 1] = u_i + v_i;
                buffer[2*(i+j+len/2)] = u_r - v_r;
                buffer[2*(i+j+len/2) + 1] = u_i - v_i;
                float tw_r = w_r * wlen_r - w_i * wlen_i;
                float tw_i = w_r * wlen_i + w_i * wlen_r;
                w_r = tw_r;
                w_i = tw_i;
            }
        }
    }
    
    if (inverse) {
        for (int i = 0; i < 2*n; i++)
            buffer[i] /= n;
    }
}

// 计算功率谱密度
static void compute_magnitude(const float *fft_buf, float *mag, int n_fft) {
    for (int i = 0; i <= n_fft / 2; i++) {
        float re = fft_buf[2*i];
        float im = fft_buf[2*i + 1];
        mag[i] = sqrtf(re*re + im*im) / (n_fft / 2);
    }
}

// ============================================================
// 特征提取
// ============================================================

// 计算 60-600Hz 频段能量占比
static float compute_snore_band_ratio(const float *mag, int n_bins, float sr) {
    float total_energy = 0;
    float snore_energy = 0;
    float hz_per_bin = sr / 2.0f / n_bins;
    
    int min_bin = (int)(SD_SNORE_FREQ_MIN / hz_per_bin);
    int max_bin = (int)(SD_SNORE_FREQ_MAX / hz_per_bin);
    if (max_bin > n_bins) max_bin = n_bins;
    
    for (int i = 1; i < n_bins; i++) {  // i=0 是DC，跳过
        float e = mag[i] * mag[i];  // 功率
        total_energy += e;
        if (i >= min_bin && i <= max_bin) {
            snore_energy += e;
        }
    }
    
    if (total_energy < 1e-12f) return 0;
    return snore_energy / total_energy;
}

// 估计基频（简化版：找 60-600Hz 范围内最强峰值）
static float estimate_fundamental(const float *mag, int n_bins, float sr) {
    float hz_per_bin = sr / 2.0f / n_bins;
    int min_bin = (int)(SD_SNORE_FREQ_MIN / hz_per_bin);
    int max_bin = (int)(SD_SNORE_FREQ_MAX / hz_per_bin);
    if (max_bin > n_bins) max_bin = n_bins;
    
    float max_mag = 0;
    int max_idx = 0;
    for (int i = min_bin; i < max_bin; i++) {
        if (mag[i] > max_mag) {
            max_mag = mag[i];
            max_idx = i;
        }
    }
    
    return max_idx * hz_per_bin;
}

// 检测周期性（自相关简化版）
static bool check_periodicity(float *energy_history, int history_len, 
                              float min_period_ms, float max_period_ms, float frame_ms) {
    // 在历史能量序列中找显著的周期性峰值
    // 简化：找能量峰值之间的间隔是否在呼吸周期范围内
    
    int min_lag = (int)(min_period_ms / frame_ms);
    int max_lag = (int)(max_period_ms / frame_ms);
    if (max_lag >= history_len) max_lag = history_len - 1;
    if (min_lag < 1) min_lag = 1;
    
    // 找局部极大值
    int peak_count = 0;
    int last_peak_idx = -1000;
    float avg_energy = 0;
    for (int i = 0; i < history_len; i++) avg_energy += energy_history[i];
    avg_energy /= history_len;
    
    float threshold = avg_energy * 1.2f;  // 比平均高 20% 才算峰
    
    for (int i = 2; i < history_len - 2; i++) {
        if (energy_history[i] > threshold &&
            energy_history[i] > energy_history[i-1] &&
            energy_history[i] > energy_history[i+1] &&
            energy_history[i] > energy_history[i-2] &&
            energy_history[i] > energy_history[i+2]) {
            
            if (last_peak_idx >= 0) {
                int interval = i - last_peak_idx;
                if (interval >= min_lag && interval <= max_lag) {
                    peak_count++;
                }
            }
            last_peak_idx = i;
        }
    }
    
    // 至少 2 个符合周期的峰才算有周期性
    return peak_count >= 2;
}

// ============================================================
// 主 API 实现
// ============================================================

void sd_init(snore_detector_t *det, sd_mode_t mode) {
    memset(det, 0, sizeof(snore_detector_t));
    det->mode = mode;
    det->noise_floor_db = -60.0f;  // 初始底噪估计
    det->noise_alpha = 0.02f;      // 底噪更新速度（越小越慢）
    det->state = 0;                // idle
    det->history_idx = 0;
    
    // 初始化历史为 0
    for (int i = 0; i < 64; i++) {
        det->energy_history[i] = -80.0f;
    }
}

void sd_reset(snore_detector_t *det) {
    det->in_snore_event = false;
    det->snore_start_ms = 0;
    det->current_ms = 0;
    det->state = 0;
    det->frame_idx = 0;
    det->total_snore_events = 0;
    det->total_false_alarms = 0;
    det->ml_invoke_count = 0;
}

void sd_set_mode(snore_detector_t *det, sd_mode_t mode) {
    det->mode = mode;
}

float sd_get_noise_floor(snore_detector_t *det) {
    return det->noise_floor_db;
}

sd_result_t sd_process_frame(snore_detector_t *det, const int16_t *frame) {
    sd_result_t result = {0};
    
    // 1. 计算能量
    float energy_db = int16_to_float_db(frame, SD_FRAME_SIZE);
    result.energy_db = energy_db;
    
    // 2. 更新底噪（只在能量较低时更新，避免鼾声拉高底噪）
    if (energy_db < det->noise_floor_db + 5.0f) {
        det->noise_floor_db = det->noise_floor_db * (1 - det->noise_alpha) + 
                              energy_db * det->noise_alpha;
    }
    
    // 3. Level 1: 能量闸门
    float snr = energy_db - det->noise_floor_db;
    if (snr < SD_ENERGY_THRESHOLD) {
        det->state = 0;  // 回到 idle
        return result;   // 不够响，直接返回
    }
    
    // 4. Level 2: 频谱分析
    // 准备 FFT 输入
    for (int i = 0; i < SD_N_FFT; i++) {
        if (i < SD_FRAME_SIZE) {
            // 加 Hann 窗
            float window = 0.5f * (1 - cosf(2 * 3.14159f * i / (SD_FRAME_SIZE - 1)));
            det->fft_buffer[2*i] = (frame[i] / 32768.0f) * window;
            det->fft_buffer[2*i + 1] = 0;
        } else {
            det->fft_buffer[2*i] = 0;
            det->fft_buffer[2*i + 1] = 0;
        }
    }
    
    simple_fft(det->fft_buffer, SD_N_FFT, false);
    compute_magnitude(det->fft_buffer, det->mag_buffer, SD_N_FFT);
    
    float spectral_ratio = compute_snore_band_ratio(det->mag_buffer, 
                                                    SD_N_FFT / 2 + 1, SD_SAMPLE_RATE);
    result.spectral_ratio = spectral_ratio;
    result.fundamental_hz = estimate_fundamental(det->mag_buffer, 
                                                  SD_N_FFT / 2 + 1, SD_SAMPLE_RATE);
    
    if (spectral_ratio < SD_SPECTRAL_RATIO) {
        // 频谱不像鼾声（高频多，可能是说话/环境音）
        if (det->state == 1) det->state = 0;
        return result;
    }
    
    // 5. 存入历史
    det->energy_history[det->history_idx] = energy_db;
    det->history_idx = (det->history_idx + 1) % 64;
    
    // 6. Level 3: 状态机 + 持续时间 + 周期性
    det->current_ms += SD_FRAME_MS;
    
    if (det->state == 0) {
        // idle → candidate
        det->state = 1;
        det->snore_start_ms = det->current_ms;
    } else if (det->state == 1) {
        // candidate → 检查持续时间
        uint32_t dur = det->current_ms - det->snore_start_ms;
        result.duration_ms = dur;
        
        if (dur > SD_MIN_DURATION_MS && dur < SD_MAX_DURATION_MS) {
            // 持续时间够了，检查周期性
            bool periodic = check_periodicity(
                det->energy_history, 64,
                SD_MIN_PERIOD_MS, SD_MAX_PERIOD_MS,
                SD_FRAME_MS
            );
            
            if (periodic) {
                det->state = 2;  // confirmed
                result.is_snore = true;
                result.confidence = 0.7f + (spectral_ratio - 0.55f) * 0.5f;
                if (result.confidence > 0.95f) result.confidence = 0.95f;
                det->total_snore_events++;
            }
        }
        
        if (dur > SD_MAX_DURATION_MS) {
            det->state = 0;  // 太长了，不是鼾声
        }
    } else if (det->state == 2) {
        // confirmed，继续累积
        uint32_t dur = det->current_ms - det->snore_start_ms;
        result.duration_ms = dur;
        result.is_snore = true;
        result.confidence = 0.85f;
        
        if (dur > SD_MAX_DURATION_MS) {
            det->state = 0;
        }
    }
    
    // 7. 混合模式：如果规则版检测到了，再用 TinyML 确认
    // （TinyML 推理实现在 snore_detector_ml.c 中）
    
    return result;
}

int sd_process(snore_detector_t *det, const int16_t *samples, int n_samples) {
    int events_detected = 0;
    
    for (int i = 0; i < n_samples; i++) {
        det->frame_buffer[det->frame_idx++] = samples[i];
        
        if (det->frame_idx >= SD_FRAME_SIZE) {
            det->frame_idx = 0;
            sd_result_t res = sd_process_frame(det, det->frame_buffer);
            if (res.is_snore) {
                events_detected++;
            }
        }
    }
    
    return events_detected;
}

void sd_get_stats_str(snore_detector_t *det, char *buf, int buf_len) {
    snprintf(buf, buf_len,
        "Mode: %d | Noise: %.1f dB | Events: %lu | ML: %lu | State: %d",
        det->mode, det->noise_floor_db, 
        (unsigned long)det->total_snore_events,
        (unsigned long)det->ml_invoke_count,
        det->state);
}
