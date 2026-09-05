/*
 * 酣眠 SnoozMate · ESP32-S3 主程序（完整参考实现）
 * =================================================
 * 
 * 功能:
 *   - I2S 麦克风采集 (INMP441)
 *   - 鼾声检测（规则版 + TinyML 双模式）
 *   - 振动片控制（PWM + 档位调节）
 *   - LED 状态指示（WS2812 / 板载LED）
 *   - Wi-Fi + 云端事件上报
 *   - 安全状态机（冷却时间/最大轮次/最大时长）
 *
 * 硬件连接（ESP32-S3）:
 *   INMP441 麦克风:
 *     VDD → 3.3V
 *     GND → GND
 *     SCK → GPIO 14  (I2S BCLK)
 *     WS  → GPIO 15  (I2S LRCLK)
 *     SD  → GPIO 32  (I2S DATA IN)
 *     L/R → GND
 *   
 *   振动片 PWM:
 *     PWM+ → GPIO 18
 *     PWM- → GND
 *
 *   WS2812 LED:
 *     DATA → GPIO 19
 *     VCC  → 5V / 3.3V
 *     GND  → GND
 *
 *   雷达 LD2410 (可选):
 *     TX → GPIO 16 (UART2 RX)
 *     RX → GPIO 17 (UART2 TX)
 *     VCC → 5V
 *     GND → GND
 *
 * 编译:
 *   - 使用 ESP-IDF v5.0+
 *   - 依赖: esp-dsp, esptool_py
 *   - 启用 TFLite Micro 时需要 tflite-micro 组件
 *
 * 作者: 酣眠团队
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_system.h"
#include "esp_log.h"
#include "driver/i2s.h"
#include "driver/ledc.h"
#include "nvs_flash.h"

#include "snore_detector.h"

// ============================================================
// 配置
// ============================================================

#define TAG "SNOOZMATE"

// I2S 配置
#define I2S_PORT          I2S_NUM_0
#define I2S_SAMPLE_RATE   16000
#define I2S_SAMPLE_BITS   32       // INMP441 是 24bit，但用 32bit 读
#define I2S_DMA_BUF_COUNT 8
#define I2S_DMA_BUF_LEN   1024     // 每次读 1024 采样 = 64ms

// GPIO 定义
#define GPIO_MIC_BCK      14
#define GPIO_MIC_WS       15
#define GPIO_MIC_DATA     32
#define GPIO_VIBRATION    18       // 振动片 PWM
#define GPIO_LED_DATA     19       // WS2812

// 振动 PWM 配置
#define VIB_PWM_TIMER     LEDC_TIMER_0
#define VIB_PWM_MODE      LEDC_LOW_SPEED_MODE
#define VIB_PWM_CHANNEL   LEDC_CHANNEL_0
#define VIB_PWM_FREQ      180      // 180Hz（Pacinian小体最敏感）
#define VIB_PWM_RES       LEDC_TIMER_10_BIT  // 10位 = 1024 级

// 检测配置
#define DETECT_MODE       SD_MODE_HYBRID  // 混合模式
#define MIN_COOLDOWN_MS   60000    // 最小冷却 60 秒
#define MAX_ROUNDS_PER_NIGHT  15   // 每晚最多 15 轮
#define MAX_VIB_DURATION_MS  120000 // 单轮最长 2 分钟

// ============================================================
// 全局状态
// ============================================================

typedef enum {
    MODE_IDLE = 0,        // 待机/白天
    MODE_CALIBRATING,     // 底噪校准中
    MODE_MONITORING,      // 监测中（未检测到鼾声）
    MODE_INTERVENING,     // 干预中（振动正在进行）
    MODE_COOLDOWN,        // 冷却中
} app_mode_t;

typedef struct {
    app_mode_t mode;
    snore_detector_t detector;
    
    // 干预计数
    int rounds_tonight;
    int current_level;
    int max_level;
    uint32_t intervention_start_ms;
    uint32_t cooldown_end_ms;
    
    // 事件记录
    uint32_t last_snore_ms;
    
    // Wi-Fi / 云端（可选）
    bool wifi_connected;
    char device_id[32];
    
    // 统计
    uint32_t snore_events_total;
    uint32_t intervention_success;
    uint32_t uptime_seconds;
    
} app_state_t;

static app_state_t g_state;

// ============================================================
// I2S 麦克风初始化
// ============================================================

static void mic_init(void) {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = I2S_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = I2S_DMA_BUF_COUNT,
        .dma_buf_len = I2S_DMA_BUF_LEN,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = GPIO_MIC_BCK,
        .ws_io_num = GPIO_MIC_WS,
        .data_out_num = -1,
        .data_in_num = GPIO_MIC_DATA
    };
    
    ESP_ERROR_CHECK(i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL));
    ESP_ERROR_CHECK(i2s_set_pin(I2S_PORT, &pin_config));
    
    ESP_LOGI(TAG, "麦克风初始化完成 (I2S %d Hz, %d bit)", I2S_SAMPLE_RATE, I2S_SAMPLE_BITS);
}

// ============================================================
// 振动片 PWM 初始化
// ============================================================

static void vibration_init(void) {
    ledc_timer_config_t timer_cfg = {
        .speed_mode = VIB_PWM_MODE,
        .timer_num = VIB_PWM_TIMER,
        .duty_resolution = VIB_PWM_RES,
        .freq_hz = VIB_PWM_FREQ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer_cfg));
    
    ledc_channel_config_t ch_cfg = {
        .gpio_num = GPIO_VIBRATION,
        .speed_mode = VIB_PWM_MODE,
        .channel = VIB_PWM_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = VIB_PWM_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ch_cfg));
    
    ESP_LOGI(TAG, "振动片 PWM 初始化完成 (%d Hz)", VIB_PWM_FREQ);
}

// 控制振动强度 (0.0 ~ 1.0)
static void vibration_set(float level) {
    if (level < 0) level = 0;
    if (level > 1.0) level = 1.0;
    
    uint32_t duty = (uint32_t)(level * 1023);  // 10位分辨率
    ledc_set_duty(VIB_PWM_MODE, VIB_PWM_CHANNEL, duty);
    ledc_update_duty(VIB_PWM_MODE, VIB_PWM_CHANNEL);
}

// 档位到强度映射 (1~3 档)
static float level_to_intensity(int level) {
    // 1档=20%, 2档=40%, 3档=70%
    float mapping[] = {0, 0.2f, 0.4f, 0.7f};
    if (level < 1) level = 1;
    if (level > 3) level = 3;
    return mapping[level];
}

// ============================================================
// 状态机
// ============================================================

static void app_state_init(void) {
    memset(&g_state, 0, sizeof(g_state));
    g_state.mode = MODE_CALIBRATING;
    g_state.max_level = 3;
    g_state.current_level = 1;
    
    sd_init(&g_state.detector, DETECT_MODE);
    
    strcpy(g_state.device_id, "esp32-s3-snoozmate-001");
    
    ESP_LOGI(TAG, "状态机初始化完成，模式: %d", DETECT_MODE);
}

static void start_intervention(int level) {
    if (g_state.rounds_tonight >= MAX_ROUNDS_PER_NIGHT) {
        ESP_LOGW(TAG, "已达每晚最大轮次 (%d)，停止干预", MAX_ROUNDS_PER_NIGHT);
        g_state.mode = MODE_COOLDOWN;
        g_state.cooldown_end_ms = xTaskGetTickCount() * portTICK_PERIOD_MS + 3600000; // 1小时
        return;
    }
    
    g_state.mode = MODE_INTERVENING;
    g_state.current_level = level;
    g_state.intervention_start_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
    g_state.rounds_tonight++;
    
    float intensity = level_to_intensity(level);
    vibration_set(intensity);
    
    ESP_LOGI(TAG, "开始干预: 档位 %d (强度 %.0f%%), 第 %d 轮", 
             level, intensity * 100, g_state.rounds_tonight);
    
    // TODO: 上报云端 vibration_start 事件
}

static void stop_intervention(bool success) {
    vibration_set(0);
    
    uint32_t now = xTaskGetTickCount() * portTICK_PERIOD_MS;
    uint32_t duration_ms = now - g_state.intervention_start_ms;
    
    if (success) {
        g_state.intervention_success++;
    }
    
    g_state.mode = MODE_COOLDOWN;
    // 冷却时间随档位增加（1档=60s, 2档=120s, 3档=180s）
    uint32_t cooldown_ms = MIN_COOLDOWN_MS * g_state.current_level;
    g_state.cooldown_end_ms = now + cooldown_ms;
    
    ESP_LOGI(TAG, "结束干预: %s, 持续 %d ms, 冷却 %d s",
             success ? "成功" : "未成功",
             duration_ms, cooldown_ms / 1000);
    
    // TODO: 上报云端 vibration_stop 事件
}

// ============================================================
// 音频处理任务
// ============================================================

static void audio_task(void *pvParameters) {
    int32_t i2s_buf[I2S_DMA_BUF_LEN];  // 32bit 采样
    int16_t audio_buf[I2S_DMA_BUF_LEN];  // 转 16bit
    size_t bytes_read;
    
    ESP_LOGI(TAG, "音频处理任务启动");
    
    while (1) {
        // 读取一帧 DMA 数据
        esp_err_t ret = i2s_read(I2S_PORT, i2s_buf, sizeof(i2s_buf), 
                                 &bytes_read, portMAX_DELAY);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "I2S 读取失败: %s", esp_err_to_name(ret));
            continue;
        }
        
        int n_samples = bytes_read / sizeof(int32_t);
        
        // INMP441 是 24bit 左对齐，转 16bit
        for (int i = 0; i < n_samples; i++) {
            audio_buf[i] = (int16_t)(i2s_buf[i] >> 16);  // 取高 16 位
        }
        
        // 送入鼾声检测器
        int events = sd_process(&g_state.detector, audio_buf, n_samples);
        
        if (events > 0) {
            g_state.snore_events_total++;
            
            uint32_t now_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
            
            // 状态机逻辑
            switch (g_state.mode) {
                case MODE_MONITORING:
                    // 检测到鼾声 → 开始干预（1档起步）
                    if (g_state.rounds_tonight < MAX_ROUNDS_PER_NIGHT) {
                        start_intervention(1);
                    }
                    break;
                    
                case MODE_INTERVENING:
                    // 干预中仍有鼾声 → 考虑升级档位
                    if (now_ms - g_state.intervention_start_ms > 10000) {  // 10秒仍未停
                        if (g_state.current_level < g_state.max_level) {
                            g_state.current_level++;
                            vibration_set(level_to_intensity(g_state.current_level));
                            ESP_LOGI(TAG, "升级档位到 %d", g_state.current_level);
                        }
                    }
                    
                    // 超时保护
                    if (now_ms - g_state.intervention_start_ms > MAX_VIB_DURATION_MS) {
                        stop_intervention(false);
                    }
                    break;
                    
                case MODE_CALIBRATING:
                    // 校准中，忽略
                    break;
                    
                case MODE_COOLDOWN:
                    // 冷却中，忽略（除非模式切换）
                    break;
                    
                default:
                    break;
            }
            
            g_state.last_snore_ms = now_ms;
        }
        
        // 冷却时间结束检查
        if (g_state.mode == MODE_COOLDOWN) {
            uint32_t now_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
            if (now_ms >= g_state.cooldown_end_ms) {
                g_state.mode = MODE_MONITORING;
                ESP_LOGI(TAG, "冷却结束，回到监测模式");
            }
        }
        
        // 干预中长时间无鼾声 → 判定成功
        if (g_state.mode == MODE_INTERVENING) {
            uint32_t now_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
            if (now_ms - g_state.last_snore_ms > 5000) {  // 5秒没鼾声
                stop_intervention(true);
            }
        }
    }
}

// ============================================================
// 主函数
// ============================================================

void app_main(void) {
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, "  酣眠 SnoozMate 固件启动");
    ESP_LOGI(TAG, "  设备: %s", g_state.device_id);
    ESP_LOGI(TAG, "  芯片: %s, Rev %d", 
             esp_chip_info()->model == CHIP_ESP32S3 ? "ESP32-S3" : "ESP32",
             esp_chip_info()->revision);
    ESP_LOGI(TAG, "================================");
    
    // 初始化 NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    
    // 初始化外设
    mic_init();
    vibration_init();
    // ws2812_init();  // LED 可选
    // radar_init();   // 雷达 可选
    // wifi_init();    // Wi-Fi 可选
    
    // 初始化状态机
    app_state_init();
    
    // 启动音频处理任务
    xTaskCreate(audio_task, "audio_task", 16384, NULL, 5, NULL);
    
    ESP_LOGI(TAG, "系统就绪，开始校准底噪...");
    
    // 校准 10 秒后进入监测模式
    vTaskDelay(pdMS_TO_TICKS(10000));
    g_state.mode = MODE_MONITORING;
    ESP_LOGI(TAG, "底噪校准完成，进入监测模式");
    
    // 主循环（状态输出 + 云端同步）
    char stats_buf[256];
    while (1) {
        sd_get_stats_str(&g_state.detector, stats_buf, sizeof(stats_buf));
        ESP_LOGI(TAG, "状态: mode=%d  rounds=%d  level=%d  %s",
                 g_state.mode, g_state.rounds_tonight, 
                 g_state.current_level, stats_buf);
        
        // TODO: 批量事件上云
        // TODO: Wi-Fi 重连
        // TODO: OTA 检查
        
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
