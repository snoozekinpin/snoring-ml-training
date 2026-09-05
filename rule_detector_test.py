"""
酣眠 SnoozMate · 鼾声检测规则版（Python验证用）
================================================
PC 端验证版本，算法和 ESP32 固件完全一致。
用来快速验证算法效果、调参数。

用法:
  python rule_detector_test.py test_snore.wav
  python rule_detector_test.py --folder dataset/snore
"""

import argparse
import sys
import numpy as np
from pathlib import Path


# ============================================================
# 参数（和固件 snore_detector.h 保持一致）
# ============================================================
SAMPLE_RATE = 16000
FRAME_MS = 32
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # 512
SNORE_FREQ_MIN = 60
SNORE_FREQ_MAX = 600
ENERGY_THRESHOLD_DB = 3.0   # dB 以上底噪
SPECTRAL_RATIO = 0.55       # 鼾声频段占比
MIN_DURATION_MS = 300
MAX_DURATION_MS = 5000
MIN_PERIOD_MS = 1500
MAX_PERIOD_MS = 7000


def load_wav(filepath):
    """加载 wav 文件，返回 16kHz mono float32"""
    try:
        import librosa
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)
    except ImportError:
        from scipy.io import wavfile
        from scipy.signal import resample
        sr, data = wavfile.read(filepath)
        if data.ndim > 1:
            data = data[:, 0]
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if sr != SAMPLE_RATE:
            n_new = int(len(data) * SAMPLE_RATE / sr)
            data = resample(data, n_new).astype(np.float32)
        return data


def compute_energy_db(frame):
    """计算帧能量 dB"""
    rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
    return 20 * np.log10(rms + 1e-10)


def compute_magnitude_fft(frame):
    """计算 FFT 幅度谱"""
    n_fft = FRAME_SIZE
    windowed = frame * np.hanning(len(frame))
    fft_result = np.fft.rfft(windowed, n_fft)
    mag = np.abs(fft_result) / (n_fft / 2)
    return mag


def compute_snore_band_ratio(mag, sr=SAMPLE_RATE):
    """60-600Hz 频段能量占比"""
    n_bins = len(mag)
    hz_per_bin = sr / 2 / n_bins
    min_bin = int(SNORE_FREQ_MIN / hz_per_bin)
    max_bin = int(SNORE_FREQ_MAX / hz_per_bin)
    max_bin = min(max_bin, n_bins - 1)
    
    power = mag ** 2
    total = np.sum(power[1:])  # 跳过 DC
    snore = np.sum(power[min_bin:max_bin+1])
    
    if total < 1e-12:
        return 0.0
    return snore / total


def estimate_fundamental(mag, sr=SAMPLE_RATE):
    """估计基频"""
    n_bins = len(mag)
    hz_per_bin = sr / 2 / n_bins
    min_bin = int(SNORE_FREQ_MIN / hz_per_bin)
    max_bin = int(SNORE_FREQ_MAX / hz_per_bin)
    max_bin = min(max_bin, n_bins - 1)
    
    peak_bin = np.argmax(mag[min_bin:max_bin+1]) + min_bin
    return peak_bin * hz_per_bin


def check_periodicity(energy_history, min_period_ms, max_period_ms, frame_ms):
    """检测周期性（找能量峰值间隔）"""
    n = len(energy_history)
    min_lag = int(min_period_ms / frame_ms)
    max_lag = int(max_period_ms / frame_ms)
    max_lag = min(max_lag, n - 1)
    if min_lag < 1:
        min_lag = 1
    
    avg = np.mean(energy_history)
    threshold = avg * 1.2
    
    # 找局部极大值
    peaks = []
    for i in range(2, n - 2):
        if (energy_history[i] > threshold and
            energy_history[i] > energy_history[i-1] and
            energy_history[i] > energy_history[i+1] and
            energy_history[i] > energy_history[i-2] and
            energy_history[i] > energy_history[i+2]):
            peaks.append(i)
    
    # 检查相邻峰的间隔
    periodic_count = 0
    for i in range(1, len(peaks)):
        interval = peaks[i] - peaks[i-1]
        if min_lag <= interval <= max_lag:
            periodic_count += 1
    
    return periodic_count >= 2, len(peaks)


class SnoreDetector:
    """Python版鼾声检测器（与固件算法一致）"""
    
    def __init__(self):
        self.noise_floor_db = -60.0
        self.noise_alpha = 0.02
        self.state = 0  # 0=idle, 1=candidate, 2=confirmed
        self.snore_start_ms = 0
        self.current_ms = 0
        self.energy_history = np.full(64, -80.0, dtype=np.float32)
        self.history_idx = 0
        
        self.total_snore_frames = 0
        self.total_frames = 0
        self.events = []  # (start_ms, end_ms, confidence)
        self.current_event_start = 0
    
    def process_frame(self, frame):
        """处理一帧，返回是否检测到鼾声"""
        self.total_frames += 1
        
        # 1. 能量
        energy_db = compute_energy_db(frame)
        
        # 2. 更新底噪
        if energy_db < self.noise_floor_db + 5.0:
            self.noise_floor_db = (self.noise_floor_db * (1 - self.noise_alpha) + 
                                   energy_db * self.noise_alpha)
        
        # 3. Level 1: 能量闸门
        snr = energy_db - self.noise_floor_db
        if snr < ENERGY_THRESHOLD_DB:
            if self.state >= 1:
                # 事件结束
                if self.state == 2:
                    self.events.append((
                        self.current_event_start, 
                        self.current_ms,
                        0.8
                    ))
                self.state = 0
            return False, 0.0, 0.0
        
        # 4. Level 2: 频谱
        mag = compute_magnitude_fft(frame)
        spectral_ratio = compute_snore_band_ratio(mag)
        fundamental = estimate_fundamental(mag)
        
        if spectral_ratio < SPECTRAL_RATIO:
            if self.state >= 1:
                if self.state == 2:
                    self.events.append((self.current_event_start, self.current_ms, 0.8))
                self.state = 0
            return False, spectral_ratio, fundamental
        
        # 5. 存入历史
        self.energy_history[self.history_idx] = energy_db
        self.history_idx = (self.history_idx + 1) % 64
        
        # 6. Level 3: 状态机
        self.current_ms += FRAME_MS
        
        is_snore = False
        confidence = 0.0
        
        if self.state == 0:
            self.state = 1
            self.current_event_start = self.current_ms
        elif self.state == 1:
            dur = self.current_ms - self.current_event_start
            if dur > MIN_DURATION_MS and dur < MAX_DURATION_MS:
                periodic, n_peaks = check_periodicity(
                    self.energy_history, 
                    MIN_PERIOD_MS, MAX_PERIOD_MS, FRAME_MS
                )
                if periodic:
                    self.state = 2
                    self.total_snore_frames += 1
                    is_snore = True
                    confidence = 0.7 + (spectral_ratio - 0.55) * 0.5
            if dur > MAX_DURATION_MS:
                self.state = 0
        elif self.state == 2:
            self.total_snore_frames += 1
            is_snore = True
            confidence = 0.85
            dur = self.current_ms - self.current_event_start
            if dur > MAX_DURATION_MS:
                self.events.append((self.current_event_start, self.current_ms, 0.85))
                self.state = 0
        
        return is_snore, confidence, spectral_ratio
    
    def process_file(self, filepath):
        """处理整个音频文件"""
        audio = load_wav(filepath)
        
        n_frames = len(audio) // FRAME_SIZE
        snore_frames = 0
        
        for i in range(n_frames):
            frame = audio[i*FRAME_SIZE:(i+1)*FRAME_SIZE]
            is_snore, conf, ratio = self.process_frame(frame)
            if is_snore:
                snore_frames += 1
        
        # 处理末尾可能的进行中事件
        if self.state == 2:
            self.events.append((self.current_event_start, self.current_ms, 0.8))
            self.state = 0
        
        return {
            'file': filepath,
            'duration_sec': len(audio) / SAMPLE_RATE,
            'total_frames': self.total_frames,
            'snore_frames': self.total_snore_frames,
            'snore_ratio': self.total_snore_frames / max(1, self.total_frames),
            'n_events': len(self.events),
            'events': self.events,
            'noise_floor_db': self.noise_floor_db,
        }


def test_directory(directory, expected_label):
    """测试整个目录的文件，输出统计"""
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"目录不存在: {directory}")
        return
    
    wav_files = list(dir_path.glob('*.wav'))
    if not wav_files:
        print(f"目录 {directory} 没有 wav 文件")
        return
    
    print(f"\n{'='*60}")
    print(f"  测试目录: {directory} (期望: {expected_label})")
    print(f"  文件数: {len(wav_files)}")
    print(f"{'='*60}")
    
    results = []
    n_correct = 0
    
    for wav_file in sorted(wav_files)[:20]:  # 最多测20个
        det = SnoreDetector()
        try:
            r = det.process_file(str(wav_file))
            results.append(r)
            
            # 判断是否检测到鼾声
            has_snore = r['n_events'] > 0 or r['snore_ratio'] > 0.1
            correct = has_snore if expected_label == 'snore' else not has_snore
            if correct:
                n_correct += 1
            
            mark = '✅' if correct else '❌'
            print(f"  {mark} {wav_file.name:30s} "
                  f"事件: {r['n_events']:2d}  "
                  f"鼾声帧占比: {r['snore_ratio']*100:5.1f}%  "
                  f"底噪: {r['noise_floor_db']:.1f}dB")
        except Exception as e:
            print(f"  ⚠️ {wav_file.name}: {e}")
    
    if results:
        acc = n_correct / len(results) * 100
        print(f"\n  准确率: {n_correct}/{len(results)} ({acc:.1f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='规则版鼾声检测器测试')
    parser.add_argument('file', nargs='?', help='单个 wav 文件')
    parser.add_argument('--folder', help='测试整个目录')
    parser.add_argument('--snore_dir', default='dataset/snore', help='鼾声样本目录')
    parser.add_argument('--noise_dir', default='dataset/noise', help='噪音样本目录')
    parser.add_argument('--full_test', action='store_true', help='完整测试（正+负样本）')
    
    args = parser.parse_args()
    
    if args.full_test:
        print("酣眠 SnoozMate · 规则版检测器完整测试")
        snore_results = test_directory(args.snore_dir, 'snore')
        noise_results = test_directory(args.noise_dir, 'noise')
        
        # 汇总
        if snore_results and noise_results:
            n_snore = len(snore_results)
            n_noise = len(noise_results)
            tp = sum(1 for r in snore_results if r['n_events'] > 0 or r['snore_ratio'] > 0.1)
            tn = sum(1 for r in noise_results if r['n_events'] == 0 and r['snore_ratio'] <= 0.1)
            total = n_snore + n_noise
            acc = (tp + tn) / total * 100
            
            print(f"\n{'='*60}")
            print(f"  汇总结果")
            print(f"{'='*60}")
            print(f"  真阳性 (鼾声判对): {tp}/{n_snore}")
            print(f"  真阴性 (噪音判对): {tn}/{n_noise}")
            print(f"  总体准确率: {acc:.1f}%")
            print()
            
            if acc < 70:
                print("  ⚠️  准确率偏低，建议:")
                print("    1. 调整 ENERGY_THRESHOLD_DB（当前 3.0 dB）")
                print("    2. 调整 SPECTRAL_RATIO（当前 0.55）")
                print("    3. 检查数据质量")
            elif acc < 85:
                print("  ✅ 基线可用，TinyML 版会进一步提升")
            else:
                print("  🎉 规则版表现很好！")
    
    elif args.folder:
        label = 'snore' if 'snore' in args.folder.lower() else 'noise'
        test_directory(args.folder, label)
    
    elif args.file:
        det = SnoreDetector()
        r = det.process_file(args.file)
        print(f"文件: {r['file']}")
        print(f"时长: {r['duration_sec']:.1f} 秒")
        print(f"总帧数: {r['total_frames']}")
        print(f"鼾声帧数: {r['snore_frames']} ({r['snore_ratio']*100:.1f}%)")
        print(f"事件数: {r['n_events']}")
        print(f"底噪估计: {r['noise_floor_db']:.1f} dB")
        if r['events']:
            print(f"事件列表:")
            for i, (start, end, conf) in enumerate(r['events'][:10]):
                print(f"  #{i+1}: {start/1000:.2f}s - {end/1000:.2f}s "
                      f"(时长 {(end-start)/1000:.2f}s)")
    else:
        print("用法:")
        print("  单文件: python rule_detector_test.py test.wav")
        print("  目录:   python rule_detector_test.py --folder dataset/snore")
        print("  完整测试: python rule_detector_test.py --full_test")


if __name__ == '__main__':
    main()
