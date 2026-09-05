"""
数据集一键下载脚本
====================
自动下载并整理鼾声检测所需的数据集

数据来源:
  1. ESC-50 (环境音，做负样本)    - 2000条，50类，600MB
  2. YouTube 鼾声音频合集          - 用 yt-dlp 下载（可选）
  3. 自录数据 (推荐)               - 用同款麦克风录，域适配效果最好

使用:
  python download_datasets.py --esc50 --output dataset
  python download_datasets.py --all --output dataset
"""

import argparse
import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path
import numpy as np


def download_esc50(output_dir):
    """下载 ESC-50 数据集（环境音分类标准数据集）"""
    print("=" * 60)
    print("  下载 ESC-50 环境音数据集")
    print("=" * 60)
    
    esc50_dir = Path(output_dir) / 'esc50'
    esc50_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = esc50_dir / 'esc-50.zip'
    
    if not zip_path.exists():
        url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
        print(f"  从 GitHub 下载: {url}")
        print(f"  约 600MB，需要几分钟...")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(url, str(zip_path), 
                                       reporthook=_progress_hook)
            print("\n  ✅ 下载完成")
        except Exception as e:
            print(f"  ⚠️ 下载失败: {e}")
            print("  请手动下载: https://github.com/karoldvl/ESC-50/archive/master.zip")
            return False
    else:
        print("  已存在，跳过下载")
    
    # 解压
    print("  解压中...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(str(esc50_dir))
        print("  ✅ 解压完成")
    except Exception as e:
        print(f"  ⚠️ 解压失败: {e}")
        return False
    
    # 分类整理到 noise/ 目录
    noise_dir = Path(output_dir) / 'noise'
    noise_dir.mkdir(parents=True, exist_ok=True)
    
    audio_dir = esc50_dir / 'ESC-50-master' / 'audio'
    if not audio_dir.exists():
        # 试试其他路径格式
        for p in esc50_dir.rglob('*.wav'):
            if p.name.startswith('1-') or p.name.startswith('2-'):
                audio_dir = p.parent
                break
    
    count = 0
    if audio_dir.exists():
        for wav_file in audio_dir.glob('*.wav'):
            # ESC-50 命名格式: {fold}-{scene}-{take}.wav
            # 全部作为噪音负样本
            dst = noise_dir / f'esc50_{wav_file.name}'
            if not dst.exists():
                shutil.copy2(str(wav_file), str(dst))
            count += 1
        print(f"  ✅ 整理了 {count} 个噪音样本 -> {noise_dir}")
    
    return count > 0


def _progress_hook(block_num, block_size, total_size):
    """下载进度显示"""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
    mb = downloaded / (1024 * 1024)
    sys.stdout.write(f"\r  进度: {percent:.1f}% ({mb:.1f} MB)")
    sys.stdout.flush()


def generate_synthetic_snores(output_dir, n_samples=200):
    """
    生成合成鼾声样本（没有真实数据时的保底方案）
    用正弦波调制模拟鼾声的周期性特征
    
    注意：这只是保底，真实准确率不会太高。有条件还是用真实数据。
    """
    print("=" * 60)
    print(f"  生成 {n_samples} 个合成鼾声样本（保底）")
    print("=" * 60)
    
    try:
        from scipy.io import wavfile
    except ImportError:
        print("  ⚠️ 需要 scipy，跳过合成")
        return False
    
    snore_dir = Path(output_dir) / 'snore'
    snore_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 16000
    duration = 1.0  # 1秒
    
    for i in range(n_samples):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # 鼾声特征：低频基频（60-120Hz）+ 谐波 + 呼吸周期调制
        fundamental = 80 + np.random.randn() * 20  # 基频随机
        if fundamental < 50:
            fundamental = 50
        
        # 基频 + 2~3次谐波
        signal = np.zeros_like(t)
        for harmonic in range(1, 4):
            amp = 1.0 / harmonic
            # 加一点颤音
            vibrato = 1 + 0.1 * np.sin(2 * np.pi * 5 * t)
            signal += amp * np.sin(2 * np.pi * fundamental * harmonic * t * vibrato)
        
        # 呼吸周期调制（4-6秒一个周期 = 0.17-0.25Hz）
        breath_freq = 0.15 + np.random.random() * 0.15
        breath_env = 0.5 + 0.5 * np.sin(2 * np.pi * breath_freq * t + np.random.rand() * 2 * np.pi)
        signal *= breath_env
        
        # 加噪声模拟气道湍流
        noise_level = 0.1 + np.random.random() * 0.2
        signal += noise_level * np.random.randn(len(t))
        
        # 随机音量
        signal *= 0.3 + np.random.random() * 0.5
        
        # 限幅
        signal = np.clip(signal, -1, 1)
        
        # 转 16-bit PCM
        signal_int16 = (signal * 32767).astype(np.int16)
        
        wav_path = snore_dir / f'synth_snore_{i:04d}.wav'
        wavfile.write(str(wav_path), sr, signal_int16)
    
    print(f"  ✅ 生成 {n_samples} 个合成鼾声 -> {snore_dir}")
    print("  ⚠️ 合成数据仅作保底，建议补充真实鼾声录音")
    return True


def generate_synthetic_noise(output_dir, n_samples=200):
    """生成合成噪音样本"""
    print("=" * 60)
    print(f"  生成 {n_samples} 个合成噪音样本")
    print("=" * 60)
    
    try:
        from scipy.io import wavfile
    except ImportError:
        print("  ⚠️ 需要 scipy，跳过合成")
        return False
    
    noise_dir = Path(output_dir) / 'noise'
    noise_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 16000
    duration = 1.0
    
    noise_types = [
        ('white', lambda n: np.random.randn(n) * 0.3),
        ('pink', lambda n: _pink_noise(n) * 0.3),
        ('sine_1k', lambda n: 0.5 * np.sin(2 * np.pi * 1000 * np.linspace(0, duration, n))),
        ('sine_100', lambda n: 0.5 * np.sin(2 * np.pi * 100 * np.linspace(0, duration, n))),
        ('sweep', lambda n: 0.3 * np.sin(2 * np.pi * np.cumsum(np.linspace(200, 2000, n))/sr)),
        ('speech_like', lambda n: _speech_like_noise(n, sr)),
    ]
    
    count = 0
    per_type = n_samples // len(noise_types)
    
    for name, gen_fn in noise_types:
        for i in range(per_type):
            n_samples_t = int(sr * duration)
            signal = gen_fn(n_samples_t)
            # 加随机音量
            signal *= 0.2 + np.random.random() * 0.6
            signal = np.clip(signal, -1, 1)
            signal_int16 = (signal * 32767).astype(np.int16)
            
            wav_path = noise_dir / f'synth_{name}_{i:03d}.wav'
            wavfile.write(str(wav_path), sr, signal_int16)
            count += 1
    
    print(f"  ✅ 生成 {count} 个合成噪音 -> {noise_dir}")
    return True


def _pink_noise(n):
    """生成粉红噪声（Voss-McCartney 算法）"""
    n_units = 8
    values = np.random.randn(n_units) * 0.1
    white = np.random.randn(n)
    
    pink = np.zeros(n)
    for i in range(n):
        # 选择一个随机位
        r = i
        idx = 0
        while r & 1 == 0 and idx < n_units:
            values[idx] = np.random.randn() * 0.1
            r >>= 1
            idx += 1
        pink[i] = np.sum(values) + white[i] * 0.1
    
    return pink / np.max(np.abs(pink))


def _speech_like_noise(n, sr):
    """类语音噪声（振幅调制的宽带噪声）"""
    noise = np.random.randn(n) * 0.5
    # 4-8Hz 振幅调制（类似语音音节速率）
    mod_freq = 4 + np.random.random() * 4
    t = np.linspace(0, n/sr, n)
    envelope = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * mod_freq * t))
    return noise * envelope


def print_status(output_dir):
    """打印数据集状态"""
    print("\n" + "=" * 60)
    print("  数据集状态")
    print("=" * 60)
    
    data_dir = Path(output_dir)
    for subdir in ['snore', 'noise']:
        d = data_dir / subdir
        if d.exists():
            files = list(d.glob('*.wav'))
            total_size = sum(f.stat().st_size for f in files)
            print(f"  {subdir:10s}: {len(files):4d} 个文件, {total_size/1024/1024:.1f} MB")
        else:
            print(f"  {subdir:10s}: 0 个文件 (目录不存在)")


def main():
    parser = argparse.ArgumentParser(description='下载/生成鼾声检测数据集')
    parser.add_argument('--output', type=str, default='dataset',
                        help='输出目录')
    parser.add_argument('--esc50', action='store_true',
                        help='下载 ESC-50 环境音数据集（600MB）')
    parser.add_argument('--synth', action='store_true',
                        help='生成合成数据（保底方案）')
    parser.add_argument('--all', action='store_true',
                        help='下载/生成所有可用数据')
    parser.add_argument('--snore_count', type=int, default=200,
                        help='合成鼾声数量')
    parser.add_argument('--noise_count', type=int, default=200,
                        help='合成噪音数量')
    parser.add_argument('--status', action='store_true',
                        help='只查看当前数据集状态')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.status:
        print_status(args.output)
        return
    
    if args.all:
        args.esc50 = True
        args.synth = True
    
    print("酣眠 SnoozMate · 数据集准备工具")
    print(f"输出目录: {os.path.abspath(args.output)}")
    print()
    
    if args.esc50:
        download_esc50(args.output)
    
    if args.synth:
        generate_synthetic_snores(args.output, args.snore_count)
        generate_synthetic_noise(args.output, args.noise_count)
    
    print_status(args.output)
    
    print("\n" + "=" * 60)
    print("  下一步")
    print("=" * 60)
    print("  1. 录制真实鼾声（推荐，提升准确率 15-20%）")
    print("  2. 运行训练: python train.py --epochs 30")
    print("  3. 部署到 ESP32: 见 ../firmware/ 目录")
    print()


if __name__ == '__main__':
    main()
