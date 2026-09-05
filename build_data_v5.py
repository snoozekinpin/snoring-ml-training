# -*- coding: utf-8 -*-
"""
v5 数据构建：v4 + snbhanja/snoring-audioset-16k
目标：5538 → 7988 样本（+44%）
"""
import os, json, time
import numpy as np
import wave
from pathlib import Path

DATA_DIR = Path(r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\dataset")
OUT_DIR = Path(r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output")

SR = 16000
N_SAMPLES = 16000  # 1 秒
N_FFT = 512
HOP = 256
N_MELS = 30
EPS = 1e-9

def load_wav(path):
    try:
        with wave.open(str(path)) as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(n)
        if sw != 2:
            return None
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        if sr != SR:
            new_len = int(len(audio) * SR / sr)
            audio = np.interp(np.linspace(0, len(audio)-1, new_len),
                              np.arange(len(audio)), audio).astype(np.float32)
        if len(audio) < N_SAMPLES:
            audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
        else:
            start = (len(audio) - N_SAMPLES) // 2
            audio = audio[start:start + N_SAMPLES]
        return audio
    except:
        return None

def mel_filterbank():
    fmin, fmax = 0.0, SR / 2
    def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), N_MELS + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((N_FFT + 1) * hz_points / SR).astype(int)
    fb = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(N_MELS):
        left, center, right = bins[i], bins[i+1], bins[i+2]
        for j in range(left, center):
            if j < fb.shape[1]:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if j < fb.shape[1]:
                fb[i, j] = (right - j) / (right - center)
    return fb

FB = mel_filterbank()
WIN = np.hanning(N_FFT).astype(np.float32)

def audio_to_features(audio):
    if len(audio) <= N_FFT:
        audio = np.pad(audio, (0, N_FFT - len(audio) + 1))
    frames = 1 + (len(audio) - N_FFT) // HOP
    if frames <= 0:
        return None
    spec = np.empty((frames, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(frames):
        frame = audio[i*HOP:i*HOP+N_FFT] * WIN
        fft = np.fft.rfft(frame)
        spec[i] = np.abs(fft[:N_FFT // 2 + 1]) ** 2
    mel_spec = spec @ FB.T
    log_spec = np.log(mel_spec + EPS)
    log_spec = (log_spec - log_spec.min()) / (log_spec.max() - log_spec.min() + EPS)
    return log_spec.astype(np.float32)

# v4 数据源
snore_sources = [
    (DATA_DIR / "adrianagaler" / "snore", "adria_s", "adria_s_*.wav"),
    (DATA_DIR / "whltalent_extra" / "snore", "extra_s", "whl_s_*.wav"),
    (DATA_DIR / "snoring_extra" / "jibran", "jib_s", "jibran_s_*.wav"),
    (DATA_DIR / "snore", "old_whl_s", "whl_s*.wav"),
    (DATA_DIR / "snore", "old_esc_s", "real_esc50_*.wav"),
    # v5 新增：snbhanja
    (DATA_DIR / "snoring-audioset-16k" / "snoring", "audioset_s", "snoring_*.wav"),
]

noise_sources = [
    (DATA_DIR / "adrianagaler" / "noise", "adria_n", "adria_n_*.wav"),
    (DATA_DIR / "whltalent_extra" / "noise", "extra_n", "whl_n_*.wav"),
    (DATA_DIR / "snoring_extra" / "jibran", "jib_n", "jibran_n_*.wav"),
    (DATA_DIR / "noise", "old_whl_n", "whl_n*.wav"),
    (DATA_DIR / "noise", "old_esc_n", "real_esc50_*.wav"),
    (DATA_DIR / "esc50_16k_1s" / "noise", "esc50_1s", "*.wav"),
    # v5 新增：snbhanja
    (DATA_DIR / "snoring-audioset-16k" / "non_snoring", "audioset_n", "non_snoring_*.wav"),
]

print("=" * 60)
print("【v5 数据加载】")
print("=" * 60)

snore_files = []
for src_dir, src_name, pattern in snore_sources:
    if src_dir.exists():
        files = list(src_dir.glob(pattern))
        snore_files.extend(files)
        print(f"  [snore/{src_name}] {len(files)} 个")
    else:
        print(f"  [snore/{src_name}] ⚠️ 目录不存在: {src_dir}")
snore_files = list(set(snore_files))
print(f"  去重: {len(snore_files)}")

noise_files = []
for src_dir, src_name, pattern in noise_sources:
    if src_dir.exists():
        files = list(src_dir.glob(pattern))
        noise_files.extend(files)
        print(f"  [noise/{src_name}] {len(files)} 个")
    else:
        print(f"  [noise/{src_name}] ⚠️ 目录不存在: {src_dir}")
noise_files = list(set(noise_files))
print(f"  去重: {len(noise_files)}")

print(f"\n总样本: {len(snore_files) + len(noise_files)}")
print(f"  鼾声/噪声比: 1:{len(noise_files)/max(len(snore_files),1):.1f}")

# 平衡采样：噪声太多，限制在 1:2 比例
target_neg = min(len(noise_files), len(snore_files) * 2)
np.random.seed(42)
if len(noise_files) > target_neg:
    noise_files = list(np.random.choice(noise_files, target_neg, replace=False))
    print(f"  噪声下采样到: {len(noise_files)}")

def process(files, label, name):
    X, y = [], []
    t0 = time.time()
    skip = 0
    for i, f in enumerate(files):
        audio = load_wav(f)
        if audio is None:
            skip += 1
            continue
        feat = audio_to_features(audio)
        if feat is None or feat.shape != (61, 30):
            skip += 1
            continue
        X.append(feat)
        y.append(label)
        if (i+1) % 500 == 0:
            print(f"  [{name}] {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  [{name}] ✅ {len(X)} 跳过 {skip} ({time.time()-t0:.0f}s)")
    return X, y

print("\n开始提取特征...")
Xs, ys = process(snore_files, 1, "snore")
Xn, yn = process(noise_files, 0, "noise")

X = np.array(Xs + Xn, dtype=np.float32)
y = np.array(ys + yn, dtype=np.float32)
print(f"\n总数据: {X.shape} y={y.shape}")
print(f"  鼾声: {int(y.sum())} ({100*y.mean():.1f}%)")

# 划分 80/20
np.random.seed(42)
idx = np.arange(len(y))
np.random.shuffle(idx)
n_val = int(len(idx) * 0.2)
val_idx = idx[:n_val]
train_idx = idx[n_val:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
print(f"\n训练: {X_train.shape} pos={int(y_train.sum())} neg={int((1-y_train).sum())}")
print(f"验证: {X_val.shape} pos={int(y_val.sum())} neg={int((1-y_val).sum())}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT_DIR / "train_data_v5.npz",
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val)
size = (OUT_DIR / "train_data_v5.npz").stat().st_size / 1024 / 1024
print(f"\n💾 保存: {OUT_DIR/'train_data_v5.npz'} ({size:.1f} MB)")
