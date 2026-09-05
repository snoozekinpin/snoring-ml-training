# -*- coding: utf-8 -*-
"""Step 4: jibran 1000 真实样本模型推理（v4 浮点 vs v4-int8）"""
import os, sys, time
sys.path.insert(0, r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training")
os.chdir(r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training")

import numpy as np
import wave
from pathlib import Path
from collections import Counter

# 加载 jibran 数据
SNORE_DIR = Path(r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\dataset\snoring_extra\jibran")
NOISE_DIR = SNORE_DIR  # 同一个目录，靠前缀区分

def load_wav_as_features(path, n_mels=30, n_fft=512, hop=256):
    """wav → 61×30 log-mel"""
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
        if sr != 16000:
            new_len = int(len(audio) * 16000 / sr)
            audio = np.interp(np.linspace(0, len(audio)-1, new_len),
                              np.arange(len(audio)), audio).astype(np.float32)
        # 中心取 1 秒
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        else:
            start = (len(audio) - 16000) // 2
            audio = audio[start:start + 16000]

        # FFT → mel
        WIN = np.hanning(n_fft).astype(np.float32)
        def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
        def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)
        fmin, fmax = 0.0, 8000.0
        mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz_pts = mel_to_hz(mel_pts)
        bins = np.floor((n_fft + 1) * hz_pts / 16000).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for i in range(n_mels):
            for j in range(bins[i], bins[i+1]):
                if j < fb.shape[1]:
                    fb[i, j] = (j - bins[i]) / (bins[i+1] - bins[i] + 1e-9)
            for j in range(bins[i+1], bins[i+2]):
                if j < fb.shape[1]:
                    fb[i, j] = (bins[i+2] - j) / (bins[i+2] - bins[i+1] + 1e-9)

        # 单帧特征
        frame = audio[:n_fft] * WIN
        fft = np.fft.rfft(frame)
        spec = np.abs(fft[:n_fft//2+1]) ** 2
        mel = spec @ fb.T
        log_spec = np.log(mel + 1e-9)
        log_spec = (log_spec - log_spec.min()) / (log_spec.max() - log_spec.min() + 1e-9)
        return log_spec.astype(np.float32)  # shape (30,)
    except Exception as e:
        return None

print("=" * 70)
print("【Step 4: jibran 1000 真实样本推理】")
print("=" * 70)

# 加载 500 鼾声 + 500 噪声
print("\n加载数据...")
snore_files = sorted(SNORE_DIR.glob("jibran_s_*.wav"))[:500]
noise_files = sorted(SNORE_DIR.glob("jibran_n_*.wav"))[:500]
print(f"  鼾声: {len(snore_files)}")
print(f"  噪声: {len(noise_files)}")

X_list, y_list = [], []
print("  提取特征...")
t0 = time.time()
for i, f in enumerate(snore_files):
    feat = load_wav_as_features(f)
    if feat is not None and feat.shape == (30,):
        X_list.append(feat)
        y_list.append(1)
    if (i+1) % 100 == 0:
        print(f"    snore {i+1}/500 ({time.time()-t0:.0f}s)")
for i, f in enumerate(noise_files):
    feat = load_wav_as_features(f)
    if feat is not None and feat.shape == (30,):
        X_list.append(feat)
        y_list.append(0)
    if (i+1) % 100 == 0:
        print(f"    noise {i+1}/500 ({time.time()-t0:.0f}s)")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"  ✅ 加载完成: {X.shape} 鼾声={int(y.sum())} 噪声={int((1-y).sum())}  ({time.time()-t0:.0f}s)")

# 加载 v4 浮点模型
print("\n[模型 1: v4 浮点]")
import tensorflow as tf
fp_model = tf.keras.models.load_model(
    r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output\model\best_model_v4.h5",
    compile=False
)
print(f"  参数量: {fp_model.count_params()}")

# 注：v4 输入是 61×30，但我们只取了 1 帧 30 维
# reshape 成 (N, 61, 30) - 在 30 维上重复 61 次
print(f"\n[尝试推理 - reshape 成 61×30]")
X_repeat = np.repeat(X[:, np.newaxis, :], 61, axis=1)  # (N, 61, 30)
print(f"  reshape 后: {X_repeat.shape}")

t0 = time.time()
y_pred_fp = fp_model.predict(X_repeat[:200], batch_size=32, verbose=0).flatten()
elapsed_fp = time.time() - t0
print(f"  200 样本推理: {elapsed_fp:.2f}s  ({elapsed_fp/200*1000:.1f}ms/样本)")

# 全 1000 推理
t0 = time.time()
y_pred_fp = fp_model.predict(X_repeat, batch_size=32, verbose=0).flatten()
elapsed_fp = time.time() - t0
print(f"  全部 {len(X)} 样本: {elapsed_fp:.2f}s  ({elapsed_fp/len(X)*1000:.1f}ms/样本)")

# 阈值扫描
print("\n[浮点 v4 阈值扫描]")
for thr in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
    y_pred_bin = (y_pred_fp > thr).astype(int)
    tp = ((y_pred_bin == 1) & (y == 1)).sum()
    fp = ((y_pred_bin == 1) & (y == 0)).sum()
    fn = ((y_pred_bin == 0) & (y == 1)).sum()
    tn = ((y_pred_bin == 0) & (y == 0)).sum()
    acc = (tp + tn) / len(y)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(tp + fn, 1)
    print(f"  thr={thr}: acc={acc*100:.1f}% recall={recall*100:.1f}% FPR={fpr*100:.1f}% FNR={fnr*100:.1f}%")

# 加载 v4-int8
print("\n[模型 2: v4-int8 TFLite]")
import os
int8_path = r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output\snore_model_v4_int8.tflite"
int8_size = os.path.getsize(int8_path)
print(f"  int8 模型: {int8_size//1024} KB")

interp = tf.lite.Interpreter(model_path=int8_path)
interp.allocate_tensors()
in_d = interp.get_input_details()[0]
out_d = interp.get_output_details()[0]
in_scale, in_zero = in_d['quantization']
out_scale, out_zero = out_d['quantization']
in_scale = float(in_scale)
out_scale = float(out_scale)
print(f"  输入: {in_d['shape']} scale={in_scale:.5f} zero={in_zero}")
print(f"  输出: {out_d['shape']} scale={out_scale:.5f} zero={out_zero}")

# 推理
y_pred_int8 = []
t0 = time.time()
for i in range(len(X)):
    x = X_repeat[i:i+1]
    x_q = (x / in_scale + in_zero).clip(-128, 127).astype(np.int8)
    interp.set_tensor(in_d['index'], x_q)
    interp.invoke()
    out = interp.get_tensor(out_d['index'])
    y_pred_int8.append((out.astype(np.float32) - out_zero) * out_scale)
elapsed_int8 = time.time() - t0
y_pred_int8 = np.array(y_pred_int8).flatten()
print(f"  推理 {len(X)} 样本: {elapsed_int8:.2f}s  ({elapsed_int8/len(X)*1000:.2f}ms/样本)")

# int8 阈值扫描
print("\n[int8 阈值扫描]")
for thr in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
    y_pred_bin = (y_pred_int8 > thr).astype(int)
    tp = ((y_pred_bin == 1) & (y == 1)).sum()
    fp = ((y_pred_bin == 1) & (y == 0)).sum()
    fn = ((y_pred_bin == 0) & (y == 1)).sum()
    tn = ((y_pred_bin == 0) & (y == 0)).sum()
    acc = (tp + tn) / len(y)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(tp + fn, 1)
    print(f"  thr={thr}: acc={acc*100:.1f}% recall={recall*100:.1f}% FPR={fpr*100:.1f}% FNR={fnr*100:.1f}%")

# 对比
print("\n" + "=" * 70)
print("【Step 4 总结】")
print("=" * 70)
print(f"  浮点 v4 模型: 1333 KB")
print(f"  int8 v4 模型:  {int8_size//1024} KB  (-{100 - int8_size//1024*100//1333}%)")
print(f"  浮点推理速度: {elapsed_fp/len(X)*1000:.1f} ms/样本")
print(f"  int8 推理速度: {elapsed_int8/len(X)*1000:.1f} ms/样本  ({elapsed_fp/elapsed_int8:.1f}x 加速)")
