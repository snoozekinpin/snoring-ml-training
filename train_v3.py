# -*- coding: utf-8 -*-
"""
加强版训练脚本 v3
- 数据升级：ESC-50(2058) + WHLTalent真实录音(1374) + 合成(1000+18)
- 关键改进：真实数据验证集（防止"同分布过拟合"假象）
  训练集 = 合成 + ESC-50
  验证集 = 50% WHLTalent真实数据（模型从未见过）
- 模型：3层CNN + GAP，float32导出（TFLite有bug，手写C推理）
"""
import os
import sys
import json
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─── 特征提取（和 ESP32 端一致） ───
SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 30
F_MIN = 40
F_MAX = 6000

def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv_mel = lambda m: 700 * (10 ** (m / 2595) - 1)
    m_min, m_max = mel(fmin), mel(fmax)
    centers = inv_mel(np.linspace(m_min, m_max, n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        left, center, right = centers[i], centers[i+1], centers[i+2]
        up = (freqs - left) / max(center - left, 1e-9)
        down = (right - freqs) / max(right - center, 1e-9)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb

FB = mel_filterbank(SR, N_FFT, N_MELS, F_MIN, F_MAX)

def audio_to_features(audio):
    """log-mel，输出 (61, 30)，取值 0~1（训练和C端一致归一化）"""
    frames = 1 + (len(audio) - N_FFT) // HOP
    if frames <= 0:
        return np.zeros((61, N_MELS), dtype=np.float32)
    spec = np.empty((frames, N_MELS), dtype=np.float32)
    win = np.hanning(N_FFT).astype(np.float32)
    for i in range(frames):
        seg = audio[i*HOP:i*HOP+N_FFT] * win
        mag = np.abs(np.fft.rfft(seg))
        spec[i] = np.log(np.dot(FB, mag) + 1e-6)
    # 帧数对齐到 61（不够补零，多了截断）
    target_frames = 61
    if frames < target_frames:
        spec = np.vstack([spec, np.zeros((target_frames - frames, N_MELS), dtype=np.float32)])
    else:
        spec = spec[:target_frames]
    # 全局归一化到 0~1
    smin, smax = spec.min(), spec.max()
    if smax - smin < 1e-9:
        return np.zeros_like(spec)
    return (spec - smin) / (smax - smin)

def load_wav(path):
    import wave
    with wave.open(str(path)) as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    assert sr == SR, f"{path}: sr={sr}"
    return data.astype(np.float32) / 32768.0

# ─── 数据加载 ───
DATA = r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\dataset"

def collect(dirpath, prefix):
    return sorted([p for p in dirpath.glob("*.wav") if p.name.startswith(prefix)])

from pathlib import Path

# 训练池：合成 + ESC-50（旧有）
train_pos = collect(Path(DATA) / "snore", "synth_")          # 合成鼾声
train_neg = collect(Path(DATA) / "noise", "esc50_") + collect(Path(DATA) / "noise", "5-")
# WHLTalent 真实数据：一半训练，一半验证（关键：真实数据做验证集）
whl_pos = collect(Path(DATA) / "snore", "whl_")
whl_neg = collect(Path(DATA) / "noise", "whl_")

rng = np.random.RandomState(42)
rng.shuffle(whl_pos)
rng.shuffle(whl_neg)
split = lambda lst, ratio: (lst[:int(len(lst)*ratio)], lst[int(len(lst)*ratio):])

whl_pos_train, whl_pos_val = split(whl_pos, 0.5)
whl_neg_train, whl_neg_val = split(whl_neg, 0.5)

print("═══ 数据划分 ═══")
print(f"训练: 正{len(train_pos)+len(whl_pos_train)} (合成{len(train_pos)}+真实{len(whl_pos_train)}) "
      f"负{len(train_neg)+len(whl_neg_train)} (ESC{len(train_neg)}+真实{len(whl_neg_train)})")
print(f"验证(真实数据): 正{len(whl_pos_val)} 负{len(whl_neg_val)}")
print(f"测试(真实数据): 用验证集同一份（小数据集，交叉使用）")

def build_dataset(paths, label):
    X, y = [], []
    for p in paths:
        try:
            audio = load_wav(p)
            if len(audio) < SR: continue
            X.append(audio_to_features(audio))
            y.append(label)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    return X, y

print("\n加载训练数据...")
Xp, yp = build_dataset(train_pos + whl_pos_train, 1)
Xn, yn = build_dataset(train_neg + whl_neg_train, 0)
X_train = np.array(Xp + Xn)
y_train = np.array(yp + yn)
print(f"训练集: {X_train.shape}, 正负比 {yp.count(1)}:{yn.count(0)}")

print("加载验证数据（全真实）...")
Xvp, yvp = build_dataset(whl_pos_val, 1)
Xvn, yvn = build_dataset(whl_neg_val, 0)
X_val = np.array(Xvp + Xvn)
y_val = np.array(yvp + yvn)
print(f"验证集: {X_val.shape}（全部真实录音）")

# ─── 数据增强（训练中动态） ───
def augment(x):
    r = np.random.rand()
    if r < 0.3:
        noise = np.random.normal(0, 0.05, x.shape).astype(np.float32)
        x = np.clip(x + noise, 0, 1)
    elif r < 0.5:
        x = x * np.random.uniform(0.7, 1.0)
    elif r < 0.7:
        shift = np.random.randint(-3, 4)
        x = np.roll(x, shift, axis=0)
    return x.astype(np.float32)

class AugmentedDataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, augment=True):
        self.X, self.y = X, y
        self.bs = batch_size
        self.augment = augment
        self.idx = np.arange(len(y))
        np.random.shuffle(self.idx)
    def __len__(self):
        return int(np.ceil(len(self.y) / self.bs))
    def __getitem__(self, i):
        sel = self.idx[i*self.bs:(i+1)*self.bs]
        xb = np.array([augment(self.X[j]) if self.augment else self.X[j] for j in sel])
        return xb, self.y[sel]
    def on_epoch_end(self):
        np.random.shuffle(self.idx)

train_gen = AugmentedDataGenerator(X_train, y_train, augment=True)
val_gen = AugmentedDataGenerator(X_val, y_val, augment=False)

# ─── 模型 ───
model = keras.Sequential([
    layers.Input(shape=(61, 30), name='input'),
    layers.Reshape((61, 30, 1)),
    layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1'),
    layers.MaxPooling2D(2, name='pool1'),
    layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2'),
    layers.MaxPooling2D(2, name='pool2'),
    layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3'),
    layers.MaxPooling2D(2, name='pool3'),
    layers.GlobalAveragePooling2D(name='gap'),
    layers.Dense(128, activation='relu', name='dense1'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid', name='output'),
])

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(name='prec'), keras.metrics.Recall(name='rec')],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5),
]

t0 = time.time()
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=40,
    callbacks=callbacks,
    verbose=2,
)
print(f"\n训练耗时: {time.time()-t0:.0f}s")

# ─── 评估：整体 + 真实数据细分 ───
print("\n═══ 真实数据验证（模型从未见过的录音） ═══")
val_pred = (model.predict(X_val, verbose=0) > 0.5).flatten()
acc = (val_pred == y_val).mean()
tp = ((val_pred == 1) & (y_val == 1)).sum()
fp = ((val_pred == 1) & (y_val == 0)).sum()
fn = ((val_pred == 0) & (y_val == 1)).sum()
tn = ((val_pred == 0) & (y_val == 0)).sum()
print(f"准确率: {acc:.4f}")
print(f"精确率: {tp/(tp+fp+1e-9):.4f}  召回率: {tp/(tp+fn+1e-9):.4f}")
print(f"混淆矩阵: TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"漏检率: {fn/(tp+fn+1e-9)*100:.1f}%  误报率: {fp/(fp+tn+1e-9)*100:.1f}%")

# ─── 保存 ───
out = Path(r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output")
model.save(out / "model" / "best_model_v3.h5")
with open(out / "model" / "history_v3.json", "w") as f:
    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)
print(f"\n✅ 模型保存: {out/'model'/'best_model_v3.h5'}")
print(f"参数量: {model.count_params():,}")
