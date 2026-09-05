# -*- coding: utf-8 -*-
"""
v3.1 召回优化版
- 真实数据验证集上：acc 84.7% / recall 72.1%（漏检27.9%）
- 本版：pos_weight 平衡 + 阈值扫描 + 高召回部署阈值
"""
import os, sys, json, time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 复用 v3 的数据和特征函数
from train_v3 import (
    load_wav, audio_to_features, AugmentedDataGenerator,
    X_train, y_train, X_val, y_val,
    DATA, SR, N_FFT, HOP, N_MELS
)
from pathlib import Path

print(f"数据已加载: 训练{X_train.shape} 验证{X_val.shape}")

# ─── 模型（同结构，loss 加权） ───
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

# pos_weight：漏检比误报代价高 → 拉高正样本权重
n_pos = y_train.sum(); n_neg = len(y_train) - n_pos
pos_weight = min(2.5, n_neg / max(n_pos, 1))
print(f"训练集正负比: {n_pos}:{n_neg} → pos_weight={pos_weight:.2f}")

def weighted_bce(y_true, y_pred):
    bce = keras.backend.binary_crossentropy(y_true, y_pred)
    return bce * (1 + (pos_weight - 1) * y_true)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=weighted_bce,
    metrics=['accuracy'],
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
]

train_gen = AugmentedDataGenerator(X_train, y_train, augment=True)
val_gen = AugmentedDataGenerator(X_val, y_val, augment=False)

t0 = time.time()
history = model.fit(train_gen, validation_data=val_gen, epochs=50, callbacks=callbacks, verbose=2)
print(f"\n训练耗时: {time.time()-t0:.0f}s")

# ─── 阈值扫描：找部署阈值（目标：漏检<15%，误报<15%） ───
probs = model.predict(X_val, verbose=0).flatten()
print("\n═══ 阈值扫描（真实数据验证集） ═══")
print(f"{'阈值':>6} {'准确率':>8} {'召回率':>8} {'精确率':>8} {'漏检%':>7} {'误报%':>7}")
best_threshold = 0.5
for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]:
    pred = (probs > t).astype(int)
    tp = ((pred==1)&(y_val==1)).sum(); fp = ((pred==1)&(y_val==0)).sum()
    fn = ((pred==0)&(y_val==1)).sum(); tn = ((pred==0)&(y_val==0)).sum()
    acc = (tp+tn)/len(y_val); rec = tp/max(tp+fn,1); prec = tp/max(tp+fp,1)
    fnr = fn/max(tp+fn,1)*100; fpr = fp/max(fp+tn,1)*100
    print(f"{t:>6} {acc:>8.3f} {rec:>8.3f} {prec:>8.3f} {fnr:>7.1f} {fpr:>7.1f}")
    # 目标平衡点：漏检和误报都不超15%
    if fnr <= 15 and fpr <= 15:
        best_threshold = t

# 选漏检<=15%的最大阈值（误报最小）
for t in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25]:
    pred = (probs > t).astype(int)
    tp = ((pred==1)&(y_val==1)).sum(); fp = ((pred==1)&(y_val==0)).sum()
    fn = ((pred==0)&(y_val==1)).sum(); tn = ((pred==0)&(y_val==0)).sum()
    fnr = fn/max(tp+fn,1)*100; fpr = fp/max(fp+tn,1)*100
    if fnr <= 15:
        best_threshold = t
        break

print(f"\n✅ 部署阈值: {best_threshold}")
pred = (probs > best_threshold).astype(int)
tp = ((pred==1)&(y_val==1)).sum(); fp = ((pred==1)&(y_val==0)).sum()
fn = ((pred==0)&(y_val==1)).sum(); tn = ((pred==0)&(y_val==0)).sum()
print(f"最终指标: acc={(tp+tn)/len(y_val):.3f} recall={tp/max(tp+fn,1):.3f} "
      f"prec={tp/max(tp+fp,1):.3f} 漏检={fn/max(tp+fn,1)*100:.1f}% 误报={fp/max(fp+tn,1)*100:.1f}%")
print(f"混淆矩阵: TP={tp} FP={fp} FN={fn} TN={tn}")

# ─── 保存 ───
out = Path(r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output")
model.save(out / "model" / "best_model_v31.h5")
meta = {
    "threshold": best_threshold,
    "val_metrics": {
        "accuracy": float((tp+tn)/len(y_val)),
        "recall": float(tp/max(tp+fn,1)),
        "precision": float(tp/max(tp+fp,1)),
        "fnr": float(fn/max(tp+fn,1)),
        "fpr": float(fp/max(fp+tn,1)),
    },
    "pos_weight": float(pos_weight),
    "data": "train: synth+esc50+50%WHLTalent, val: 50% WHLTalent (real)",
}
with open(out / "model" / "model_v31_meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"\n✅ 保存: best_model_v31.h5 + model_v31_meta.json")
