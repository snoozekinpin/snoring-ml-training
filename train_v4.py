# -*- coding: utf-8 -*-
"""
v4 训练：大数据集 + 强增强 + 阈值扫描
- 数据: 1846 真实鼾声 + 3692 真实噪声 (5538 样本, 80/20 划分)
- 模型: 同 v3 结构 (3层CNN + GAP), 但更长训练 + 强增强
- 阈值扫描 + 部署
"""
import os, json, time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_PATH = r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output\train_data_v4.npz"
OUT_DIR = r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output"

print("加载数据...")
data = np.load(DATA_PATH)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
print(f"训练: {X_train.shape} pos={int(y_train.sum())} neg={int((1-y_train).sum())}")
print(f"验证: {X_val.shape} pos={int(y_val.sum())} neg={int((1-y_val).sum())}")

# ─── 数据增强（在线）───
class SnoreAugment(tf.keras.utils.Sequence):
    """在 log-mel 频谱上加噪声/时移/频率mask"""
    def __init__(self, X, y, batch=64, augment=True):
        self.X = X
        self.y = y
        self.batch = batch
        self.augment = augment
        self.n = len(X)

    def __len__(self):
        return self.n // self.batch

    def __getitem__(self, idx):
        start = idx * self.batch
        end = min(start + self.batch, self.n)
        Xb = self.X[start:end].copy()
        yb = self.y[start:end]
        if self.augment:
            # 1. 时移（水平偏移）
            n_aug = len(Xb) // 2
            if n_aug > 0:
                shifts = np.random.randint(-5, 5, n_aug)
                for i, s in enumerate(shifts):
                    if s > 0:
                        Xb[i] = np.pad(Xb[i, :-s], ((s, 0), (0, 0)))
                    elif s < 0:
                        Xb[i] = np.pad(Xb[i, -s:], ((0, -s), (0, 0)))
            # 2. 频率 mask（垂直 mask）
            n_mask = len(Xb) // 3
            for i in range(n_mask):
                f_start = np.random.randint(0, 20)
                f_width = np.random.randint(2, 6)
                Xb[i, :, f_start:f_start+f_width] = 0
            # 3. 噪声
            noise = np.random.normal(0, 0.02, Xb.shape).astype(np.float32)
            Xb = np.clip(Xb + noise, 0, 1)
        return Xb, yb

# ─── 模型 ───
def build_model():
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

    # pos_weight: 1:2 比例, 略加权即可
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = min(1.5, n_neg / max(n_pos, 1))
    print(f"pos_weight: {pos_weight:.2f}")

    def weighted_bce(y_true, y_pred):
        bce = keras.backend.binary_crossentropy(y_true, y_pred)
        return bce * (1 + (pos_weight - 1) * y_true)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=weighted_bce,
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()],
    )
    return model

model = build_model()

# 训练
train_gen = SnoreAugment(X_train, y_train, batch=64, augment=True)
val_gen = SnoreAugment(X_val, y_val, batch=64, augment=False)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
]

t0 = time.time()
history = model.fit(train_gen, validation_data=val_gen,
                    epochs=60, callbacks=callbacks, verbose=2)
print(f"\n训练耗时: {time.time()-t0:.0f}s")

# ─── 阈值扫描 ───
probs = model.predict(X_val, verbose=0).flatten()
print(f"\n{'阈值':>6} {'准确率':>8} {'召回率':>8} {'精确率':>8} {'漏检%':>7} {'误报%':>7}")
results = []
for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
    pred = (probs > t).astype(int)
    tp = ((pred==1)&(y_val==1)).sum()
    fp = ((pred==1)&(y_val==0)).sum()
    fn = ((pred==0)&(y_val==1)).sum()
    tn = ((pred==0)&(y_val==0)).sum()
    acc = (tp+tn)/len(y_val)
    rec = tp/max(tp+fn,1)
    prec = tp/max(tp+fp,1)
    fnr = fn/max(tp+fn,1)*100
    fpr = fp/max(fp+tn,1)*100
    print(f"{t:>6} {acc:>8.3f} {rec:>8.3f} {prec:>8.3f} {fnr:>7.1f} {fpr:>7.1f}")
    results.append((t, acc, rec, prec, fnr, fpr))

# 选漏检≤15% 误报≤15% 中的最严阈值
best = None
for t, acc, rec, prec, fnr, fpr in results:
    if fnr <= 15 and fpr <= 15:
        if best is None or t > best[0]:
            best = (t, acc, rec, prec, fnr, fpr)

# fallback: 召回最高
if best is None:
    best = max(results, key=lambda r: r[2])

print(f"\n部署阈值: {best[0]}")
print(f"  acc={best[1]:.3f} rec={best[2]:.3f} prec={best[3]:.3f} 漏检={best[4]:.1f}% 误报={best[5]:.1f}%")

# 保存
os.makedirs(os.path.join(OUT_DIR, "model"), exist_ok=True)
model.save(os.path.join(OUT_DIR, "model", "best_model_v4.h5"))
with open(os.path.join(OUT_DIR, "model", "model_v4_meta.json"), "w") as f:
    json.dump({
        "version": "v4",
        "threshold": best[0],
        "val_metrics": {
            "accuracy": float(best[1]),
            "recall": float(best[2]),
            "precision": float(best[3]),
            "fnr": float(best[4]),
            "fpr": float(best[5]),
        },
        "data_sources": [
            "adrianagaler/Snoring-Detection (1000 真实)",
            "WHLTalent-snore (s*=snore+e*=env, 2354 真实切片)",
            "jibran-mujtaba/Snore_Detection_Project (1000 真实)",
            "ESC-50 (1960 噪声切片)",
        ],
        "total_samples": 5538,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "positive_ratio": float(y_train.mean()),
    }, f, indent=2, ensure_ascii=False)
print(f"\n✅ 保存: best_model_v4.h5 + model_v4_meta.json")
