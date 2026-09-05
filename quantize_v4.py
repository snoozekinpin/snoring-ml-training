# -*- coding: utf-8 -*-
"""
v4-int8 量化（Plan B 启动）
- 跳过数据扩增（snbhanja 数据已下架）
- 直接对 v4 (5538 样本训练的 97.3% 模型) 做 int8 量化
- 目标：权重 427KB → ~107KB，准确率损失 < 1%
"""
import os, json, time
import numpy as np
import wave
from pathlib import Path

DATA_DIR = Path(r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\dataset")
OUT_DIR = Path(r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output")

print("=" * 60)
print("【v4-int8 量化】")
print("=" * 60)
print(f"数据目录: {DATA_DIR}")
print(f"输出目录: {OUT_DIR}")

# 加载 v4 .npz（之前已生成的 5538 样本）
npz_path = OUT_DIR / "train_data_v4.npz"
if not npz_path.exists():
    print(f"❌ {npz_path} 不存在！先跑 build_data_v4.py")
    exit(1)

print(f"\n加载 {npz_path} ...")
data = np.load(npz_path)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
print(f"  训练: {X_train.shape} pos={int(y_train.sum())}")
print(f"  验证: {X_val.shape} pos={int(y_val.sum())}")

# 加载 v4 模型
h5_path = OUT_DIR / "model" / "best_model_v4.h5"
if not h5_path.exists():
    h5_path = OUT_DIR / "best_model_v4.h5"
print(f"\n加载 {h5_path} ...")
import tensorflow as tf
model = tf.keras.models.load_model(str(h5_path), compile=False)
print(f"  模型结构:")
model.summary()

# 手动 compile（Keras 3 加载旧 h5 需要）
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# 浮点 baseline
print("\n[1/3] 浮点 baseline 评估...")
results = model.evaluate(X_val, y_val, verbose=0)
print(f"  浮点 val: {dict(zip(model.metrics_names, [f'{v:.4f}' for v in results]))}")
# Keras 3 把 compile metrics 合并到 'compile_metrics' 字段
acc_fp = results[model.metrics_names.index('compile_metrics')]

# ─── Phase 3: int8 量化 ───
print("\n[2/3] int8 量化 (Post-Training Quantization)...")

def representative_dataset():
    """校准数据：200 个随机训练样本"""
    idx = np.random.choice(len(X_train), 200, replace=False)
    for i in idx:
        yield [X_train[i:i+1].astype(np.float32)]

# TFLite 转换器（绕过 Keras 3 wrapping bug）
print("  转换中...")
t0 = time.time()

# 用 ConcreteFunction 直接转换（绕开 Keras 3 wrapping）
# 注意：v4 模型输入是 3D (61, 30) 不是 4D batch
@tf.function
def inference(x):
    return model(x, training=False)

# 实际输入 shape: (None, 61, 30) — v4 模型是 3D 输入
concrete_func = inference.get_concrete_function(
    tf.TensorSpec([1, 61, 30], tf.float32)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
print(f"  ✅ 转换完成 ({time.time()-t0:.1f}s)")

# 保存
tflite_path = OUT_DIR / "snore_model_v4_int8.tflite"
tflite_path.write_bytes(tflite_model)
fp_size = (OUT_DIR / "model" / "best_model_v4.h5").stat().st_size if (OUT_DIR / "model" / "best_model_v4.h5").exists() else (OUT_DIR / "best_model_v4.h5").stat().st_size
int8_size = len(tflite_model)
print(f"  浮点 .h5: {fp_size//1024} KB")
print(f"  int8 .tflite: {int8_size//1024} KB")
print(f"  压缩比: {fp_size/int8_size:.2f}x")

# ─── 验证 int8 准确率 ───
print("\n[3/3] int8 推理验证...")
interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 量化参数
in_scale, in_zero = input_details[0]['quantization']
out_scale, out_zero = output_details[0]['quantization']
print(f"  输入量化: scale={in_scale:.5f}, zero={in_zero}")
print(f"  输出量化: scale={out_scale:.5f}, zero={out_zero}")

# 推理
y_pred_fp32 = model.predict(X_val, verbose=0).flatten()
y_pred_int8 = []
for i in range(len(X_val)):
    x = X_val[i:i+1].astype(np.float32)
    x_quant = (x / in_scale + in_zero).clip(-128, 127).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], x_quant)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    # 反量化
    out_fp = (out.astype(np.float32) - out_zero) * out_scale
    y_pred_int8.append(out_fp[0, 0])
y_pred_int8 = np.array(y_pred_int8)

# 阈值扫描（0.65）
threshold = 0.65
y_true = (y_val > 0.5).astype(int)
y_pred_fp32_bin = (y_pred_fp32 > threshold).astype(int)
y_pred_int8_bin = (y_pred_int8 > threshold).astype(int)

def metrics(y_true, y_pred, name):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    acc = (tp + tn) / len(y_true)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(tp + fn, 1)
    print(f"  {name}: acc={acc*100:.2f}% recall={recall*100:.2f}% fpr={fpr*100:.2f}% fnr={fnr*100:.2f}%")
    return acc, recall, fpr, fnr

metrics(y_true, y_pred_fp32_bin, "FP32")
acc_int8, recall_int8, fpr_int8, fnr_int8 = metrics(y_true, y_pred_int8_bin, "INT8")
acc_diff = (acc_fp - acc_int8) * 100
print(f"\n📊 准确率损失: {acc_diff:+.2f}% (阈值={threshold})")
print(f"{'✅ 量化通过 (< 1%)' if abs(acc_diff) < 1 else '❌ 量化损失过大'}")

# 保存元数据
meta = {
    "version": "v4-int8",
    "fp_size_bytes": fp_size,
    "int8_size_bytes": int8_size,
    "compression_ratio": round(fp_size / int8_size, 2),
    "threshold": threshold,
    "fp32_metrics": {
        "accuracy": float(acc_fp),
        "samples": len(y_val)
    },
    "int8_metrics": {
        "accuracy": float(acc_int8),
        "recall": float(recall_int8),
        "fpr": float(fpr_int8),
        "fnr": float(fnr_int8)
    },
    "accuracy_loss_pct": float(acc_diff),
    "input_quantization": {"scale": float(in_scale), "zero_point": int(in_zero)},
    "output_quantization": {"scale": float(out_scale), "zero_point": int(out_zero)},
    "data": "v4 train_data (5538 samples: 1846 snore + 3692 noise)"
}
meta_path = OUT_DIR / "model_v4_int8_meta.json"
meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(f"\n💾 元数据: {meta_path}")
