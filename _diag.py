
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# 加载 TFLite 模型
model_path = "output/model/model_float32.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"TFLite 输入: {input_details[0]["shape"]} {input_details[0]["dtype"]}")
print(f"TFLite 输出: {output_details[0]["shape"]} {output_details[0]["dtype"]}")

# 加载 Keras 模型
keras_model = tf.keras.models.load_model("output/model/best_model_nobn.h5")
print(f"Keras 输入: {keras_model.input_shape}")
print(f"Keras 输出: {keras_model.output_shape}")

# 加载数据
import sys
sys.path.insert(0, ".")
from train import audio_to_features, load_dataset

X, y = load_dataset("dataset")
X_feat = audio_to_features(X[:100])

# 对比前10个
print("\n前10个样本对比:")
print(f"{'idx':>4} {'label':>5} {'keras':>8} {'tflite':>8} {'match':>6}")
for i in range(10):
    inp = X_feat[i:i+1].astype(np.float32)
    kp = keras_model.predict(inp, verbose=0)[0][0]
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    tp = interpreter.get_tensor(output_details[0]['index'])[0][0]
    match = "OK" if (kp > 0.5) == (tp > 0.5) else "FAIL"
    print(f"{i:>4} {y[i]:>5} {kp:>8.4f} {tp:>8.4f} {match:>6}")

# 统计
kp_all = keras_model.predict(X_feat, verbose=0).flatten()
tp_all = []
for i in range(len(X_feat)):
    interpreter.set_tensor(input_details[0]['index'], X_feat[i:i+1].astype(np.float32))
    interpreter.invoke()
    tp_all.append(interpreter.get_tensor(output_details[0]['index'])[0][0])
tp_all = np.array(tp_all)

# 相关性
corr = np.corrcoef(kp_all, tp_all)[0,1]
print(f"\n预测相关性: {corr:.4f}")
print(f"Keras 范围: [{kp_all.min():.4f}, {kp_all.max():.4f}], mean={kp_all.mean():.4f}")
print(f"TFLite 范围: [{tp_all.min():.4f}, {tp_all.max():.4f}], mean={tp_all.mean():.4f}")

# TFLite 准确率
tflite_preds = (tp_all > 0.5).astype(int)
true_labels = y
acc = np.mean(tflite_preds == true_labels) * 100
print(f"TFLite 准确率: {acc:.1f}%")

# 看看是不是反向了
tflite_preds_inv = (tp_all < 0.5).astype(int)
acc_inv = np.mean(tflite_preds_inv == true_labels) * 100
print(f"TFLite 反向准确率: {acc_inv:.1f}%")
