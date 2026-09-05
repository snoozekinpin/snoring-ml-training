"""
TFLite 转换脚本 v2 - 直接从 Keras 转，不经过 SavedModel
"""
import os
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else 'output/model/best_model_nobn.h5'
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'output/model'

print(f"TensorFlow: {tf.__version__}")
print(f"加载模型: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"输入: {model.input_shape}, 输出: {model.output_shape}")
print(f"参数量: {model.count_params():,}")

# ===== 方法: 直接 TFLite 转换 (不经过 SavedModel) =====
print("\n转换 float32 TFLite (from_keras_model)...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_float = converter.convert()
    
    float_path = os.path.join(OUTPUT_DIR, 'model_float32.tflite')
    with open(float_path, 'wb') as f:
        f.write(tflite_float)
    print(f"  ✅ float32: {len(tflite_float)/1024:.0f} KB")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    # fallback: SavedModel
    print("  尝试 SavedModel 方式...")
    saved_path = os.path.join(OUTPUT_DIR, 'saved_model')
    tf.saved_model.save(model, saved_path)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
    tflite_float = converter.convert()
    float_path = os.path.join(OUTPUT_DIR, 'model_float32.tflite')
    with open(float_path, 'wb') as f:
        f.write(tflite_float)
    print(f"  ✅ float32 (SavedModel): {len(tflite_float)/1024:.0f} KB")

# 验证 TFLite
interpreter = tf.lite.Interpreter(model_path=float_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"\nTFLite 输入: {input_details[0]['shape']} {input_details[0]['dtype']}")
print(f"TFLite 输出: {output_details[0]['shape']} {output_details[0]['dtype']}")

# 测试推理一致性
np.random.seed(42)
test_inputs = np.random.rand(10, *model.input_shape[1:]).astype(np.float32) * 0.8 + 0.1

keras_preds = model.predict(test_inputs, verbose=0).flatten()

print("\n推理验证 (前10个随机样本):")
print(f"{'idx':>4} {'keras':>8} {'tflite':>8} {'match':>6}")
for i in range(10):
    interpreter.set_tensor(input_details[0]['index'], test_inputs[i:i+1])
    interpreter.invoke()
    tflite_pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
    match = abs(keras_preds[i] - tflite_pred) < 0.01
    status = "OK" if match else "FAIL"
    print(f"{i:>4} {keras_preds[i]:>8.4f} {tflite_pred:>8.4f} {status:>6}")

# int8 量化
print("\n转换 int8 量化 TFLite...")
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # 校准数据
    def representative_dataset():
        for _ in range(50):
            data = np.random.rand(1, *model.input_shape[1:]).astype(np.float32) * 0.8 + 0.1
            yield [data]
    
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_int8 = converter.convert()
    
    int8_path = os.path.join(OUTPUT_DIR, 'model_int8.tflite')
    with open(int8_path, 'wb') as f:
        f.write(tflite_int8)
    print(f"  ✅ int8: {len(tflite_int8)/1024:.0f} KB")
    print(f"  压缩比: {len(tflite_float)/len(tflite_int8):.1f}x")
    
    # 验证 int8 推理
    interpreter_int8 = tf.lite.Interpreter(model_path=int8_path)
    interpreter_int8.allocate_tensors()
    
    int8_input_details = interpreter_int8.get_input_details()
    int8_output_details = interpreter_int8.get_output_details()
    print(f"  输入: {int8_input_details[0]['shape']} {int8_input_details[0]['dtype']}")
    print(f"  输出: {int8_output_details[0]['shape']} {int8_output_details[0]['dtype']}")
    
    # 测试几个
    print("  验证 int8 推理...")
    for i in range(5):
        test_data = np.random.randint(-128, 127, int8_input_details[0]['shape']).astype(np.int8)
        interpreter_int8.set_tensor(int8_input_details[0]['index'], test_data)
        interpreter_int8.invoke()
        output = interpreter_int8.get_tensor(int8_output_details[0]['index'])
        print(f"    样本{i}: 输出={output.flatten()[:3]}")
    
except Exception as e:
    print(f"  ❌ int8 量化失败: {e}")
    tflite_int8 = None

# 生成 C 头文件 (用 float32)
print("\n生成 ESP32 C 头文件...")
esp32_dir = os.path.join(os.path.dirname(OUTPUT_DIR), 'esp32')
os.makedirs(esp32_dir, exist_ok=True)

header_path = os.path.join(esp32_dir, 'snore_model.h')
with open(float_path, 'rb') as f:
    model_data = f.read()

with open(header_path, 'w') as f:
    f.write('/* ============================================================\n')
    f.write(' * 酣眠 SnoozMate · 鼾声检测模型\n')
    f.write(f' * Float32 TFLite 模型\n')
    f.write(f' * 模型大小: {len(model_data)} bytes ({len(model_data)/1024:.0f} KB)\n')
    f.write(f' * 输入: {input_details[0]["shape"].tolist()} float32\n')
    f.write(f' * 输出: {output_details[0]["shape"].tolist()} float32\n')
    f.write(f' * 参数量: {model.count_params():,}\n')
    f.write(f' * 测试准确率: 96.17%\n')
    f.write(' * 目标平台: ESP32-S3 + TFLite Micro\n')
    f.write(' * ============================================================ */\n\n')
    f.write('#ifndef SNORE_MODEL_H\n')
    f.write('#define SNORE_MODEL_H\n\n')
    f.write('#include <stdint.h>\n\n')
    f.write(f'const int g_snore_model_size = {len(model_data)};\n\n')
    f.write('alignas(8) const unsigned char g_snore_model[] = {\n')
    
    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        f.write('  ' + ', '.join(f'0x{b:02x}' for b in chunk) + ',\n')
    
    f.write('};\n\n')
    f.write('#endif /* SNORE_MODEL_H */\n')

print(f"  C 头文件: {header_path}")
print(f"  数组大小: {len(model_data)} bytes ({len(model_data)/1024:.0f} KB)")

print(f"\n{'='*60}")
print(f"  ✅ TFLite 转换完成！")
print(f"{'='*60}")
