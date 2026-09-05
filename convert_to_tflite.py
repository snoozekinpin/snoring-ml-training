"""
TFLite 转换修复版
直接从 Keras 3 模型转 TFLite，用 tf.lite.TFLiteConverter.from_keras_model
或者 fallback 到 SavedModel
"""
import os
import sys
import numpy as np
import tensorflow as tf

# TensorFlow 2.21 + Keras 3: 用 from_keras_model 应该能工作
# 如果不行，就保存成 SavedModel 再转

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else 'output/model/best_model.h5'
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'output/model'

print(f"TensorFlow 版本: {tf.__version__}")
print(f"Keras 版本: {tf.keras.__version__}")
print(f"加载模型: {MODEL_PATH}")

# 加载模型
model = tf.keras.models.load_model(MODEL_PATH)
print(f"输入 shape: {model.input_shape}")
print(f"输出 shape: {model.output_shape}")
print(f"参数量: {model.count_params():,}")

# 保存为 SavedModel 格式（最兼容）
saved_model_path = os.path.join(OUTPUT_DIR, 'saved_model')
print(f"\n保存 SavedModel: {saved_model_path}")
tf.saved_model.save(model, saved_model_path)

# 从 SavedModel 转 TFLite (float32)
print("\n转换 float32 TFLite (从 SavedModel)...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_float = converter.convert()

float_path = os.path.join(OUTPUT_DIR, 'model_float32.tflite')
with open(float_path, 'wb') as f:
    f.write(tflite_float)
print(f"  float32: {len(tflite_float)/1024:.0f} KB")

# 生成校准数据
print("\n准备校准数据...")
input_shape = model.input_shape[1:]
np.random.seed(42)
# 模拟 log-mel 归一化后的值 (0~1)
calib_data = np.random.rand(200, *input_shape).astype(np.float32)
calib_data = calib_data * 0.8 + 0.1  # 0.1~0.9 范围

def representative_dataset():
    for i in range(200):
        yield [calib_data[i:i+1]]

# int8 量化
print("\n转换 int8 量化 TFLite...")
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_int8 = converter.convert()
    
    int8_path = os.path.join(OUTPUT_DIR, 'model_int8.tflite')
    with open(int8_path, 'wb') as f:
        f.write(tflite_int8)
    print(f"  int8: {len(tflite_int8)/1024:.0f} KB")
    print(f"  压缩比: {len(tflite_float)/len(tflite_int8):.1f}x")
    
    # 验证推理
    print("\n验证 int8 模型推理...")
    interpreter = tf.lite.Interpreter(model_content=tflite_int8)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  输入: {input_details[0]['shape'].tolist()} {input_details[0]['dtype']}")
    print(f"  输出: {output_details[0]['shape'].tolist()} {output_details[0]['dtype']}")
    
    # 测试 10 次推理
    import time
    times = []
    for i in range(10):
        test_input = np.random.randint(-128, 127, input_details[0]['shape']).astype(np.int8)
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.time() - start)
    
    avg_ms = np.mean(times) * 1000
    print(f"  平均推理时间: {avg_ms:.2f} ms (PC CPU)")
    print(f"  输出示例: {output.flatten()[:5]}")
    
    # 生成 C 头文件
    print("\n生成 ESP32 C 头文件...")
    esp32_dir = os.path.join(os.path.dirname(OUTPUT_DIR), 'esp32')
    os.makedirs(esp32_dir, exist_ok=True)
    
    header_path = os.path.join(esp32_dir, 'snore_model.h')
    with open(int8_path, 'rb') as f:
        model_data = f.read()
    
    with open(header_path, 'w') as f:
        f.write('/* ============================================================\n')
        f.write(' * 酣眠 SnoozMate · 鼾声检测模型 (int8 量化)\n')
        f.write(' * TFLite Micro 部署用\n')
        f.write(f' * 模型大小: {len(model_data)} bytes ({len(model_data)/1024:.0f} KB)\n')
        f.write(f' * 输入: {input_details[0]["shape"].tolist()} int8\n')
        f.write(f' * 输出: {output_details[0]["shape"].tolist()} int8\n')
        f.write(f' * 参数量: {model.count_params():,}\n')
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
    print(f"  数组大小: {len(model_data)} bytes")
    
    print("\n✅ TFLite 转换全部完成！")
    
except Exception as e:
    print(f"❌ int8 量化失败: {e}")
    import traceback
    traceback.print_exc()
