"""
从 .keras 文件加载并转 TFLite（绕过 SavedModel bug）
"""
import os
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else 'output/model/best_model_nobn.h5'
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else 'output/model'

print(f"TF 版本: {tf.__version__}")
print(f"加载: {MODEL_PATH}")

# 加载模型
model = tf.keras.models.load_model(MODEL_PATH)
print(f"输入: {model.input_shape}")
print(f"参数量: {model.count_params():,}")

# 先保存为 SavedModel（Keras 3 格式）
saved_model_path = os.path.join(OUTPUT_DIR, 'saved_model_v2')
print(f"\n保存 SavedModel (Keras 3): {saved_model_path}")
tf.saved_model.save(model, saved_model_path)

# 检查 SavedModel 内容
print(f"SavedModel 内容:")
for f in os.listdir(saved_model_path):
    print(f"  {f}")

# 尝试用 TF 1.x 兼容的方式转
print("\n尝试 TFLite 转换...")
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    # 禁用可能引起问题的选项
    converter._experimental_lower_tensor_list_ops = False
    converter.optimize = [tf.lite.Optimize.DEFAULT]
    
    tflite_float = converter.convert()
    
    float_path = os.path.join(OUTPUT_DIR, 'model_float32.tflite')
    with open(float_path, 'wb') as f:
        f.write(tflite_float)
    print(f"  ✅ float32: {len(tflite_float)/1024:.0f} KB")
except Exception as e:
    print(f"  ❌ SavedModel 转换失败: {e}")
    print(f"  尝试其他方法...")
    
    # fallback: 用 tf.function 重新包装模型
    print("\n  重新包装模型...")
    
    class SnoreModel(tf.Module):
        def __init__(self, keras_model):
            self.model = keras_model
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 63, 30], dtype=tf.float32)])
        def __call__(self, x):
            return self.model(x, training=False)
    
    wrapped = SnoreModel(model)
    exported_path = os.path.join(OUTPUT_DIR, 'exported_module')
    tf.saved_model.save(wrapped, exported_path)
    
    print(f"\n  从 wrapper SavedModel 转换...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(exported_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_float = converter.convert()
        
        float_path = os.path.join(OUTPUT_DIR, 'model_float32.tflite')
        with open(float_path, 'wb') as f:
            f.write(tflite_float)
        print(f"  ✅ float32 (wrapper): {len(tflite_float)/1024:.0f} KB")
    except Exception as e2:
        print(f"  ❌ 也失败: {e2}")
        sys.exit(1)

# 验证
float_path = os.path.join(OUTPUT_DIR, 'model_float32.tflite')
if os.path.exists(float_path):
    interpreter = tf.lite.Interpreter(model_path=float_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n验证 TFLite 推理:")
    print(f"  输入: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"  输出: {output_details[0]['shape']} {output_details[0]['dtype']}")
    
    # 测试几个样本
    np.random.seed(42)
    test_inputs = np.random.rand(5, *model.input_shape[1:]).astype(np.float32) * 0.8 + 0.1
    
    keras_preds = model.predict(test_inputs, verbose=0).flatten()
    
    print(f"\n{'idx':>4} {'keras':>8} {'tflite':>8} {'match':>6}")
    for i in range(5):
        interpreter.set_tensor(input_details[0]['index'], test_inputs[i:i+1])
        interpreter.invoke()
        tflite_pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
        match = abs(keras_preds[i] - tflite_pred) < 0.05
        status = "✅" if match else "❌"
        print(f"{i:>4} {keras_preds[i]:>8.4f} {tflite_pred:>8.4f} {status:>6}")
