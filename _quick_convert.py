
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

model = tf.keras.models.load_model('output/model/best_model_nobn.h5')
print(f"TF: {tf.__version__}")
print(f"模型输入: {model.input_shape}")

# 直接从 Keras 模型转，不用 SavedModel
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

try:
    tflite = converter.convert()
    with open('output/model/model_final.tflite', 'wb') as f:
        f.write(tflite)
    print(f"✅ TFLite: {len(tflite)/1024:.0f} KB")
    
    # 验证
    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    print(f"输入: {inp[0]['shape']} {inp[0]['dtype']}")
    print(f"输出: {out[0]['shape']} {out[0]['dtype']}")
    
    # 测试推理
    import numpy as np
    np.random.seed(42)
    test = np.random.rand(5, 63, 30).astype(np.float32) * 0.8 + 0.1
    kp = model.predict(test, verbose=0).flatten()
    tp = []
    for i in range(5):
        interp.set_tensor(inp[0]['index'], test[i:i+1])
        interp.invoke()
        tp.append(interp.get_tensor(out[0]['index'])[0][0])
    print("\n验证:")
    for i in range(5):
        match = abs(kp[i] - tp[i]) < 0.01
        print(f"  {i}: keras={kp[i]:.4f} tflite={tp[i]:.4f} {'OK' if match else 'FAIL'}")
        
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()
