"""
模型权重提取 + 生成 C 头文件
生成手写 CNN 推理代码（不依赖 TFLite）
"""
import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_PATH = 'output/model/best_model_nobn.h5'
OUTPUT_DIR = 'output'

print(f"加载模型: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"参数量: {model.count_params():,}")

# 提取所有层的权重和偏置
layers_info = []
for layer in model.layers:
    weights = layer.get_weights()
    if not weights:
        continue
    
    layer_type = type(layer).__name__
    layer_name = layer.name
    
    if 'Conv2D' in layer_type:
        # Conv2D: kernel shape [kh, kw, in_ch, out_ch]
        # 转置为 [out_ch, in_ch, kh, kw] 方便 C 代码使用
        kernel = np.transpose(weights[0], (3, 2, 0, 1))
        bias = weights[1]
        layers_info.append({
            'name': layer_name,
            'type': 'conv',
            'kernel': kernel,
            'bias': bias,
            'kernel_shape': kernel.shape
        })
    elif 'Dense' in layer_type:
        kernel = weights[0]
        bias = weights[1]
        layers_info.append({
            'name': layer_name,
            'type': 'dense',
            'kernel': kernel,
            'bias': bias,
            'kernel_shape': kernel.shape
        })

print(f"\n提取了 {len(layers_info)} 层权重:")
for info in layers_info:
    print(f"  {info['name']:15s} {info['type']:4s} kernel={info['kernel_shape']}")

# 保存 .npz
npz_path = os.path.join(OUTPUT_DIR, 'model_weights.npz')
np.savez_compressed(npz_path,
    **{f"conv1_k": layers_info[0]['kernel'], 'conv1_b': layers_info[0]['bias'],
       f"conv2_k": layers_info[1]['kernel'], 'conv2_b': layers_info[1]['bias'],
       f"conv3_k": layers_info[2]['kernel'], 'conv3_b': layers_info[2]['bias'],
       f"fc1_k": layers_info[3]['kernel'], 'fc1_b': layers_info[3]['bias'],
       f"fc2_k": layers_info[4]['kernel'], 'fc2_b': layers_info[4]['bias']})
print(f"\n保存权重: {npz_path}")

# 生成 C 头文件
esp32_dir = os.path.join(OUTPUT_DIR, 'esp32')
os.makedirs(esp32_dir, exist_ok=True)

with open(os.path.join(esp32_dir, 'snore_model_weights.h'), 'w') as f:
    f.write('/* ============================================================\n')
    f.write(' * 酣眠 SnoozMate 鼾声检测模型权重\n')
    f.write(f' * 参数量: {model.count_params():,} ({model.count_params()*4/1024:.0f} KB)\n')
    f.write(' * 输入: 63×30 log-mel 频谱图\n')
    f.write(' * 输出: 1个标量 (鼾声概率 0~1)\n')
    f.write(' * 架构: 3×Conv2D(3×3) → GAP → Dense(128) → Dense(1)\n')
    f.write(' * 测试准确率: 96.17%\n')
    f.write(' * ============================================================ */\n\n')
    f.write('#ifndef SNORE_MODEL_WEIGHTS_H\n')
    f.write('#define SNORE_MODEL_WEIGHTS_H\n\n')
    f.write('#include <stdint.h>\n\n')
    
    for i, info in enumerate(layers_info):
        var_name = 'conv1' if i == 0 else 'conv2' if i == 1 else 'conv3' if i == 2 else 'fc1' if i == 3 else 'fc2'
        
        # kernel
        k = info['kernel']
        f.write(f'// {info["name"]} kernel: {k.shape}\n')
        f.write(f'const float {var_name}_kernel[{k.size}] = {{\n')
        for j in range(0, k.size, 12):
            chunk = k.ravel()[j:j+12]
            f.write('  ' + ', '.join(f'{v:.7e}f' for v in chunk) + ',\n')
        f.write('};\n\n')
        
        # bias
        b = info['bias']
        f.write(f'// {info["name"]} bias: {b.shape}\n')
        f.write(f'const float {var_name}_bias[{b.size}] = {{\n')
        for j in range(0, b.size, 12):
            chunk = b.ravel()[j:j+12]
            f.write('  ' + ', '.join(f'{v:.7e}f' for v in chunk) + ',\n')
        f.write('};\n\n')
    
    f.write('#endif /* SNORE_MODEL_WEIGHTS_H */\n')

print(f"✅ C 头文件: {esp32_dir}/snore_model_weights.h")

# 计算模型大小
total_float32_bytes = sum(l['kernel'].size + l['bias'].size for l in layers_info) * 4
print(f"float32 权重大小: {total_float32_bytes} bytes ({total_float32_bytes/1024:.0f} KB)")
print(f"int8 估计: {total_float32_bytes/4/1024:.0f} KB")
