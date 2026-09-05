# -*- coding: utf-8 -*-
"""导出 v4 权重到 ESP32 C 头文件"""
import os, numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MODEL_PATH = r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output\model\best_model_v4.h5"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"模型: {MODEL_PATH}")
print(f"参数量: {model.count_params():,}")

layers_info = []
for layer in model.layers:
    weights = layer.get_weights()
    if not weights:
        continue
    layer_type = type(layer).__name__
    if 'Conv2D' in layer_type:
        kernel = np.transpose(weights[0], (3, 2, 0, 1))
        layers_info.append({'name': layer.name, 'type': 'conv',
                            'kernel': kernel, 'bias': weights[1]})
    elif 'Dense' in layer_type:
        layers_info.append({'name': layer.name, 'type': 'dense',
                            'kernel': weights[0], 'bias': weights[1]})

names = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
out_h = r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output\esp32\snore_model_weights_v4.h"
with open(out_h, 'w') as f:
    f.write('/* v4 - 5538 real samples, val acc 0.97 recall 0.97 @0.65 */\n')
    f.write('/* Data: adrianagaler 1k + WHLTalent 866 + jibran 1k + ESC-50 1960 */\n')
    f.write('#ifndef SNORE_MODEL_WEIGHTS_V4_H\n#define SNORE_MODEL_WEIGHTS_V4_H\n\n')
    f.write('#include <stdint.h>\n\n')
    for i, info in enumerate(layers_info):
        var = names[i]
        k = info['kernel'].ravel()
        f.write(f'const float {var}_kernel[{k.size}] = {{\n')
        for j in range(0, k.size, 12):
            f.write('  ' + ', '.join(f'{v:.7e}f' for v in k[j:j+12]) + ',\n')
        f.write('};\n\n')
        b = info['bias'].ravel()
        f.write(f'const float {var}_bias[{b.size}] = {{\n')
        for j in range(0, b.size, 12):
            f.write('  ' + ', '.join(f'{v:.7e}f' for v in b[j:j+12]) + ',\n')
        f.write('};\n\n')
    f.write('/* 部署阈值（见 model_v4_meta.json） */\n')
    f.write('#define SNORE_THRESHOLD 0.65f\n\n')
    f.write('#endif\n')

total_bytes = sum(l['kernel'].size + l['bias'].size for l in layers_info) * 4
print(f"保存: {out_h} ({total_bytes/1024:.0f} KB float32)")
print(f"输入帧数: 61 (16000 samples, hop 256)")
print(f"部署阈值: 0.65 (来自 model_v4_meta.json)")
