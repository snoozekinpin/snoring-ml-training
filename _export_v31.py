
import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_PATH = 'output/model/best_model_v31.h5'
print(f"加载模型: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
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
                            'kernel': kernel, 'bias': weights[1], 'kernel_shape': kernel.shape})
    elif 'Dense' in layer_type:
        layers_info.append({'name': layer.name, 'type': 'dense',
                            'kernel': weights[0], 'bias': weights[1], 'kernel_shape': weights[0].shape})

names = ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']
npz = {f"{names[i]}_k": l['kernel'] for i, l in enumerate(layers_info)}
npz.update({f"{names[i]}_b": l['bias'] for i, l in enumerate(layers_info)})
np.savez_compressed('output/model_weights_v31.npz', **npz)

out_h = 'output/esp32/snore_model_weights_v31.h'
with open(out_h, 'w') as f:
    f.write('/* v3.1 - trained on 4432 samples (30% real recordings) */\n')
    f.write('/* val: acc 0.86 recall 0.72 @0.5 | recall 0.81 @0.3 */\n')
    f.write('#ifndef SNORE_MODEL_WEIGHTS_V31_H\n#define SNORE_MODEL_WEIGHTS_V31_H\n\n#include <stdint.h>\n\n')
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
    f.write('#endif\n')

total_bytes = sum(l['kernel'].size + l['bias'].size for l in layers_info) * 4
print(f"保存: {out_h} ({total_bytes/1024:.0f} KB float32)")
print(f"输入帧数: 61 (16000 samples, hop 256)")
print("[OK]")
