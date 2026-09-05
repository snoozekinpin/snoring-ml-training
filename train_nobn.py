"""
快速训练一个没有 BatchNormalization 的模型
TFLite 转换更稳定，ESP32 部署更可靠
"""
import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# 复用 train.py 的数据加载和特征提取
sys.path.insert(0, os.path.dirname(__file__))
from train import (
    SAMPLE_RATE, N_SAMPLES, load_dataset, audio_to_features,
    N_MELS, N_FFT, HOP_LENGTH, CONV_FILTERS, DENSE_UNITS, DROPOUT
)
from sklearn.model_selection import train_test_split


def build_model_no_bn(input_shape):
    """无 BN 的模型（TFLite 更稳定）"""
    model = models.Sequential([
        layers.Input(shape=input_shape, name='input'),
        layers.Reshape((input_shape[0], input_shape[1], 1), name='expand_channel'),
        layers.Resizing(32, 30, name='resize'),
        
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        layers.GlobalAveragePooling2D(name='gap'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout'),
        layers.Dense(1, activation='sigmoid', name='output'),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    params = model.count_params()
    print(f"模型参数量: {params:,} ({params*4/1024:.0f} KB float32)")
    
    return model


def main():
    data_dir = "dataset"
    output_dir = "output/model"
    epochs = 25
    batch_size = 32
    
    print("=" * 60)
    print("  酣眠 SnoozMate · 模型训练 (无BN版，TFLite友好)")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据集...")
    X, y = load_dataset(data_dir)
    print(f"  总样本: {len(X)}")
    
    # 2. 划分
    print("\n[2/5] 划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.18, random_state=42, stratify=y_train)
    
    print(f"  训练: {len(X_train)}  验证: {len(X_val)}  测试: {len(X_test)}")
    
    # 3. 特征提取
    print("\n[3/5] 提取 log-mel 特征...")
    X_train_feat = audio_to_features(X_train)
    X_val_feat = audio_to_features(X_val)
    X_test_feat = audio_to_features(X_test)
    print(f"  特征 shape: {X_train_feat.shape[1:]}")
    
    # 4. 训练
    print("\n[4/5] 训练模型...")
    model = build_model_no_bn(X_train_feat.shape[1:])
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=7,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4,
            min_lr=1e-6, verbose=1
        ),
    ]
    
    # 类权重
    n_pos = np.sum(y_train == 0)
    n_neg = np.sum(y_train == 1)
    total = n_pos + n_neg
    class_weight = {
        0: total / (2 * n_pos),
        1: total / (2 * n_neg),
    }
    
    history = model.fit(
        X_train_feat, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_feat, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    
    # 5. 评估
    print("\n[5/5] 评估模型...")
    test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test_feat, y_test, verbose=0)
    print(f"  Test Acc:  {test_acc*100:.2f}%")
    print(f"  Test Prec: {test_prec*100:.2f}%")
    print(f"  Test Rec:  {test_rec*100:.2f}%")
    
    # 保存
    model.save(os.path.join(output_dir, 'best_model_nobn.h5'))
    print(f"\n  模型已保存: {output_dir}/best_model_nobn.h5")
    
    # 转 TFLite
    print("\n转换 TFLite...")
    
    # 保存 SavedModel
    saved_path = os.path.join(output_dir, 'saved_model_nobn')
    tf.saved_model.save(model, saved_path)
    
    # float32
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
    tflite_float = converter.convert()
    float_path = os.path.join(output_dir, 'model_float32.tflite')
    with open(float_path, 'wb') as f:
        f.write(tflite_float)
    print(f"  float32: {len(tflite_float)/1024:.0f} KB")
    
    # 验证 TFLite
    interpreter = tf.lite.Interpreter(model_content=tflite_float)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 跑几个样本验证
    correct = 0
    for i in range(min(50, len(X_test_feat))):
        inp = X_test_feat[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
        pred_label = 1 if pred > 0.5 else 0
        if pred_label == y_test[i]:
            correct += 1
    
    tflite_acc = correct / min(50, len(X_test_feat))
    print(f"  TFLite 验证准确率 ({min(50, len(X_test_feat))}样本): {tflite_acc*100:.1f}%")
    
    # 生成 C 头文件 (float32)
    esp32_dir = os.path.join(os.path.dirname(output_dir), 'esp32')
    os.makedirs(esp32_dir, exist_ok=True)
    
    header_path = os.path.join(esp32_dir, 'snore_model.h')
    with open(float_path, 'rb') as f:
        model_data = f.read()
    
    with open(header_path, 'w') as f:
        f.write('/* ============================================================\n')
        f.write(' * 酣眠 SnoozMate 鼾声检测模型\n')
        f.write(' * Float32 TFLite Micro 模型\n')
        f.write(f' * 模型大小: {len(model_data)} bytes ({len(model_data)/1024:.0f} KB)\n')
        f.write(f' * 输入: {input_details[0]["shape"].tolist()} float32\n')
        f.write(f' * 输出: {output_details[0]["shape"].tolist()} float32\n')
        f.write(f' * 参数量: {model.count_params():,}\n')
        f.write(f' * 测试准确率: {test_acc*100:.2f}%\n')
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
    print(f"  模型大小: {len(model_data)} bytes ({len(model_data)/1024:.0f} KB)")
    
    print("\n" + "=" * 60)
    print(f"  ✅ 完成！测试准确率: {test_acc*100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
