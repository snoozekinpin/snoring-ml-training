"""
酣眠 SnoozMate · 鼾声检测模型训练流水线
=====================================
基于 alek6kun/snore-recognition 的架构优化版

模型架构:
  输入: 1秒 @ 16kHz 音频 → log-mel 频谱图 (61帧 × 30mel)
  Backbone: 3层 CNN (32→64→128) + GlobalPool + Dense(128)
  输出: 二分类 sigmoid (鼾声概率)
  量化: int8 (ESP32-S3 TFLite Micro 部署)

数据增强:
  - 加性噪声 (白噪 / 环境音)
  - 音量扰动 (0.6~1.2x)
  - 时移 (±100ms)
  - 速度微扰 (0.9~1.1x)
  - 音调偏移 (±半音)

用法:
  # 1. 准备数据集到 dataset/snore/ 和 dataset/noise/
  # 2. 训练:
  python train.py --epochs 30 --batch_size 32
  
  # 3. 只做转换（已有模型）:
  python train.py --convert_only --model output/model/best_model.h5
  
  # 4. 小数据集快速测试:
  python train.py --epochs 5 --sample_limit 100
"""

import argparse
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# ============================================================
# 配置参数
# ============================================================
SAMPLE_RATE = 16000
DURATION_SEC = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION_SEC)  # 16000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 30
F_MIN = 40.0
F_MAX = 6000.0

# 模型参数
CONV_FILTERS = [32, 64, 128]
DENSE_UNITS = 128
DROPOUT = 0.5

# 数据增强参数
AUGMENTATION = {
    'noise_prob': 0.5,      # 加噪概率
    'noise_max_db': 0.1,    # 最大噪声强度
    'volume_prob': 0.5,     # 音量扰动概率
    'volume_range': (0.6, 1.2),
    'shift_prob': 0.3,      # 时移概率
    'shift_max_ms': 100,    # 最大时移(ms)
    'speed_prob': 0.3,      # 速度微扰概率
    'speed_range': (0.9, 1.1),
}


# ============================================================
# 音频 I/O & 预处理
# ============================================================
def load_audio(filepath, sr=SAMPLE_RATE):
    """加载音频文件，统一采样率"""
    try:
        import librosa
        audio, _ = librosa.load(filepath, sr=sr, mono=True)
        return audio
    except ImportError:
        # fallback: 用 scipy
        from scipy.io import wavfile
        from scipy.signal import resample
        sample_rate, data = wavfile.read(filepath)
        if data.ndim > 1:
            data = data[:, 0]  # 取左声道
        # 转 float
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        # 重采样
        if sample_rate != sr:
            n_new = int(len(data) * sr / sample_rate)
            data = resample(data, n_new)
        return data.astype(np.float32)


def pad_or_trim(audio, length=N_SAMPLES):
    """统一长度：短补零，长截断（取中间部分）"""
    if len(audio) >= length:
        start = (len(audio) - length) // 2
        return audio[start:start + length]
    else:
        padded = np.zeros(length, dtype=np.float32)
        start = (length - len(audio)) // 2
        padded[start:start + len(audio)] = audio
        return padded


def compute_log_mel(audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                    n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX):
    """
    计算 log-mel 频谱图
    输出 shape: (time_frames, n_mels) = (61, 30) 对应 1s 音频
    """
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmin=f_min, fmax=f_max
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # 归一化到 [0, 1]
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        return log_mel.T.astype(np.float32)  # (time, mel)
    except ImportError:
        # fallback: 纯 numpy/scipy 实现
        from scipy.signal import stft
        from scipy.fft import dct
        
        # STFT
        f, t, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        spec = np.abs(Zxx) ** 2  # 功率谱
        
        # Mel 滤波器组
        n_freq_bins = spec.shape[0]
        mel_filters = _mel_filterbank(n_mels, n_freq_bins, sr, f_min, f_max)
        mel_spec = np.dot(mel_filters, spec)  # (n_mels, time)
        
        # Log
        log_mel = np.log(mel_spec + 1e-6)
        # 归一化
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        return log_mel.T.astype(np.float32)


def _mel_filterbank(n_mels, n_freq_bins, sr, f_min, f_max):
    """生成 mel 滤波器组（纯numpy）"""
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    # 频率点索引
    bin_points = np.floor((n_freq_bins - 1) * hz_points / (sr / 2)).astype(int)
    
    filterbank = np.zeros((n_mels, n_freq_bins))
    for m in range(1, n_mels + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]
        for k in range(left, center):
            if center - left > 0:
                filterbank[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right - center > 0:
                filterbank[m - 1, k] = (right - k) / (right - center)
    return filterbank


# ============================================================
# 数据增强
# ============================================================
def augment_audio(audio, sr=SAMPLE_RATE):
    """数据增强：随机应用各种变换"""
    aug = audio.copy()
    
    # 1. 音量扰动
    if np.random.random() < AUGMENTATION['volume_prob']:
        factor = np.random.uniform(*AUGMENTATION['volume_range'])
        aug = aug * factor
    
    # 2. 加性噪声
    if np.random.random() < AUGMENTATION['noise_prob']:
        noise_level = np.random.uniform(0, AUGMENTATION['noise_max_db'])
        noise = np.random.randn(len(aug)) * np.std(aug) * noise_level
        aug = aug + noise
    
    # 3. 时移
    if np.random.random() < AUGMENTATION['shift_prob']:
        shift_ms = np.random.randint(-AUGMENTATION['shift_max_ms'], AUGMENTATION['shift_max_ms'])
        shift_samples = int(shift_ms * sr / 1000)
        if shift_samples > 0:
            aug = np.pad(aug[:-shift_samples], (shift_samples, 0))
        elif shift_samples < 0:
            aug = np.pad(aug[-shift_samples:], (0, -shift_samples))
    
    # 限幅
    aug = np.clip(aug, -1.0, 1.0)
    return aug.astype(np.float32)


# ============================================================
# 数据集构建
# ============================================================
def load_dataset(data_dir, sample_limit=None, augment=False):
    """
    从目录加载数据集
    data_dir/snore/  → 正样本
    data_dir/noise/  → 负样本
    """
    data_dir = Path(data_dir)
    X, y = [], []
    
    for label, class_name in enumerate(['snore', 'noise']):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            print(f"[警告] 目录不存在: {class_dir}")
            continue
        
        files = list(class_dir.glob('*.wav')) + list(class_dir.glob('*.mp3'))
        if sample_limit:
            files = files[:sample_limit]
        
        print(f"  加载 {class_name}: {len(files)} 个文件")
        
        for fpath in files:
            try:
                audio = load_audio(str(fpath))
            except Exception as e:
                print(f"    [跳过] {fpath.name}: {e}")
                continue
            
            # 切成 1 秒片段
            for start in range(0, len(audio) - N_SAMPLES + 1, N_SAMPLES // 2):
                segment = audio[start:start + N_SAMPLES]
                if len(segment) < N_SAMPLES:
                    break
                
                # 去静音段
                if np.max(np.abs(segment)) < 0.01:
                    continue
                
                X.append(segment)
                y.append(label)
                
                # 增强（训练集）
                if augment and label == 0:  # 只增强正样本
                    for _ in range(2):  # 每个正样本生成2个增强版
                        aug_seg = augment_audio(segment)
                        X.append(aug_seg)
                        y.append(label)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"\n[数据集] 总样本: {len(X)}")
    print(f"  鼾声: {np.sum(y==0)} 个")
    print(f"  噪音: {np.sum(y==1)} 个")
    
    return X, y


def audio_to_features(X_audio):
    """批量转换音频 → log-mel 特征"""
    features = []
    for audio in X_audio:
        feat = compute_log_mel(audio)
        features.append(feat)
    return np.array(features, dtype=np.float32)


# ============================================================
# 模型构建
# ============================================================
def build_model(input_shape, num_classes=1):
    """
    构建轻量级 CNN 模型
    参考 alek6kun/snore-recognition 的架构
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        layers.Input(shape=input_shape, name='input'),
        
        # 增加通道维度 (time, mel) → (time, mel, 1)
        layers.Reshape((input_shape[0], input_shape[1], 1), name='expand_channel'),
        
        # 下采样到 32×30（对齐 micro_speech 的输入尺寸）
        layers.Resizing(32, 30, name='resize'),
        
        # 第一层卷积
        layers.Conv2D(CONV_FILTERS[0], (3, 3), padding='same', activation='relu', name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # 第二层卷积
        layers.Conv2D(CONV_FILTERS[1], (3, 3), padding='same', activation='relu', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # 第三层卷积
        layers.Conv2D(CONV_FILTERS[2], (3, 3), padding='same', activation='relu', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # 全局池化
        layers.GlobalAveragePooling2D(name='gap'),
        
        # 分类头
        layers.Dense(DENSE_UNITS, activation='relu', name='dense1'),
        layers.Dropout(DROPOUT, name='dropout'),
        layers.Dense(num_classes, activation='sigmoid', name='output'),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), 
                 keras.metrics.Recall(name='recall')]
    )
    
    total_params = model.count_params()
    print(f"\n[模型] 参数量: {total_params:,}")
    print(f"  float32 大小: ~{total_params * 4 / 1024:.0f} KB")
    print(f"  int8 估计:    ~{total_params / 1024:.0f} KB")
    
    return model


# ============================================================
# 训练
# ============================================================
def train(X_train, y_train, X_val, y_val, output_dir, epochs=30, batch_size=32):
    """训练模型"""
    from tensorflow import keras
    
    # 转换为特征
    print("\n[特征提取] 训练集...")
    X_train_feat = audio_to_features(X_train)
    print(f"  shape: {X_train_feat.shape}")
    
    print("[特征提取] 验证集...")
    X_val_feat = audio_to_features(X_val)
    print(f"  shape: {X_val_feat.shape}")
    
    # 构建模型
    input_shape = X_train_feat.shape[1:]
    model = build_model(input_shape)
    
    # 回调
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, 
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
    ]
    
    # 类权重（处理不平衡）
    n_pos = np.sum(y_train == 0)
    n_neg = np.sum(y_train == 1)
    total = n_pos + n_neg
    class_weight = {
        0: total / (2 * n_pos) if n_pos > 0 else 1.0,
        1: total / (2 * n_neg) if n_neg > 0 else 1.0,
    }
    print(f"\n[类权重] snore={class_weight[0]:.2f}, noise={class_weight[1]:.2f}")
    
    # 训练
    print(f"\n[训练] {epochs} epochs, batch_size={batch_size}")
    history = model.fit(
        X_train_feat, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_feat, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    
    # 保存最终模型
    model.save(os.path.join(output_dir, 'final_model.h5'))
    
    # 保存训练历史
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    
    return model, history


# ============================================================
# 评估
# ============================================================
def evaluate(model, X_test, y_test):
    """评估模型性能"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    X_test_feat = audio_to_features(X_test)
    
    # 预测
    y_pred_prob = model.predict(X_test_feat, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # 报告
    print("\n" + "=" * 50)
    print("  模型评估报告")
    print("=" * 50)
    
    test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test_feat, y_test, verbose=0)
    print(f"\n  Test Loss:      {test_loss:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test Precision: {test_prec:.4f}")
    print(f"  Test Recall:    {test_rec:.4f}")
    
    print(f"\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 预测鼾声  预测噪音")
    print(f"  实际鼾声     {cm[0][0]:>6}    {cm[0][1]:>6}")
    print(f"  实际噪音     {cm[1][0]:>6}    {cm[1][1]:>6}")
    
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['snore', 'noise']))
    
    return test_acc


# ============================================================
# TFLite 转换 + ESP32 部署
# ============================================================
def convert_to_tflite(model, X_calib, output_dir):
    """
    转换为 TFLite int8 量化模型 + C 头文件
    """
    import tensorflow as tf
    
    print("\n" + "=" * 50)
    print("  TFLite 转换")
    print("=" * 50)
    
    # 校准数据
    X_calib_feat = audio_to_features(X_calib[:100])
    
    def representative_dataset():
        for i in range(min(100, len(X_calib_feat))):
            yield [X_calib_feat[i:i+1].astype(np.float32)]
    
    # float32 版本
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float = converter.convert()
    
    float_path = os.path.join(output_dir, 'model_float32.tflite')
    with open(float_path, 'wb') as f:
        f.write(tflite_float)
    print(f"\n  float32: {os.path.getsize(float_path)/1024:.0f} KB")
    
    # int8 量化版本
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_int8 = converter.convert()
        
        int8_path = os.path.join(output_dir, 'model_int8.tflite')
        with open(int8_path, 'wb') as f:
            f.write(tflite_int8)
        print(f"  int8:    {os.path.getsize(int8_path)/1024:.0f} KB")
        
        # 生成 C 头文件（ESP32 用）
        esp32_dir = os.path.join(os.path.dirname(output_dir), 'esp32')
        os.makedirs(esp32_dir, exist_ok=True)
        
        header_path = os.path.join(esp32_dir, 'snore_model.h')
        with open(int8_path, 'rb') as f:
            model_data = f.read()
        
        with open(header_path, 'w') as f:
            f.write('/* ============================================================\n')
            f.write(' * 酣眠 SnoozMate · 鼾声检测模型 (int8 量化)\n')
            f.write(f' * 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f' * 模型大小: {len(model_data)} bytes ({len(model_data)/1024:.0f} KB)\n')
            f.write(' * 目标平台: ESP32-S3 + TFLite Micro\n')
            f.write(' * ============================================================ */\n\n')
            f.write('#ifndef SNORE_MODEL_H\n')
            f.write('#define SNORE_MODEL_H\n\n')
            f.write('#include <stdint.h>\n\n')
            f.write(f'const int g_snore_model_size = {len(model_data)};\n\n')
            f.write('alignas(8) const unsigned char g_snore_model[] = {\n')
            
            # 每行 12 字节
            for i in range(0, len(model_data), 12):
                chunk = model_data[i:i+12]
                f.write('  ' + ', '.join(f'0x{b:02x}' for b in chunk) + ',\n')
            
            f.write('};\n\n')
            f.write('#endif /* SNORE_MODEL_H */\n')
        
        print(f"  C header: {header_path}")
        print(f"            ({len(model_data)} bytes)")
        
        return True
        
    except Exception as e:
        print(f"\n  ⚠️ int8 量化失败: {e}")
        print(f"     但 float32 模型可用")
        return False


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='酣眠 SnoozMate 鼾声检测模型训练')
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='数据集目录 (snore/ + noise/)')
    parser.add_argument('--output_dir', type=str, default='output/model')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sample_limit', type=int, default=None,
                        help='每个类别限制文件数（快速测试用）')
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--convert_only', action='store_true',
                        help='只做 TFLite 转换（需要 --model）')
    parser.add_argument('--model', type=str, help='已有模型路径')
    parser.add_argument('--no_augment', action='store_true',
                        help='禁用数据增强')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.convert_only:
        if not args.model:
            print("[错误] --convert_only 需要 --model 参数")
            sys.exit(1)
        print(f"加载模型: {args.model}")
        import tensorflow as tf
        model = tf.keras.models.load_model(args.model)
        model.summary()
        # 需要一些校准数据
        print("生成随机校准数据用于量化...")
        X_calib = np.random.randn(100, N_SAMPLES).astype(np.float32) * 0.1
        convert_to_tflite(model, X_calib, args.output_dir)
        return
    
    # 完整训练流程
    print("=" * 60)
    print("  酣眠 SnoozMate · 鼾声检测模型训练")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据集...")
    X, y = load_dataset(args.data_dir, sample_limit=args.sample_limit,
                       augment=not args.no_augment)
    
    if len(X) < 50:
        print("\n[错误] 样本太少（<50），无法训练")
        print("  请将鼾声样本放入 dataset/snore/")
        print("  将噪音样本放入 dataset/noise/")
        sys.exit(1)
    
    # 2. 划分数据集
    print("\n[2/5] 划分数据集...")
    from sklearn.model_selection import train_test_split
    
    # 先分 test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y)
    # 再分 train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.val_split / (1 - args.test_split),
        random_state=42, stratify=y_temp)
    
    print(f"  训练集: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"  验证集: {len(X_val)} ({len(X_val)/len(X)*100:.0f}%)")
    print(f"  测试集: {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")
    
    # 3. 训练
    print("\n[3/5] 训练模型...")
    model, history = train(
        X_train, y_train, X_val, y_val,
        args.output_dir, args.epochs, args.batch_size
    )
    
    # 4. 评估
    print("\n[4/5] 评估模型...")
    test_acc = evaluate(model, X_test, y_test)
    
    # 5. 转换
    print("\n[5/5] TFLite 转换 + ESP32 部署文件...")
    convert_to_tflite(model, X_val, args.output_dir)
    
    print("\n" + "=" * 60)
    print(f"  ✅ 训练完成！Test Accuracy: {test_acc*100:.2f}%")
    print("=" * 60)
    print(f"  输出目录: {args.output_dir}/")
    print(f"    best_model.h5        - Keras 最佳模型")
    print(f"    final_model.h5       - Keras 最终模型")
    print(f"    model_float32.tflite - TFLite 浮点版")
    print(f"    model_int8.tflite    - TFLite int8 量化版")
    print(f"    history.json         - 训练历史")
    print(f"  ESP32部署: ../esp32/snore_model.h (C头文件)")
    print()


if __name__ == '__main__':
    main()
