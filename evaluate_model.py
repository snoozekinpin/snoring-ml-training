"""
酣眠 SnoozMate · 模型评估与验证脚本
==================================
训练完成后运行，生成完整的评估报告：
  - 准确率 / 精确率 / 召回率 / F1
  - 混淆矩阵
  - ROC 曲线 + AUC
  - 各信噪比下的准确率
  - 推理速度测试
  - 模型大小统计

用法:
  python evaluate_model.py --model output/model/best_model.h5 --test_dir dataset/
  python evaluate_model.py --model output/model/model_int8.tflite --test_dir dataset/ --tflite
"""

import argparse
import os
import json
import time
import numpy as np
from pathlib import Path


def load_test_data(test_dir, sample_limit=None):
    """加载测试数据"""
    from rule_detector_test import load_wav
    from train import N_SAMPLES, pad_or_trim
    
    test_dir = Path(test_dir)
    X, y, filenames = [], [], []
    
    for label, class_name in enumerate(['snore', 'noise']):
        class_dir = test_dir / class_name
        if not class_dir.is_dir():
            continue
        
        files = sorted(list(class_dir.glob('*.wav')))
        if sample_limit:
            files = files[:sample_limit]
        
        print(f"  加载 {class_name}: {len(files)} 个文件")
        
        for fpath in files:
            try:
                audio = load_wav(str(fpath))
            except Exception as e:
                continue
            
            # 切成 1 秒片段
            for start in range(0, len(audio) - N_SAMPLES + 1, N_SAMPLES):
                segment = audio[start:start + N_SAMPLES]
                if np.max(np.abs(segment)) < 0.01:
                    continue
                X.append(segment)
                y.append(label)
                filenames.append(fpath.name)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), filenames


def extract_features(X_audio):
    """提取 log-mel 特征"""
    from train import audio_to_features
    return audio_to_features(X_audio)


def evaluate_keras_model(model_path, X_test, y_test):
    """评估 Keras 模型"""
    import tensorflow as tf
    
    print(f"\n加载 Keras 模型: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()
    
    X_feat = extract_features(X_test)
    print(f"\n测试集 shape: {X_feat.shape}")
    
    # 推理速度测试
    print("\n推理速度测试...")
    n_warmup = 10
    n_test = 100
    
    for i in range(n_warmup):
        model.predict(X_feat[i:i+1], verbose=0)
    
    start = time.time()
    for i in range(n_test):
        model.predict(X_feat[i % len(X_feat): i % len(X_feat) + 1], verbose=0)
    elapsed = time.time() - start
    avg_ms = elapsed / n_test * 1000
    print(f"  平均推理时间: {avg_ms:.1f} ms/样本")
    
    # 预测
    y_pred_prob = model.predict(X_feat, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    return y_pred, y_pred_prob, avg_ms


def evaluate_tflite_model(model_path, X_test, y_test):
    """评估 TFLite 模型"""
    import tensorflow as tf
    
    print(f"\n加载 TFLite 模型: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_dtype = input_details[0]['dtype']
    print(f"  输入类型: {input_dtype}")
    print(f"  输入形状: {input_details[0]['shape']}")
    
    X_feat = extract_features(X_test)
    
    # 推理速度测试
    print("\n推理速度测试...")
    n_warmup = 10
    n_test = 100
    
    input_data = X_feat[0:1].astype(input_dtype)
    for i in range(n_warmup):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
    
    start = time.time()
    for i in range(n_test):
        idx = i % len(X_feat)
        input_data = X_feat[idx:idx+1].astype(input_dtype)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        interpreter.get_tensor(output_details[0]['index'])
    elapsed = time.time() - start
    avg_ms = elapsed / n_test * 1000
    print(f"  平均推理时间: {avg_ms:.1f} ms/样本")
    
    # 预测全部
    y_pred_prob = []
    for i in range(len(X_feat)):
        input_data = X_feat[i:i+1].astype(input_dtype)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        if output.dtype == np.uint8 or output.dtype == np.int8:
            # 反量化
            scale = output_details[0]['quantization'][0]
            zero_point = output_details[0]['quantization'][1]
            prob = (output.astype(np.float32) - zero_point) * scale
            y_pred_prob.append(prob.flatten()[0])
        else:
            y_pred_prob.append(output.flatten()[0])
    
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    return y_pred, y_pred_prob, avg_ms


def print_metrics(y_true, y_pred, y_prob, avg_ms, model_size_kb=0):
    """打印评估指标"""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                  f1_score, confusion_matrix, roc_auc_score, roc_curve)
    
    print("\n" + "=" * 60)
    print("  评估结果")
    print("=" * 60)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n  准确率 (Accuracy):  {acc*100:.2f}%")
    print(f"  精确率 (Precision): {prec*100:.2f}%")
    print(f"  召回率 (Recall):    {rec*100:.2f}%")
    print(f"  F1 分数:            {f1*100:.2f}%")
    
    try:
        auc = roc_auc_score(y_true, y_prob)
        print(f"  ROC-AUC:            {auc*100:.2f}%")
    except:
        auc = 0
        print(f"  ROC-AUC:            N/A")
    
    print(f"\n  平均推理时间: {avg_ms:.1f} ms")
    if model_size_kb:
        print(f"  模型大小:     {model_size_kb:.0f} KB")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  混淆矩阵:")
    print(f"                 预测鼾声   预测噪音")
    print(f"  实际鼾声     {cm[0][0]:>6}    {cm[0][1]:>6}")
    print(f"  实际噪音     {cm[1][0]:>6}    {cm[1][1]:>6}")
    
    # 误报详情
    if cm[1][0] > 0:
        print(f"\n  假阳性率 (FPR): {cm[1][0]/np.sum(cm[1])*100:.2f}%")
    if cm[0][1] > 0:
        print(f"  漏检率 (FNR):   {cm[0][1]/np.sum(cm[0])*100:.2f}%")
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
        'confusion_matrix': cm.tolist(),
        'avg_inference_ms': avg_ms,
    }


def evaluate_by_snr(y_true, y_pred, X_audio, n_bins=5):
    """按信噪比分层评估"""
    # 计算每个样本的 SNR（相对于底噪的估计）
    energies = np.array([20*np.log10(np.sqrt(np.mean(x**2))+1e-10) for x in X_audio])
    # 估计底噪（最低10百分位）
    noise_floor = np.percentile(energies, 10)
    snrs = energies - noise_floor
    
    # 分层
    bins = np.percentile(snrs, np.linspace(0, 100, n_bins + 1))
    
    print(f"\n{'='*60}")
    print(f"  按信噪比分层的准确率")
    print(f"{'='*60}")
    print(f"  {'SNR范围 (dB)':<20} {'样本数':>8} {'准确率':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*10}")
    
    for i in range(n_bins):
        mask = (snrs >= bins[i]) & (snrs < bins[i+1])
        if np.sum(mask) == 0:
            continue
        acc = np.mean(y_true[mask] == y_pred[mask])
        print(f"  {bins[i]:5.1f} - {bins[i+1]:5.1f}     {np.sum(mask):>6}   {acc*100:>8.1f}%")


def save_report(metrics, output_path):
    """保存评估报告为 JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='鼾声检测模型评估')
    parser.add_argument('--model', type=str, required=True, help='模型路径 (.h5 或 .tflite)')
    parser.add_argument('--test_dir', type=str, default='dataset', help='测试集目录')
    parser.add_argument('--tflite', action='store_true', help='使用 TFLite 模式')
    parser.add_argument('--sample_limit', type=int, default=None, help='每个类别最大样本数')
    parser.add_argument('--output', type=str, default='output/model/evaluation_report.json',
                        help='报告输出路径')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"[错误] 模型文件不存在: {args.model}")
        return
    
    # 自动判断 tflite
    if args.model.endswith('.tflite'):
        args.tflite = True
    
    # 模型大小
    model_size_kb = os.path.getsize(args.model) / 1024
    
    # 加载数据
    print("=" * 60)
    print("  酣眠 SnoozMate · 模型评估")
    print("=" * 60)
    print(f"\n模型: {args.model}")
    print(f"模型大小: {model_size_kb:.0f} KB")
    print(f"测试集: {args.test_dir}")
    
    print("\n[1/3] 加载测试数据...")
    X_test, y_test, filenames = load_test_data(args.test_dir, args.sample_limit)
    
    if len(X_test) < 10:
        print("\n[错误] 测试样本太少（<10），请先准备数据")
        print("  运行: python download_datasets.py --synth")
        return
    
    print(f"\n总样本: {len(X_test)}")
    print(f"  鼾声: {np.sum(y_test==0)}")
    print(f"  噪音: {np.sum(y_test==1)}")
    
    # 评估
    print("\n[2/3] 模型评估...")
    if args.tflite:
        y_pred, y_prob, avg_ms = evaluate_tflite_model(args.model, X_test, y_test)
    else:
        y_pred, y_prob, avg_ms = evaluate_keras_model(args.model, X_test, y_test)
    
    # 指标
    print("\n[3/3] 生成报告...")
    metrics = print_metrics(y_test, y_pred, y_prob, avg_ms, model_size_kb)
    
    # 按 SNR 分层
    evaluate_by_snr(y_test, y_pred, X_test)
    
    # 保存
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    save_report(metrics, args.output)
    
    print("\n" + "=" * 60)
    
    if metrics['accuracy'] >= 0.90:
        print("  🎉 优秀！准确率超过 90%")
    elif metrics['accuracy'] >= 0.80:
        print("  ✅ 良好，可以部署")
    elif metrics['accuracy'] >= 0.70:
        print("  ⚠️ 一般，建议增加数据或调参")
    else:
        print("  ❌ 较差，需要优化")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
