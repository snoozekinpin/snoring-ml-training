# -*- coding: utf-8 -*-
"""v4 vs v31 推理对比：用同一音频测两条模型"""
import os, wave, numpy as np
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

DATA = r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\dataset\adrianagaler\snore"
m_v4 = tf.keras.models.load_model(r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output\model\best_model_v4.h5", compile=False)
m_v31 = tf.keras.models.load_model(r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\output\model\best_model_v31.h5", compile=False)

# 特征
SR, N_FFT, HOP, N_MELS = 16000, 512, 256, 30
EPS = 1e-9

def hz_to_mel(hz): return 2595 * np.log10(1 + hz/700)
def mel_to_hz(m): return 700*(10**(m/2595)-1)
mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(SR/2), N_MELS+2)
hz_pts = mel_to_hz(mel_pts)
bins = np.floor((N_FFT+1)*hz_pts/SR).astype(int)
fb = np.zeros((N_MELS, N_FFT//2+1), dtype=np.float32)
for i in range(N_MELS):
    l, c, r = bins[i], bins[i+1], bins[i+2]
    for j in range(l, c):
        if j<fb.shape[1]: fb[i,j] = (j-l)/(c-l)
    for j in range(c, r):
        if j<fb.shape[1]: fb[i,j] = (r-j)/(r-c)
WIN = np.hanning(N_FFT).astype(np.float32)

def feat(audio):
    if len(audio) <= N_FFT: audio = np.pad(audio, (0, N_FFT-len(audio)+1))
    frames = 1 + (len(audio)-N_FFT)//HOP
    spec = np.empty((frames, N_FFT//2+1), dtype=np.float32)
    for i in range(frames):
        f = audio[i*HOP:i*HOP+N_FFT]*WIN
        spec[i] = np.abs(np.fft.rfft(f)[:N_FFT//2+1])**2
    mel = spec @ fb.T
    mel = np.log(mel+EPS)
    return ((mel-mel.min())/(mel.max()-mel.min()+EPS)).astype(np.float32)

# 测 5 个鼾声 + 5 个噪声
import random
snore_files = sorted(Path(DATA).glob("*.wav"))[:5]
noise_files = sorted(Path(r"D:\Codex\BOSS\AIXORIGIN\snoring-ml-training\dataset\adrianagaler\noise").glob("*.wav"))[:5]

print(f"{'文件':>25} {'真实':>6} {'v4概率':>8} {'v31概率':>8} {'v4@0.65':>8} {'v31@0.3':>8}")
for f in list(snore_files) + list(noise_files):
    with wave.open(str(f)) as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768
    if ch==2: audio = audio.reshape(-1,2).mean(axis=1)
    if len(audio)<16000: audio = np.pad(audio,(0,16000-len(audio)))
    audio = audio[:16000]
    x = feat(audio).reshape(1, 61, 30)
    p4 = m_v4.predict(x, verbose=0)[0,0]
    p31 = m_v31.predict(x, verbose=0)[0,0]
    true_label = "snore" if "snore" in str(f) else "noise"
    print(f"{f.name[:24]:>25} {true_label:>6} {p4:>8.3f} {p31:>8.3f} {'YES' if p4>0.65 else 'no':>8} {'YES' if p31>0.3 else 'no':>8}")

