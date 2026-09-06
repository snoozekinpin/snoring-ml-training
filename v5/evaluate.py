"""Clip-level metrics, robustness sweeps and int8 inference (spec section 7)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from v5 import features as F
from v5.data.augment import ISM_ORDER, apply_rir, mix_noise, normalise_direct_path
from v5.data.dataset import precompute_features


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def recall_at_fpr(y, p, max_fpr: float = 0.02) -> float:
    fpr, tpr, _ = roc_curve(np.asarray(y).astype(int), np.asarray(p, dtype=np.float64))
    ok = fpr <= max_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def clip_metrics(y, p, tau: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
        "recall_at_fpr2": recall_at_fpr(y, p, 0.02) if two_classes else float("nan"),
        "tau": float(tau),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fnr": fn / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "cm": [[tn, fp], [fn, tp]],
    }


def predict_probs(model, X, batch: int = 256) -> np.ndarray:
    return sigmoid(model.predict(X, batch_size=batch, verbose=0).ravel())


def make_distance_rirs(distance_m: float, n: int = 5, seed: int = 0, max_len: int = 8000) -> list[np.ndarray]:
    import pyroomacoustics as pra

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        dims = [4.0, 4.0, 2.6]
        e_abs, _ = pra.inverse_sabine(0.3, dims)
        room = pra.ShoeBox(dims, fs=F.SR, materials=pra.Material(e_abs), max_order=ISM_ORDER)
        mic = np.array([2.0, 1.0, 0.7])
        az = rng.uniform(-np.pi / 3, np.pi / 3)
        src = mic + np.array([distance_m * np.sin(az), distance_m * np.cos(az), -0.15])
        room.add_source(src.tolist())
        room.add_microphone(mic.tolist())
        room.compute_rir()
        h = np.asarray(room.rir[0][0], dtype=np.float32)
        h = h[:max_len] if len(h) >= max_len else np.pad(h, (0, max_len - len(h)))
        out.append(normalise_direct_path(h, distance_m))
    return out


def robustness_sweep(model, audio_i16, y, noise_i16, tau: float, snrs=(20, 10, 5, 0), distances=(0.5, 1.0, 1.5), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = [F.int16_to_float(a) for a in audio_i16]
    result = {"clean": clip_metrics(y, predict_probs(model, precompute_features(audio_i16)), tau), "snr": {}, "distance": {}}
    for snr in snrs:
        mixed = [mix_noise(xi, F.int16_to_float(noise_i16[int(rng.integers(0, len(noise_i16)))]), float(snr)) for xi in x]
        result["snr"][str(snr)] = clip_metrics(y, predict_probs(model, precompute_features([F.float_to_int16(m) for m in mixed])), tau)
    for d in distances:
        rirs = make_distance_rirs(d, n=5, seed=seed)
        conv = [apply_rir(xi, rirs[i % len(rirs)]) for i, xi in enumerate(x)]
        result["distance"][str(d)] = clip_metrics(y, predict_probs(model, precompute_features([F.float_to_int16(c) for c in conv])), tau)
    return result


def make_interpreter(tflite):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # fallback for older TensorFlow builds
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    if isinstance(tflite, (bytes, bytearray)):
        return Interpreter(model_content=bytes(tflite))
    return Interpreter(model_path=str(tflite))


def int8_probs(tflite, X) -> np.ndarray:
    interp = make_interpreter(tflite)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    probs = np.empty(len(X), np.float64)
    for i, xi in enumerate(np.asarray(X, dtype=np.float32)):
        q = F.quantize(xi.reshape(F.N_FRAMES, F.N_MELS), float(in_scale), int(in_zp)).reshape(inp["shape"])
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        o = interp.get_tensor(out["index"]).astype(np.float64).reshape(-1)[0]
        probs[i] = float((o - out_zp) * out_scale)
    return probs


def int8_parity(p_fp, p_int8, y, tau: float) -> dict:
    p_fp, p_int8, y = np.asarray(p_fp, dtype=np.float64), np.asarray(p_int8, dtype=np.float64), np.asarray(y).astype(int)
    two = len(np.unique(y)) == 2
    delta = abs(roc_auc_score(y, p_fp) - roc_auc_score(y, p_int8)) if two else 0.0
    agreement = float(((p_fp >= tau) == (p_int8 >= tau)).mean())
    max_abs = float(np.abs(p_fp - p_int8).max())
    return {"delta_auc": float(delta), "agreement": agreement, "max_abs_diff": max_abs, "passed": bool(delta < 0.005 and agreement >= 0.99 and max_abs <= 0.05)}
