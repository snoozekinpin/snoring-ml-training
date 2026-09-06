"""Streaming benchmark: synthetic nights -> features -> predictor -> FSM -> product metrics (spec section 7)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v5 import features as F
from v5.config import load_config, resolve
from v5.data import manifest as M
from v5.data.sources import decode
from v5.evaluate import int8_probs, make_distance_rirs, predict_probs
from v5.nights import NightSpec, generate_night
from v5.streaming import EpisodeFsm, FsmParams


def window_features(audio, hop: int = F.STREAM_HOP) -> np.ndarray:
    n = 1 + (len(audio) - F.WIN) // hop
    return np.stack([F.extract(audio[k * hop: k * hop + F.WIN]) for k in range(n)]).astype(np.float32)[..., None]


def tick_end_time(tick: int, tick_s: float = 0.5) -> float:
    return (tick + 2) * tick_s  # window k spans [k*hop, k*hop + 1 s)


def run_fsm(p_seq, params: FsmParams):
    fsm = EpisodeFsm(params)
    tick_s = params.tick_ms / 1000.0
    events, actives = [], []
    for p in p_seq:
        ev = fsm.tick(float(p))
        if ev is not None:
            ev["t_end"] = tick_end_time(ev["tick"], tick_s)
            events.append(ev)
        actives.append(fsm.active)
    return events, actives


def match_events(events, episodes, tol_s: float):
    starts = sorted((ev for ev in events if ev["type"] == "episode_start"), key=lambda ev: ev["t_end"])
    used, matches, unmatched = set(), [], []
    for ev in starts:
        found = next((j for j, (s, e) in enumerate(episodes) if j not in used and s - tol_s <= ev["t_end"] <= e + tol_s), None)
        if found is None:
            unmatched.append(ev)
        else:
            used.add(found)
            matches.append((found, ev))
    return matches, unmatched


def _stats(values):
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.percentile(values, 90))


def score_night(events, actives, episodes, tick_s: float = 0.5, tol_s: float = 5.0, duration_s: float = 3600.0) -> dict:
    matches, unmatched = match_events(events, episodes, tol_s)
    matched = dict(matches)
    ends_by_id = {ev.get("episode_id"): ev for ev in events if ev["type"] == "episode_end"}
    latency, stop_lat, end_delay = [], [], []
    for j, (s, e) in enumerate(episodes):
        if j not in matched:
            continue
        start_ev = matched[j]
        latency.append(max(0.0, start_ev["t_end"] - s))
        drop = next((k for k in range(int(start_ev["tick"]), len(actives)) if not actives[k]), None)
        if drop is not None:
            stop_lat.append(max(0.0, tick_end_time(drop, tick_s) - e))
        end_ev = ends_by_id.get(start_ev.get("episode_id"))
        if end_ev is not None:
            end_delay.append(max(0.0, end_ev["t_end"] - e))
    lat_m, lat_p90 = _stats(latency)
    stop_m, stop_p90 = _stats(stop_lat)
    hours = duration_s / 3600.0
    return {
        "n_episodes": len(episodes), "detected": len(matched), "detection_rate": (len(matched) / len(episodes)) if episodes else float("nan"),
        "confirm_latency_mean_s": lat_m, "confirm_latency_p90_s": lat_p90,
        "false_confirms": len(unmatched), "false_confirms_per_hour": len(unmatched) / hours,
        "stop_latency_mean_s": stop_m, "stop_latency_p90_s": stop_p90, "end_event_delay_mean_s": _stats(end_delay)[0],
    }


def bench_bed_paths(rows) -> list[str]:
    """MS-SNSD negatives only: bench also holds WHLTalent snore recordings, which must never become a bed."""
    return sorted({r["path"] for r in rows if r["split"] == "bench" and r["source"] == "mssnsd" and r["label"] == 0})


def _bench_beds(rows, data_dir) -> list[np.ndarray]:
    return [decode(Path(data_dir) / p) for p in bench_bed_paths(rows)]


def _aggregate(nights, snrs) -> dict:
    out = {}
    for snr in snrs:
        sub = [m for m in nights if m["snr_db"] == snr]
        out[str(snr)] = {
            "nights": len(sub), "detection_rate": float(np.nanmean([m["detection_rate"] for m in sub])),
            "confirm_latency_mean_s": float(np.nanmean([m["confirm_latency_mean_s"] for m in sub])),
            "false_confirms_per_hour": float(np.mean([m["false_confirms_per_hour"] for m in sub])),
            "stop_latency_mean_s": float(np.nanmean([m["stop_latency_mean_s"] for m in sub])),
        }
    return out


def benchmark_config(cfg: dict) -> dict:
    """The approved benchmark (v5/configs/default.yaml `benchmark:`); a report binds to exactly this and the card refuses any other."""
    b = cfg["benchmark"]
    return {"n_nights": int(b["n_nights"]), "night_s": float(b["night_s"]), "snrs_db": [float(s) for s in b["snrs"]], "seed": int(cfg["seed"])}


def run_benchmark(predictors: dict, params: FsmParams, cfg: dict, seed: int = 0) -> dict:
    out = Path(cfg["paths"]["out_dir"])
    rows = M.read_manifest(out / "manifest.csv")
    _, audio, _ = M.load_cache(out)
    beds = _bench_beds(rows, cfg["paths"]["data_dir"])
    if not beds:
        raise RuntimeError("no bench MS-SNSD files in the manifest")
    snore_ids = [r["id"] for r in rows if r["split"] == "bench" and r["label"] == 1]
    distract_ids = [r["id"] for r in rows if r["split"] == "bench" and r["label"] == 0 and r["source"] != "mssnsd"]
    if not snore_ids or not distract_ids:
        raise RuntimeError("bench partition has no snore or distractor windows")
    snore = F.int16_to_float(audio[snore_ids])
    distract = F.int16_to_float(audio[distract_ids])
    bcfg = cfg["benchmark"]
    snrs, n_nights, night_s = list(bcfg["snrs"]), int(bcfg["n_nights"]), float(bcfg["night_s"])
    rirs = make_distance_rirs(1.0, n=3, seed=seed)
    rng = np.random.default_rng(seed)
    names = list(predictors)
    nights = {name: [] for name in names}
    agree_hits, agree_total, start_counts = {}, 0, {name: 0 for name in names}
    for k in range(n_nights):
        snr = snrs[k % len(snrs)]
        rir = rirs[k % len(rirs)] if k % 2 else None
        spec = NightSpec(duration_s=night_s, snr_db=float(snr), bed_dbfs=float(rng.uniform(-50, -30)))
        wave, episodes = generate_night(rng, beds, snore, distract, spec, rir)
        X = window_features(wave)
        probs = {name: np.asarray(fn(X), dtype=np.float64) for name, fn in predictors.items()}
        for name in names:
            events, actives = run_fsm(probs[name], params)
            m = score_night(events, actives, episodes, params.tick_ms / 1000.0, duration_s=night_s)
            m.update({"night": k, "snr_db": snr, "rir": rir is not None})
            nights[name].append(m)
            start_counts[name] += sum(1 for ev in events if ev["type"] == "episode_start")
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                key = f"{names[a]}~{names[b]}"
                agree_hits[key] = agree_hits.get(key, 0) + int(((probs[names[a]] >= params.tau) == (probs[names[b]] >= params.tau)).sum())
        agree_total += len(X)
    agreement = {key: {"tick_decision_agreement": hits / max(agree_total, 1), "episode_starts": dict(start_counts)} for key, hits in agree_hits.items()}
    return {"params": params.to_dict(), "nights": nights, "by_snr": {name: _aggregate(nights[name], snrs) for name in names}, "agreement": agreement}


def to_markdown(result: dict) -> str:
    lines = ["# Streaming benchmark (synthetic nights from the bench partition)", "", f"FSM params: `{json.dumps(result['params'])}`", ""]
    for name, by_snr in result["by_snr"].items():
        lines += [f"## predictor: {name}", "", "| SNR dB | nights | detection | confirm latency s | false confirms / h | stop latency s |", "|---|---|---|---|---|---|"]
        for snr, m in by_snr.items():
            lines.append(f"| {snr} | {m['nights']} | {m['detection_rate']:.3f} | {m['confirm_latency_mean_s']:.1f} | {m['false_confirms_per_hour']:.2f} | {m['stop_latency_mean_s']:.1f} |")
        lines.append("")
    for key, a in result.get("agreement", {}).items():
        lines.append(f"tick decision agreement {key}: {a['tick_decision_agreement']:.4f}; episode starts {a['episode_starts']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    import keras

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--tflite", default=None)
    ap.add_argument("--threshold", default=None)
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    out = Path(cfg["paths"]["out_dir"])
    thr = json.loads(Path(args.threshold or out / "deployed" / "threshold.json").read_text())
    params = FsmParams.from_config(thr["tau"], thr["fsm"])
    import hashlib

    from v5.train import file_sha256

    model_path = Path(args.model or out / "deployed" / "model.keras")
    tflite_path = Path(args.tflite or out / "deliverables" / "snore_v5_int8.tflite")
    model = keras.models.load_model(model_path, compile=False)
    tflite = tflite_path.read_bytes()
    predictors = {"float": lambda X: predict_probs(model, X), "int8": lambda X: int8_probs(tflite, X)}
    bench = benchmark_config(cfg)
    result = run_benchmark(predictors, params, cfg, bench["seed"])
    # bound to the files actually evaluated and to the approved benchmark configuration; the card refuses anything else
    result["bound_to"] = {"model_sha256": file_sha256(model_path), "tflite_sha256": hashlib.sha256(tflite).hexdigest(), "manifest_sha256": file_sha256(out / "manifest.csv"),
                          "tau": params.tau, "fsm": params.to_dict(), "benchmark": bench}
    (out / "deliverables").mkdir(parents=True, exist_ok=True)
    text = to_markdown(result)
    (out / "deliverables" / "benchmark_nights.md").write_text(text, encoding="utf-8")
    result["bound_to"]["report_md_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()  # the accepted markdown is the one rendered from this JSON
    (out / "benchmark_nights.json").write_text(json.dumps(result, indent=1))
    print(text)


if __name__ == "__main__":
    main()
