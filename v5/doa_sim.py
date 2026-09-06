"""Simulation sweep validating the DoA module (spec section 9)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v5 import features as F
from v5.config import load_config, resolve
from v5.doa import DoaParams, DoaTracker


def simulate_stereo(rng, snore, spacing: float, distance: float, azimuth_deg: float, snr_db: float, rt60: float = 0.3, fs: int = F.SR):
    import pyroomacoustics as pra

    dims = [4.0, 4.0, 2.6]
    e_abs, max_order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=fs, materials=pra.Material(e_abs), max_order=min(int(max_order), 6))
    cx, cy, h = 2.0, 1.0, 0.7
    mics = np.array([[cx - spacing / 2, cx + spacing / 2], [cy, cy], [h, h]])
    az = np.deg2rad(azimuth_deg)
    src = [cx + distance * np.sin(az), cy + distance * np.cos(az), 0.55]
    room.add_source(src, signal=np.asarray(snore, np.float64))
    room.add_microphone_array(mics)
    room.simulate()
    sig = room.mic_array.signals[:, : len(snore)]
    out = []
    for ch in sig:
        ch = np.asarray(ch, np.float64)
        noise = rng.standard_normal(len(ch))
        ch_rms, n_rms = np.sqrt(np.mean(ch ** 2)) + 1e-12, np.sqrt(np.mean(noise ** 2))
        out.append((ch + noise * ch_rms / (n_rms * 10 ** (snr_db / 20.0))).astype(np.float32))
    expected = "right" if azimuth_deg > 0 else "left"  # +x is towards the R microphone
    return out[0], out[1], expected


def classify(l, r, params: DoaParams) -> dict:
    tracker = DoaTracker(params)
    for k in range(0, len(l) - params.n_fft + 1, params.n_fft // 2):
        tracker.frame(l[k: k + params.n_fft], r[k: k + params.n_fft])
    return tracker.episode()


def sweep_config(cfg: dict, rows) -> dict:
    """The approved sweep (v5/configs/default.yaml `doa:`) plus the bench snore count it must have used."""
    d = cfg.get("doa", {})
    return {"trials": int(d.get("sim_trials", 50)), "seed": int(cfg["seed"]), "spacings_m": list(d.get("sim_spacings_m", [0.04, 0.06, 0.08, 0.12])),
            "distances_m": list(d.get("sim_distances_m", [0.5, 1.0, 1.5])), "snrs_db": list(d.get("sim_snrs_db", [0, 5, 10, 20])),
            "n_snore_windows": sum(1 for r in rows if r["split"] == "bench" and r["label"] == 1)}


def run_sweep(snore_windows, spacings=(0.04, 0.06, 0.08, 0.12), distances=(0.5, 1.0, 1.5), snrs=(0, 5, 10, 20), trials: int = 50, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for s in spacings:
        for d in distances:
            for snr in snrs:
                correct = unknown = 0
                for _ in range(trials):
                    w = snore_windows[int(rng.integers(0, len(snore_windows)))]
                    az = float(rng.uniform(20.0, 70.0)) * float(rng.choice([-1.0, 1.0]))
                    l, r, exp = simulate_stereo(rng, w, s, d, az, float(snr))
                    res = classify(l, r, DoaParams(spacing_m=s))
                    if res["side"] == "unknown":
                        unknown += 1
                    elif res["side"] == exp:
                        correct += 1
                rows.append({"spacing_m": s, "distance_m": d, "snr_db": snr, "trials": trials, "accuracy": correct / trials, "unknown_rate": unknown / trials})
                print(f"spacing {s} m, distance {d} m, snr {snr} dB: accuracy {correct / trials:.3f}, unknown {unknown / trials:.3f}", flush=True)
    return rows


def to_markdown(rows) -> str:
    lines = ["# DoA simulation sweep", "", "| spacing m | distance m | SNR dB | side accuracy | unknown rate |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['spacing_m']} | {r['distance_m']} | {r['snr_db']} | {r['accuracy']:.3f} | {r['unknown_rate']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    from v5.data import manifest as M

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--trials", type=int, default=None, help="override doa.sim_trials (the model card then refuses the report)")
    args = ap.parse_args(argv)
    cfg = resolve(load_config(args.config))
    out = Path(cfg["paths"]["out_dir"])
    rows = M.read_manifest(out / "manifest.csv")
    _, audio, _ = M.load_cache(out)
    sw = sweep_config(cfg, rows)
    if args.trials is not None:
        sw["trials"] = int(args.trials)
    snore = F.int16_to_float(audio[[r["id"] for r in rows if r["split"] == "bench" and r["label"] == 1]])  # never the test split
    assert len(snore) == sw["n_snore_windows"]
    result = run_sweep(snore, spacings=sw["spacings_m"], distances=sw["distances_m"], snrs=sw["snrs_db"], trials=sw["trials"], seed=sw["seed"])
    from v5.train import file_sha256

    # the card accepts this report only if every parameter equals the approved configuration for the current manifest
    # and the markdown is the one rendered from this JSON
    import hashlib

    (out / "deliverables").mkdir(parents=True, exist_ok=True)
    text = to_markdown(result)
    (out / "deliverables" / "doa_sim.md").write_text(text, encoding="utf-8")
    from v5.export import release_binding  # the DoA sweep does not use the model, but the report is evidence of one release only

    bound = {**release_binding(out), **sw, "report_md_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    (out / "doa_sim.json").write_text(json.dumps({"rows": result, "bound_to": bound}, indent=1))


if __name__ == "__main__":
    main()
