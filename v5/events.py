"""Edge event schema and the cloud EventIn payload (spec section 10)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from v5.versioning import FORBIDDEN_VERSION_SUBSTRINGS, MODEL_VERSION, check_model_version  # noqa: F401  (re-exported)

NOTE_MAX = 1000


@dataclass
class EdgeEvent:
    ts: int
    snore_p: float
    fsm_state: str
    episode_id: int | None = None
    side: str = "unknown"
    lag_ms: float = 0.0
    doa_conf: float = 0.0
    level_dbfs: float = 0.0
    radar_presence: bool | None = None
    radar_motion: float | None = None
    radar_breath_rate: float | None = None
    temp_ok: bool = True
    vib_level: int = 0
    vib_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def to_cloud_event(device_id: str, ts: int, episode: dict, doa: dict | None = None, radar: dict | None = None, model_version: str = MODEL_VERSION) -> dict:
    check_model_version(model_version)
    doa, radar = doa or {}, radar or {}
    note = {"side": doa.get("side", "unknown"), "lag_ms": round(float(doa.get("lag_ms", 0.0)), 3), "doa_conf": round(float(doa.get("conf", 0.0)), 3),
            "n_bursts": int(episode.get("n_bursts", 0)), "level_dbfs": round(float(episode.get("level_dbfs", 0.0)), 1)}
    note_s = json.dumps(note, separators=(",", ":"))
    if len(note_s) > NOTE_MAX:
        raise ValueError("note exceeds the cloud limit")
    payload = {
        "device_id": device_id,
        "timestamp": int(ts),
        "event_type": "snore_detected",
        "snore_duration_sec": round(float(episode["duration_s"]), 1),
        "snore_confidence": round(min(max(float(episode["mean_p"]), 0.0), 1.0), 3),
    }
    # radar fields are sent only when the radar observed them (spec 13: in_bed = radar_presence, body_motion_level = radar_motion);
    # without a radar the keys are omitted and the cloud schema's own defaults apply, the edge never invents an observation
    if radar.get("presence") is not None:
        payload["in_bed"] = bool(radar["presence"])
    if radar.get("motion") is not None:
        payload["body_motion_level"] = float(min(max(float(radar["motion"]), 0.0), 1.0))
    payload["model_version"] = model_version
    payload["note"] = note_s
    return payload
