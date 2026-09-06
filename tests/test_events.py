import json
import os
import sys
from pathlib import Path

import pytest

from v5 import events as EV
from v5.config import ROOT

EPISODE = {"type": "episode_end", "duration_s": 42.5, "mean_p": 0.87, "n_bursts": 11, "level_dbfs": -38.2}


def test_edge_event_dataclass_roundtrip():
    e = EV.EdgeEvent(ts=1, snore_p=0.9, fsm_state="CONFIRMED", side="left")
    d = e.to_dict()
    assert d["radar_presence"] is None and d["side"] == "left" and d["vib_level"] == 0


def test_cloud_payload_shape():
    p = EV.to_cloud_event("dev_1", 1700000000, EPISODE, {"side": "left", "lag_ms": 0.12, "conf": 0.8}, {"presence": True, "motion": 0.3})
    assert p["event_type"] == "snore_detected" and p["model_version"] == "cnn_v5_int8"
    assert p["snore_duration_sec"] == 42.5 and p["snore_confidence"] == 0.87 and p["in_bed"] is True and p["body_motion_level"] == 0.3
    note = json.loads(p["note"])
    assert note["side"] == "left" and note["n_bursts"] == 11 and len(p["note"]) <= EV.NOTE_MAX


def test_cloud_payload_omits_unobserved_radar_fields():
    p = EV.to_cloud_event("dev_1", 1700000000, EPISODE, {"side": "unknown"}, None)  # no radar: nothing is invented
    assert "in_bed" not in p and "body_motion_level" not in p
    p = EV.to_cloud_event("dev_1", 1700000000, EPISODE, None, {"presence": False, "motion": None})
    assert p["in_bed"] is False and "body_motion_level" not in p


def test_forbidden_version_rejected():
    for bad in ("cnn_demo", "simulator_v5", "MOCK"):
        with pytest.raises(ValueError):
            EV.to_cloud_event("d", 1, EPISODE, model_version=bad)


def test_payload_validates_against_cloud_schema():
    cloud = Path(os.environ.get("SNOOZMATE_CLOUD_DIR", ROOT.parent / "snoozmate-cloud-upload-latest"))
    if not (cloud / "app" / "models" / "schemas.py").exists():
        pytest.skip("cloud repo not present")
    sys.path.insert(0, str(cloud))
    try:
        from app.models.schemas import EventIn
    except Exception as exc:  # cloud deps missing in this venv
        pytest.skip(f"cannot import cloud schemas: {exc}")
    finally:
        sys.path.pop(0)
    payload = EV.to_cloud_event("dev_1", 1700000000, EPISODE, {"side": "right", "lag_ms": -0.1, "conf": 0.9}, {"presence": True, "motion": 0.1})
    ev = EventIn(**payload)
    assert ev.event_type == "snore_detected" and ev.model_version == "cnn_v5_int8"
