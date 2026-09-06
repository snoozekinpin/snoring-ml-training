import pytest

from v5.config import ROOT, load_config
from v5.data import manifest as M

OUT = ROOT / "output" / "v5"


@pytest.mark.dataset
def test_real_manifest_release_gate():
    path = OUT / "manifest.csv"
    if not path.exists():
        pytest.fail("dataset is present but output/v5/manifest.csv is missing: run python -m v5.data.manifest")
    rows = M.read_manifest(path)
    M.check_invariants(rows)
    M.check_min_counts(rows, load_config()["data"]["min_counts"])
    kaggle = [r for r in rows if r["source"].startswith("kaggle")]
    assert all(r["split"] in ("test", "train", "drop") for r in kaggle)
    prov = {e["id"]: e for e in M.load_drop_provenance(OUT)}
    assert all(r["id"] in prov and prov[r["id"]]["reason"] in ("exact_dup", "exact_conflict", "conflict", "unverified_positive", "snore_in_negative") for r in kaggle if r["split"] == "drop")
    assert not any(r["category"] == "snoring" and r["label"] == 0 for r in rows)
    assert {r["category"] for r in rows if r["source"] == "whl_s" and r["split"] == "val"} == {"000002"}
