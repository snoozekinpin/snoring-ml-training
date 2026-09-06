from pathlib import Path
from v5.config import ROOT, load_config, resolve


def test_default_config_loads_and_resolves():
    cfg = resolve(load_config())
    assert cfg["seed"] == 42
    assert cfg["fsm"]["confirm_s"] == 10
    assert Path(cfg["paths"]["out_dir"]).is_absolute()
    assert Path(cfg["paths"]["data_dir"]) == ROOT / "dataset"
    assert cfg["model_version"] == "cnn_v5_int8"
    assert cfg["data"]["min_counts"]["train_pos"] == 400 and cfg["threshold"]["min_calib_neg"] == 300 and cfg["data"]["kaggle_train_frac"] == 0.5
    assert cfg["data"]["whl_val_batches"] == ["000002", "100002"]


def test_model_version_rule():
    import pytest

    from v5.versioning import MODEL_VERSION, check_model_version

    assert check_model_version(MODEL_VERSION) == "cnn_v5_int8"
    for bad in ("cnn_demo", "simulator_v5", "MOCK"):
        with pytest.raises(ValueError):
            check_model_version(bad)
