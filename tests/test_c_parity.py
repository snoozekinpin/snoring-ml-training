import shutil
import subprocess

import numpy as np
import pytest

from v5 import export as X
from v5 import features as F
from v5 import golden as G
from v5.config import ROOT
from v5.doa import DoaParams
from v5.streaming import FsmParams

FW = ROOT / "esp32_firmware" / "v5"
SCALE, ZP = 1.0 / 127.0, 0


def _synthetic_golden(golden_dir):
    rng = np.random.default_rng(7)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * np.arange(F.WIN) / F.SR)
    snore = [F.float_to_int16(0.1 * rng.standard_normal(F.WIN) * env) for _ in range(4)]
    noise = [F.float_to_int16(0.02 * rng.standard_normal(F.WIN)) for _ in range(4)]
    names, x, Xg, q = G.make_golden(snore, noise, scale=SCALE, zero_point=ZP)
    G.write_golden_bin(golden_dir / "features.bin", x, Xg, q, SCALE, ZP)


def _run(target, golden):
    build = subprocess.run(["make", "-C", str(FW), target, f"GOLDEN={golden}"], capture_output=True, text=True)
    print(build.stdout[-4000:], build.stderr[-4000:])
    assert build.returncode == 0, f"{target} failed; see output above"


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_features_match_python_reference(tmp_path):
    golden = tmp_path / "golden"
    X.write_headers_only(FW / "generated", golden, FsmParams(), DoaParams())
    _synthetic_golden(golden)
    _run("test-features", golden)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_fsm_and_doa_match_python(tmp_path):
    golden = tmp_path / "golden"
    X.write_headers_only(FW / "generated", golden, FsmParams(), DoaParams())
    _synthetic_golden(golden)
    _run("test", golden)
