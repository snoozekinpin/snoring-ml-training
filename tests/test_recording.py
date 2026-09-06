import csv

import numpy as np
import soundfile as sf

from v5 import features as F
from v5.recording import record_session as R


def _fake_frames(seconds, channels=1, block=4096):
    rng = np.random.default_rng(0)
    total = int(seconds * F.SR)
    sent = 0
    while sent < total:
        n = min(block, total - sent)
        yield (0.01 * rng.standard_normal((n, channels))).astype(np.float32)
        sent += n


def test_chunk_writer_rolls_files(tmp_path):
    w = R.ChunkWriter(tmp_path / "s", channels=1, chunk_s=2.0)
    for fr in _fake_frames(5.0):
        w.push(fr)
    w.flush()
    assert [p.name for p in w.files] == ["chunk_0000.wav", "chunk_0001.wav", "chunk_0002.wav"]
    lens = [sf.info(p).frames for p in w.files]
    assert lens == [2 * F.SR, 2 * F.SR, F.SR]
    assert sf.info(w.files[0]).samplerate == F.SR and sf.info(w.files[0]).subtype == "PCM_16"


def test_record_with_fake_source_writes_template(tmp_path):
    files = R.record(tmp_path / "s", minutes=0.05, channels=2, frame_source=_fake_frames(10.0, channels=2), chunk_s=1.0)
    assert len(files) == 3 and sf.info(files[0]).channels == 2  # 3 s requested
    with open(tmp_path / "s" / "labels_template.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == R.TEMPLATE_COLUMNS and rows[1][3] in R.LABELS
    assert (tmp_path / "s" / "README.txt").exists()
