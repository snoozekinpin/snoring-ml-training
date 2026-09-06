"""Self-recording kit (spec section 12): 16 kHz WAV chunks plus a labelling template."""
from __future__ import annotations

import argparse
import csv
import queue
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from v5 import features as F
from v5.config import ROOT

LABELS = ("snore", "breathing", "speech", "tv", "fan", "other")
TEMPLATE_COLUMNS = ["chunk", "start_s", "end_s", "label", "side", "posture", "distance_m", "pillow"]


def write_labels_template(session_dir) -> Path:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    p = session_dir / "labels_template.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(TEMPLATE_COLUMNS)
        wr.writerow(["chunk_0000.wav", "0.0", "10.0", "snore", "left", "supine", "0.8", "thin"])
    (session_dir / "README.txt").write_text(
        "Copy labels_template.csv to labels.csv and add one row per labelled span.\n"
        f"label: {', '.join(LABELS)}\nside: left | right | unknown\nposture: supine | lateral | prone | unknown\n"
        "start_s / end_s are seconds inside the chunk file.\n", encoding="utf-8")
    return p


class ChunkWriter:
    def __init__(self, session_dir, sr: int = F.SR, channels: int = 1, chunk_s: float = 60.0):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sr, self.channels, self.chunk_samples = sr, channels, int(chunk_s * sr)
        self.index, self.buf, self.buffered, self.files = 0, [], 0, []

    def push(self, frames) -> None:
        frames = np.asarray(frames, dtype=np.float32).reshape(-1, self.channels)
        self.buf.append(frames)
        self.buffered += len(frames)
        while self.buffered >= self.chunk_samples:
            data = np.concatenate(self.buf)
            self._write(data[: self.chunk_samples])
            rest = data[self.chunk_samples:]
            self.buf, self.buffered = ([rest] if len(rest) else []), len(rest)

    def _write(self, chunk) -> None:
        path = self.dir / f"chunk_{self.index:04d}.wav"
        sf.write(path, chunk, self.sr, subtype="PCM_16")
        self.files.append(path)
        self.index += 1

    def flush(self) -> None:
        if self.buffered:
            self._write(np.concatenate(self.buf))
            self.buf, self.buffered = [], 0


def _sounddevice_frames(device, channels: int, total: int):
    import sounddevice as sd

    q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    with sd.InputStream(samplerate=F.SR, channels=channels, device=device, dtype="float32", blocksize=4096, callback=callback):
        got = 0
        while got < total:
            frames = q.get()
            got += len(frames)
            yield frames


def record(session_dir, minutes: float, channels: int = 1, device=None, frame_source=None, chunk_s: float = 60.0) -> list[Path]:
    writer = ChunkWriter(session_dir, channels=channels, chunk_s=chunk_s)
    write_labels_template(session_dir)
    total = int(minutes * 60 * F.SR)
    got = 0
    for frames in (frame_source if frame_source is not None else _sounddevice_frames(device, channels, total)):
        remaining = total - got
        frames = np.asarray(frames)[:remaining]
        writer.push(frames)
        got += len(frames)
        if got >= total:
            break
    writer.flush()
    return writer.files


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Record a labelled self-recording session")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--channels", type=int, choices=(1, 2), default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args(argv)
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    session = Path(args.session) if args.session else ROOT / "recordings" / datetime.now().strftime("%Y%m%d_%H%M%S")
    files = record(session, args.minutes, args.channels, args.device)
    print(f"wrote {len(files)} chunks to {session}; fill in labels.csv (see README.txt)")


if __name__ == "__main__":
    main()
