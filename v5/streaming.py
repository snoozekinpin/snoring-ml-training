"""Deterministic episode state machine with integer tick arithmetic (spec section 8).

Mirrored line for line by esp32_firmware/v5/snore_episode_fsm.c.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

IDLE, ACTIVE, CONFIRMED = "IDLE", "ACTIVE", "CONFIRMED"


@dataclass(frozen=True)
class FsmParams:
    tau: float = 0.65
    tick_ms: int = 500
    hold_ticks: int = 12
    confirm_ticks: int = 20
    verify_ticks: int = 30
    min_bursts: int = 3
    period_min_ticks: int = 3
    period_max_ticks: int = 14

    @classmethod
    def from_config(cls, tau: float, fsm: dict) -> "FsmParams":
        tick = int(fsm.get("tick_ms", 500))

        def ticks(seconds) -> int:
            return int(round(float(seconds) * 1000.0 / tick))

        return cls(tau=float(tau), tick_ms=tick, hold_ticks=ticks(fsm.get("hold_s", 6)), confirm_ticks=ticks(fsm.get("confirm_s", 10)),
                   verify_ticks=ticks(fsm.get("verify_s", 15)), min_bursts=int(fsm.get("min_bursts", 3)),
                   period_min_ticks=ticks(fsm.get("period_min_s", 1.5)), period_max_ticks=ticks(fsm.get("period_max_s", 7.0)))

    def to_dict(self) -> dict:
        return asdict(self)


class EpisodeFsm:
    def __init__(self, params: FsmParams):
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.tick_i = 0
        self.state = IDLE
        self.p_hist = deque(maxlen=self.p.hold_ticks + 1)
        self.streak_q = 0
        self.in_burst = False
        self.burst_starts = deque()
        self.hits = deque()  # (tick, p, level) for hits inside the retained window
        self.active = False
        self.activity = 0.0
        self.below_ticks = 0
        self.ep = None
        self.last_episode = None
        self.episode_count = 0

    def _periodic(self) -> bool:
        starts = list(self.burst_starts)
        if len(starts) < self.p.min_bursts:
            return False
        gaps = sorted(b - a for a, b in zip(starts, starts[1:]))
        med = gaps[(len(gaps) - 1) // 2]  # lower median, same rule in C
        return self.p.period_min_ticks <= med <= self.p.period_max_ticks

    def tick(self, p: float, level_dbfs: float = 0.0) -> dict | None:
        P = self.p
        i = self.tick_i
        self.tick_i += 1
        self.p_hist.append(float(p))
        self.activity = max(self.p_hist)
        hit = p >= P.tau
        self.active = self.activity >= P.tau
        burst_started = hit and not self.in_burst
        if burst_started:
            self.burst_starts.append(i)
        if hit:
            self.hits.append((i, float(p), float(level_dbfs)))
        self.in_burst = hit
        max_age = P.confirm_ticks + P.hold_ticks
        while self.burst_starts and i - self.burst_starts[0] > max_age:
            self.burst_starts.popleft()
        while self.hits and i - self.hits[0][0] > max_age:
            self.hits.popleft()
        if self.active:
            self.streak_q = min(self.streak_q + 2, 4 * P.confirm_ticks)
        else:
            self.streak_q = max(0, self.streak_q - 1)
        event = None
        if self.state == IDLE and self.active:
            self.state = ACTIVE
        if self.state == ACTIVE:
            if self.streak_q == 0:
                self.state = IDLE
            elif self.streak_q >= 2 * P.confirm_ticks and self._periodic():
                first = int(self.burst_starts[0])
                sel = [h for h in self.hits if h[0] >= first]
                self.state = CONFIRMED
                self.below_ticks = 0
                self.episode_count += 1
                self.ep = {"first_burst_tick": first, "last_hit_tick": sel[-1][0] if sel else i, "n_bursts": len(self.burst_starts),
                           "sum_p": sum(h[1] for h in sel), "n_hits": len(sel), "sum_level": sum(h[2] for h in sel)}
                event = {"type": "episode_start", "episode_id": self.episode_count, "tick": i, "t": i * P.tick_ms / 1000.0, "n_bursts": self.ep["n_bursts"]}
        elif self.state == CONFIRMED:
            if hit:
                self.ep["last_hit_tick"] = i
                self.ep["sum_p"] += float(p)
                self.ep["n_hits"] += 1
                self.ep["sum_level"] += float(level_dbfs)
                if burst_started:
                    self.ep["n_bursts"] += 1
            if not self.active:
                self.below_ticks += 1
                if self.below_ticks >= P.verify_ticks:
                    ep, n = self.ep, max(self.ep["n_hits"], 1)
                    duration_ticks = ep["last_hit_tick"] - ep["first_burst_tick"] + 2  # one window = 2 ticks
                    event = {"type": "episode_end", "episode_id": self.episode_count, "tick": i, "t": i * P.tick_ms / 1000.0, "start_t": ep["first_burst_tick"] * P.tick_ms / 1000.0,
                             "duration_s": duration_ticks * P.tick_ms / 1000.0, "mean_p": ep["sum_p"] / n, "n_bursts": ep["n_bursts"],
                             "n_hits": ep["n_hits"], "level_dbfs": ep["sum_level"] / n}
                    self.last_episode = event
                    self.state = IDLE
                    self.streak_q = 0
                    self.burst_starts.clear()
                    self.hits.clear()
                    self.in_burst = False
                    self.ep = None
            else:
                self.below_ticks = 0
        return event


def run_sequence(p_seq, params: FsmParams, levels=None):
    fsm = EpisodeFsm(params)
    events, states, actives = [], [], []
    for k, p in enumerate(p_seq):
        ev = fsm.tick(float(p), 0.0 if levels is None else float(levels[k]))
        if ev is not None:
            events.append(ev)
        states.append(fsm.state)
        actives.append(fsm.active)
    return events, states, actives
