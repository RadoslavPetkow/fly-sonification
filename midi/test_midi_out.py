"""Tests for midi/midi_out.py on synthetic signals (test fixtures, not network data).

    .venv/bin/python -m midi.test_midi_out

Groups: the family grouping's mechanisms (4 rate groups, 3 spike groups) and the 41-group type grouping.
Patterns: ramp up/down, burst, silence, a rate sitting exactly on thresh_on, chatter around the thresholds,
constant firing, and a pattern still active at the end. For each:
  - every note_on is followed by exactly one note_off of the same note (balanced, never doubled)
  - no note is still sounding at the end of the file
  - the .mid read back has length == ticks * tick_ms
  - silence gives no notes; ramp-up and ends-active give exactly one note per group
Both scale modes are exercised. The velocity signal of spike groups is checked to be the trailing-window rate.
"""
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mido
import numpy as np

from config import MidiConfig
from midi.midi_out import group_signals, note_numbers, sonify, write_midi

T = 400         # 20 s of 50 ms ticks


def thresholds(G):
    return {"on_hz": [5.0] * G, "off_hz": [3.0] * G, "velocity_min_hz": [0.0] * G, "velocity_max_hz": [40.0] * G,
            "population_min_hz": 0.0, "population_max_hz": 20.0}


def patterns(G, rng):
    t = np.arange(T)
    ramp = np.tile(np.linspace(0, 40, T)[:, None], (1, G))
    burst = np.zeros((T, G))
    for g in range(G):
        for start in rng.choice(T - 20, size=5, replace=False):
            burst[start:start + rng.integers(1, 20), g] = rng.uniform(6, 40)
    ends = np.zeros((T, G))
    ends[T // 2:] = 30.0
    return {"ramp up": ramp, "ramp down": ramp[::-1].copy(), "burst": burst, "silence": np.zeros((T, G)),
            "at on-threshold": np.full((T, G), 5.0), "chatter": 4.0 + 2.0 * np.sign(np.sin(t / 3.0))[:, None] * np.ones((1, G)),
            "constant": np.full((T, G), 25.0), "ends active": ends}


def check(name, mechanisms, rates, cfg, tmpdir):
    G = len(mechanisms)
    spikes = np.ceil(rates / 20.0)                        # a spike count consistent with each rate
    velocity = rates
    events = sonify(mechanisms, spikes, rates, velocity, rates.mean(axis=1), thresholds(G), cfg)
    path = Path(tmpdir) / f"{name.replace(' ', '_')}_{G}_{cfg.scale_mode}.mid"
    length = write_midi(events, T, path, cfg)
    assert abs(length - T * cfg.tick_ms / 1e3) < 1e-9, f"{name}: length {length}"
    sounding, n_on, n_off, per_note = set(), 0, 0, {}
    for msg in mido.MidiFile(path).tracks[0]:
        if msg.type == "note_on" and msg.velocity > 0:
            assert msg.note not in sounding, f"{name}: double note_on {msg.note}"
            sounding.add(msg.note)
            n_on += 1
            per_note[msg.note] = per_note.get(msg.note, 0) + 1
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            assert msg.note in sounding, f"{name}: note_off without note_on {msg.note}"
            sounding.remove(msg.note)
            n_off += 1
    assert not sounding, f"{name}: stuck notes {sorted(sounding)}"
    assert n_on == n_off
    assert all(cfg.velocity_min <= e[3] <= cfg.velocity_max for e in events if e[1] == "on")
    notes = note_numbers(G, cfg)
    if name == "silence":
        assert n_on == 0
    if name in ("ramp up", "ends active", "constant"):
        assert all(per_note.get(n, 0) == 1 for n in notes), f"{name}: expected one note per group"
    if name == "at on-threshold":
        for g, m in enumerate(mechanisms):
            assert per_note.get(notes[g], 0) == (0 if m == "rate" else 1), f"{name}: group {g} ({m})"
    print(f"  {name:<16} G={G:<2} {cfg.scale_mode:<10} note_on {n_on:>5}  note_off {n_off:>5}  stuck 0  "
          f"length {length:.3f} s  PASS")


def check_velocity_window():
    cfg = MidiConfig()
    groups = {"groups": [{"matrix_index": [0], "mechanism": "spike"}, {"matrix_index": [1, 2], "mechanism": "rate"}]}
    counts = np.zeros((40, 3), dtype=np.uint8)
    counts[5:15, 0] = 2
    counts[:, 1:] = 1
    spikes, rates, vel = group_signals(counts, groups, cfg.tick_ms, cfg)
    w = round(cfg.spike_velocity_window_ms / cfg.tick_ms)
    t = 20
    expected = counts[t - w + 1:t + 1, 0].sum() / w / (cfg.tick_ms / 1e3)
    assert np.isclose(vel[t, 0], expected), (vel[t, 0], expected)
    assert np.allclose(vel[:, 1], rates[:, 1]) and np.allclose(rates[:, 1], 20.0)
    print(f"  velocity signal: spike group = trailing {cfg.spike_velocity_window_ms:g} ms rate ({vel[t, 0]:.1f} Hz at tick {t}), "
          f"rate group = its rate  PASS")


def main():
    rng = np.random.default_rng(0)
    family = ["rate", "rate", "rate", "rate", "spike", "spike", "spike"]
    typed = ["spike"] * 32 + ["rate"] * 9
    with tempfile.TemporaryDirectory() as tmp:
        for mech in (family, typed):
            for mode in ("chromatic", "pentatonic"):
                cfg = replace(MidiConfig(), scale_mode=mode)
                notes = note_numbers(len(mech), cfg)
                print(f"{len(mech)} groups, {mode}: notes {notes[0]}..{notes[-1]}")
                for name, r in patterns(len(mech), rng).items():
                    check(name, mech, r, cfg, tmp)
    check_velocity_window()
    print("all MIDI tests passed")


if __name__ == "__main__":
    main()
