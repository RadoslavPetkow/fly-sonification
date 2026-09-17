"""Offline audio file -> external current on the sensory input groups.

    .venv/bin/python -m audio.audio_in floor [--force]   # measure i_ext_floor -> cache/calibration.json
    .venv/bin/python -m audio.audio_in info PATH         # band levels, currents, dynamic range of a file
    .venv/bin/python -m audio.audio_in groups            # check the band->group split against the
                                                         # transmission experiment's cached run

    .venv/bin/python -m audio.audio_in safe [--force]    # measure i_ext_safe -> cache/calibration.json

Pipeline per AudioConfig:
  load (mono, resampled to sr) -> causal frames: hop = block_ms, window = window_mult * hop,
  Hann -> one-sided power spectrum -> n_bins log-spaced bands over [f_min_hz, f_max_hz]
  -> band level in dBFS (0 dB = a full-scale sine entirely inside the band)
  -> minus ONE global clip reference (level_reference "clip_p95": the clip_ref_percentile of
     all band levels of the whole file, pooled over bands; "absolute": 0 dB)
  -> clip to [db_floor, db_ceil] -> x in [0, 1] -> x ** compression
  -> current in [i_ext_safe, i_ext_max] (clip_p95) or [floor_margin * i_ext_floor, i_ext_max]
     (absolute, v1), all from cache/calibration.json.

Band b drives sensory group b of data.groups.ordered_groups (sorted by type then bodyId,
AudioConfig.n_bins groups), the same split experiments/transmission.py used. Frame k's
window ends at (k + 1) * hop, so the current applied during [k * hop, (k + 1) * hop)
depends only on audio up to the end of that interval.
"""
import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.signal.windows import hann

from config import AudioConfig, CalibConfig, Paths, SimConfig, TransmissionConfig
from data.groups import load_neurons_and_indices, ordered_groups

CACHE = ROOT / Paths().cache
CALIBRATION = CACHE / "calibration.json"
CALIBRATED_KEYS = ("normalization", "w_scale", "g_inh", "b_adapt", "tau_adapt_ms", "sensory_adapt_exempt",
                   "sigma_noise", "refrac_steps")


# =============================================================================
# audio -> band levels
# =============================================================================

def load_audio(path, sr=None):
    """Mono float32 samples at `sr` (default AudioConfig.sr)."""
    sr = AudioConfig().sr if sr is None else sr
    data, file_sr = sf.read(str(path), dtype="float64", always_2d=True)
    if data.size == 0:
        raise ValueError(f"{path}: no audio samples")
    mono = data.mean(axis=1)
    if file_sr != sr:
        g = math.gcd(int(file_sr), int(sr))
        mono = resample_poly(mono, sr // g, file_sr // g)
    return mono.astype(np.float32), sr


def framing(cfg):
    hop = round(cfg.sr * cfg.block_ms / 1e3)
    win = hop * cfg.window_mult
    n_fft = 1 << math.ceil(math.log2(win * cfg.fft_pad_mult))
    return hop, win, n_fft


def band_edges(cfg):
    return np.geomspace(cfg.f_min_hz, cfg.f_max_hz, cfg.n_bins + 1)


def band_bins(cfg):
    """Per band, the FFT bin indices whose centre frequency lies in [lo, hi) (last band closed)."""
    _, _, n_fft = framing(cfg)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / cfg.sr)
    edges = band_edges(cfg)
    bins = []
    for b in range(cfg.n_bins):
        hi_ok = freqs <= edges[b + 1] if b == cfg.n_bins - 1 else freqs < edges[b + 1]
        idx = np.flatnonzero((freqs >= edges[b]) & hi_ok)
        if idx.size == 0:
            raise ValueError(f"band {b} [{edges[b]:.1f}, {edges[b + 1]:.1f}) Hz contains no FFT bin")
        bins.append(idx)
    return bins


def band_levels(y, cfg):
    """(times_s, levels_db): levels (n_frames, n_bins) in dBFS for the causal frames of y."""
    hop, win, n_fft = framing(cfg)
    w = hann(win, sym=False)
    n_frames = math.ceil(y.size / hop)
    padded = np.concatenate([np.zeros(win - hop, dtype=np.float64), y.astype(np.float64),
                             np.zeros(n_frames * hop - y.size, dtype=np.float64)])
    starts = np.arange(n_frames) * hop
    frames = np.lib.stride_tricks.sliding_window_view(padded, win)[starts]
    X = np.fft.rfft(frames * w, n=n_fft, axis=1)
    # one-sided power per bin, normalized so the bins of a sine of amplitude A sum to A^2 / 2
    p = 2.0 * np.abs(X) ** 2 / (n_fft * np.sum(w ** 2))
    band_power = np.stack([p[:, idx].sum(axis=1) for idx in band_bins(cfg)], axis=1)
    levels = 10.0 * np.log10(np.maximum(band_power, 1e-20) / 0.5)
    return starts / cfg.sr, levels


# =============================================================================
# levels -> current
# =============================================================================

def current_range(cfg, calibration=None):
    cal = json.loads(CALIBRATION.read_text()) if calibration is None else calibration
    if cfg.level_reference in ("clip_range", "clip_p95"):
        if "i_ext_safe" not in cal:
            raise KeyError("cache/calibration.json has no i_ext_safe; run `python -m audio.audio_in safe` first")
        lo = cal["i_ext_safe"]
    elif cfg.level_reference == "absolute":
        if "i_ext_floor" not in cal:
            raise KeyError("cache/calibration.json has no i_ext_floor; run `python -m audio.audio_in floor` first")
        lo = cfg.floor_margin * cal["i_ext_floor"]
    else:
        raise ValueError(f"unknown level_reference {cfg.level_reference!r}")
    hi = cal["i_ext_max"]
    if not 0 < lo < hi:
        raise ValueError(f"invalid current range [{lo}, {hi}]")
    return lo, hi


def frame_total_db(levels_db):
    """Total band power of each frame, in dB (same reference as the band levels)."""
    return 10.0 * np.log10(np.sum(10.0 ** (levels_db / 10.0), axis=1))


def clip_reference(levels_db, cfg):
    """Global level statistics of a WHOLE clip (never per band).

    silence_threshold_db  frames whose total band power is below this are silent
    p_lo_db, p_hi_db      clip_range_percentiles of all band levels of the non-silent frames (clip_range)
    p95_all_db            clip_ref_percentile of all band levels of all frames (clip_p95, v2)"""
    tot = frame_total_db(levels_db)
    thr = float(np.percentile(tot, cfg.clip_ref_percentile) + cfg.silence_rel_db)
    nonsilent = tot >= thr
    if not nonsilent.any():
        raise ValueError("every frame of the clip is silent")
    p_lo, p_hi = np.percentile(levels_db[nonsilent], cfg.clip_range_percentiles)
    if not p_hi > p_lo:
        raise ValueError(f"degenerate clip level range [{p_lo}, {p_hi}]")
    return {"mode": cfg.level_reference, "silence_threshold_db": thr, "silent_frac": float(1 - nonsilent.mean()),
            "p_lo_db": float(p_lo), "p_hi_db": float(p_hi),
            "p95_all_db": float(np.percentile(levels_db, cfg.clip_ref_percentile))}


def level_position(levels_db, cfg, ref):
    """x in [0, 1] before compression, and whether each value was clipped below / above."""
    if cfg.level_reference == "clip_range":
        raw = (levels_db - ref["p_lo_db"]) / (ref["p_hi_db"] - ref["p_lo_db"])
    else:
        off = ref["p95_all_db"] if cfg.level_reference == "clip_p95" else 0.0
        raw = (levels_db - off - cfg.db_floor) / (cfg.db_ceil - cfg.db_floor)
    return np.clip(raw, 0.0, 1.0), raw < 0, raw > 1


def levels_to_current(levels_db, cfg, lo, hi, ref):
    x, _, _ = level_position(levels_db, cfg, ref)
    x = x ** cfg.compression
    if cfg.level_reference == "clip_range":
        return lo * (hi / lo) ** x
    if cfg.level_reference in ("clip_p95", "absolute"):
        return lo + (hi - lo) * x
    raise ValueError(f"unknown level_reference {cfg.level_reference!r}")


def sensory_rate_curve(calibration=None):
    """Measured (current, mean sensory rate) points: the i_ext_safe run at i_ext_safe (seed mean)
    followed by the calibration f-I curve above it."""
    cal = json.loads(CALIBRATION.read_text()) if calibration is None else calibration
    safe = cal["i_ext_safe"]
    row = next(r for r in cal["i_ext_safe_measurement"]["table"] if r["current"] == safe)
    pts = [(safe, float(np.mean([x["sensory_rate_hz"] for x in row["seeds"]])))]
    pts += [(p["current"], p["sensory_rate_hz"]) for p in cal["fi_curve"] if p["current"] > safe]
    return np.array(pts)


def implied_sensory_rate(current, curve):
    """Interpolated in log current; clamped to the measured range."""
    return np.interp(np.log(current), np.log(curve[:, 0]), curve[:, 1])


def dynamic_range_report(levels_db, cfg, lo, hi, ref, calibration=None):
    """Where a section's levels land in the mapping: clipping, current span in doublings, implied sensory rates."""
    x, below, above = level_position(levels_db, cfg, ref)
    cur = levels_to_current(levels_db, cfg, lo, hi, ref)
    curve = sensory_rate_curve(calibration)

    def span(c):
        c5, c95 = np.percentile(c, [5, 95])
        r5, r95 = implied_sensory_rate(np.array([c5, c95]), curve)
        return {"current_p5": float(c5), "current_p95": float(c95), "doublings": float(np.log2(c95 / c5)),
                "sensory_hz_p5": float(r5), "sensory_hz_p95": float(r95), "sensory_span_hz": float(r95 - r5)}

    rep = {"level_reference": cfg.level_reference, "reference": ref, "current_window": [lo, hi],
           "window_doublings": float(np.log2(hi / lo)), "compression": cfg.compression,
           "all_bands": {"frac_clipped_low": float(below.mean()), "frac_clipped_high": float(above.mean()), **span(cur)},
           "per_band": []}
    for b in range(levels_db.shape[1]):
        rep["per_band"].append({"band": b, "frac_clipped_low": float(below[:, b].mean()),
                                "frac_clipped_high": float(above[:, b].mean()),
                                "level_p5_db": float(np.percentile(levels_db[:, b], 5)),
                                "level_p95_db": float(np.percentile(levels_db[:, b], 95)), **span(cur[:, b])})
    return rep


# =============================================================================
# groups and frames
# =============================================================================

class AudioDrive:
    """Maps an audio file to per-frame external current vectors for the network."""

    def __init__(self, cfg=None, cache=CACHE, calibration=None):
        self.cfg = AudioConfig() if cfg is None else cfg
        neurons, indices = load_neurons_and_indices(cache)
        self.N = len(neurons)
        self.sensory_idx = np.array(indices["sensory_idx"])
        self.groups = ordered_groups(neurons, self.sensory_idx, self.cfg.n_bins)
        pos = {int(n): k for k, n in enumerate(self.sensory_idx)}
        self.group_of_sensory = np.empty(self.sensory_idx.size, dtype=np.int64)
        for g, members in enumerate(self.groups):
            for n in members:
                self.group_of_sensory[pos[int(n)]] = g
        self.lo, self.hi = current_range(self.cfg, calibration)
        self._sens_t = torch.from_numpy(self.sensory_idx)

    def band_currents(self, path, offset_s=0.0, duration_s=None):
        """(times, levels_dbfs, currents, ref) for the section; ref (clip_reference) comes from the whole file."""
        y, sr = load_audio(path, self.cfg.sr)
        start = round(offset_s * sr)
        stop = y.size if duration_s is None else min(y.size, start + round(duration_s * sr))
        if start >= stop:
            raise ValueError(f"{path}: offset {offset_s} s is beyond the end ({y.size / sr:.2f} s)")
        ref = clip_reference(band_levels(y, self.cfg)[1], self.cfg)
        times, levels = band_levels(y[start:stop], self.cfg)
        return times, levels, levels_to_current(levels, self.cfg, self.lo, self.hi, ref), ref

    def frames(self, path, offset_s=0.0, duration_s=None):
        """Yield (t_sec, I_ext): I_ext is a float32 tensor of length N, zero outside the sensory indices."""
        times, _, currents, _ = self.band_currents(path, offset_s, duration_s)
        per_sensory = torch.from_numpy(currents[:, self.group_of_sensory].astype(np.float32))
        for k, t in enumerate(times):
            I = torch.zeros(self.N, dtype=torch.float32)
            I[self._sens_t] = per_sensory[k]
            yield float(t), I


def frames(path, offset_s=0.0, duration_s=None):
    return AudioDrive().frames(path, offset_s, duration_s)


def verify_groups_against_transmission():
    """The band->group split must be the one the transmission run drove.

    cache/transmission_results.json does not store group membership, so the check uses the cached
    simulation itself: every active sensory neuron's binned count must correlate best with the drive
    of the group data.groups assigns it to."""
    from experiments.transmission import SIM_CACHE, abs_corr

    tc, cfg = TransmissionConfig(), AudioConfig()
    results = json.loads((CACHE / "transmission_results.json").read_text())
    n_groups = results["layers"]["input groups"]["neurons"]
    if n_groups != cfg.n_bins:
        raise AssertionError(f"transmission used {n_groups} input groups, AudioConfig.n_bins is {cfg.n_bins}")
    d = np.load(SIM_CACHE)
    counts, drive = d["counts"], d["drive_binned"]
    neurons, indices = load_neurons_and_indices()
    sens = np.array(indices["sensory_idx"])
    groups = ordered_groups(neurons, sens, cfg.n_bins)
    assigned = np.empty(neurons.shape[0], dtype=np.int64)
    for g, members in enumerate(groups):
        assigned[members] = g
    active = sens[counts[:, sens].sum(axis=0) >= tc.min_spikes]
    r = abs_corr(counts[:, active].T.astype(np.float64), drive.T.astype(np.float64))
    best = r.argmax(axis=1)
    match = best == assigned[active]
    print(f"groups: sizes {[len(g) for g in groups]}; {int(match.sum())}/{active.size} active sensory neurons "
          f"correlate best with their own group's drive in the cached transmission run "
          f"(median |r| own {np.median(r[np.arange(active.size), assigned[active]]):.3f})")
    if not match.all():
        raise AssertionError(f"group split does not match the transmission run for bodyIds "
                             f"{neurons.iloc[active[~match]]['bodyId'].tolist()}")
    return groups


# =============================================================================
# ignition floor
# =============================================================================

def measure_ignition_floor(force=False):
    from sim.calibrate import Runner, header, rheobase

    cfg, cc = AudioConfig(), CalibConfig()
    cal = json.loads(CALIBRATION.read_text())
    if "i_ext_floor" in cal and not force:
        print(f"i_ext_floor already in calibration.json: {cal['i_ext_floor']} (--force to re-measure)")
        return cal["i_ext_floor"]
    sc = replace(SimConfig(), **{k: cal[k] for k in CALIBRATED_KEYS})
    neurons, indices = load_neurons_and_indices()
    n_sens = len(indices["sensory_idx"])
    runner = Runner(sc, np.ones(len(neurons), dtype=bool))
    header(f"IGNITION FLOOR - constant current into all {n_sens} sensory neurons, from rest, "
           f"{cc.warmup_ms:g} ms warm-up + {cfg.ignition_ms:g} ms, seeds {cfg.ignition_seeds}")
    print(f"calibrated: " + ", ".join(f"{k}={cal[k]}" for k in CALIBRATED_KEYS) + f"; rheobase {rheobase(sc):g}")
    table = []
    for current in cfg.ignition_currents:
        row = {"current": float(current), "seeds": []}
        for seed in cfg.ignition_seeds:
            m, _ = runner.measure(cal["normalization"], cal["g_inh"], cal["b_adapt"], cal["w_scale"], cal["sigma_noise"],
                                  current, cc.warmup_ms, cfg.ignition_ms, seed=seed, exempt=cal["sensory_adapt_exempt"])
            ns = (m["rate_hz"] * len(neurons) - m["sensory_rate_hz"] * n_sens) / (len(neurons) - n_sens)
            row["seeds"].append({"seed": seed, "nonsensory_rate_hz": ns, "sensory_rate_hz": m["sensory_rate_hz"],
                                 "motor_rate_hz": m["motor_rate_hz"]})
        table.append(row)
        print(f"  I {current:<5g}: non-sensory " + "  ".join(f"{s['nonsensory_rate_hz']:>7.4g}" for s in row["seeds"])
              + " Hz | sensory " + "  ".join(f"{s['sensory_rate_hz']:>6.3g}" for s in row["seeds"])
              + " Hz | motor " + "  ".join(f"{s['motor_rate_hz']:>6.3g}" for s in row["seeds"]) + " Hz", flush=True)
    if table[0]["current"] != 0.0:
        raise ValueError("ignition_currents must start at 0 (the no-input level)")
    no_input = max(s["nonsensory_rate_hz"] for s in table[0]["seeds"])
    threshold = no_input + cfg.ignition_rate_hz
    ignites = [all(s["nonsensory_rate_hz"] > threshold for s in row["seeds"]) for row in table]
    for row, ig in zip(table, ignites):
        row["ignites"] = ig
    k = next((i for i in range(len(table)) if all(ignites[i:])), None)
    if k is None or not ignites[-1]:
        raise RuntimeError(f"no tested current ignites the network on every seed (threshold {threshold:.3g} Hz)")
    floor = table[k]["current"]
    print(f"no-input non-sensory rate (max over seeds) {no_input:.4g} Hz; ignition = every seed > {threshold:.4g} Hz")
    print(f"ignites: " + ", ".join(f"{r['current']:g}:{'Y' if r['ignites'] else 'n'}" for r in table))
    print(f"i_ext_floor = {floor:g} (= {floor / rheobase(sc):g} x rheobase; resolution: next lower tested current "
          f"{table[k - 1]['current'] if k > 0 else 'n/a'})")
    cal["i_ext_floor"] = floor
    cal["i_ext_floor_measurement"] = {
        "definition": (f"smallest tested constant current into all sensory neurons at which the non-sensory population "
                       f"rate exceeds the no-input rate by {cfg.ignition_rate_hz:g} Hz on every seed, with every "
                       f"larger tested current also igniting; from rest, {cc.warmup_ms:g} ms warm-up, "
                       f"{cfg.ignition_ms:g} ms measured"),
        "no_input_rate_hz": no_input, "table": table, "audio_config": asdict(cfg),
    }
    CALIBRATION.write_text(json.dumps(cal, indent=2, default=str))
    print(f"wrote i_ext_floor to {CALIBRATION.relative_to(ROOT)}")
    return floor


def measure_safe_current(force=False):
    from sim.calibrate import Runner, header

    cfg, cc = AudioConfig(), CalibConfig()
    cal = json.loads(CALIBRATION.read_text())
    if "i_ext_safe" in cal and not force:
        print(f"i_ext_safe already in calibration.json: {cal['i_ext_safe']} (--force to re-measure)")
        return cal["i_ext_safe"]
    if "i_ext_floor" not in cal:
        raise KeyError("measure i_ext_floor first (`python -m audio.audio_in floor`)")
    floor = cal["i_ext_floor"]
    currents = [c for c in cfg.safe_currents if c >= floor]
    if not currents:
        raise ValueError(f"no safe_currents at or above i_ext_floor {floor}")
    ignition_threshold = cal["i_ext_floor_measurement"]["no_input_rate_hz"] + cfg.ignition_rate_hz
    sc = replace(SimConfig(), **{k: cal[k] for k in CALIBRATED_KEYS})
    neurons, indices = load_neurons_and_indices()
    sens = np.array(indices["sensory_idx"])
    N = len(neurons)
    runner = Runner(sc, np.ones(N, dtype=bool))
    header(f"SAFE CURRENT - sensory >= {cfg.safe_min_sensory_hz:g} Hz with the network running; currents from "
           f"i_ext_floor {floor:g} upward, from rest, {cc.warmup_ms:g} ms warm-up + {cfg.safe_ms:g} ms, seeds {cfg.safe_seeds}")
    table = []
    for current in currents:
        row = {"current": float(current), "seeds": []}
        for seed in cfg.safe_seeds:
            m, x = runner.measure(cal["normalization"], cal["g_inh"], cal["b_adapt"], cal["w_scale"], cal["sigma_noise"],
                                  current, cc.warmup_ms, cfg.safe_ms, seed=seed, exempt=cal["sensory_adapt_exempt"])
            ns = (m["rate_hz"] * N - m["sensory_rate_hz"] * sens.size) / (N - sens.size)
            row["seeds"].append({"seed": seed, "sensory_rate_hz": m["sensory_rate_hz"], "nonsensory_rate_hz": ns,
                                 "frac_sensory_neurons_ge_min": float((x["rates"][sens] >= cfg.safe_min_sensory_hz).mean()),
                                 "running": ns > ignition_threshold})
        row["passes"] = all(s["running"] and s["sensory_rate_hz"] >= cfg.safe_min_sensory_hz for s in row["seeds"])
        table.append(row)
        print(f"  I {current:<5g}: sensory " + "  ".join(f"{s['sensory_rate_hz']:>6.3g}" for s in row["seeds"])
              + " Hz (neurons >= min: " + " ".join(f"{s['frac_sensory_neurons_ge_min']:.0%}" for s in row["seeds"])
              + ") | non-sensory " + "  ".join(f"{s['nonsensory_rate_hz']:>5.3g}" for s in row["seeds"])
              + f" Hz | {'PASS' if row['passes'] else 'fail'}", flush=True)
    k = next((i for i in range(len(table)) if all(r["passes"] for r in table[i:])), None)
    if k is None:
        raise RuntimeError("no tested current keeps the sensory set >= the minimum rate on every seed")
    safe = table[k]["current"]
    print(f"i_ext_safe = {safe:g} (next lower tested current {table[k - 1]['current'] if k > 0 else 'n/a'}); "
          f"drive window [{safe:g}, {cal['i_ext_max']:.4g}] = x{cal['i_ext_max'] / safe:.3g} "
          f"(was [{cfg.floor_margin * floor:.4g}, {cal['i_ext_max']:.4g}] = x{cal['i_ext_max'] / (cfg.floor_margin * floor):.3g})")
    cal["i_ext_safe"] = safe
    cal["i_ext_safe_measurement"] = {
        "definition": (f"lowest tested constant current >= i_ext_floor at which, on every seed, the non-sensory rate "
                       f"exceeds the ignition threshold ({ignition_threshold:g} Hz) and the sensory set's mean rate is "
                       f">= {cfg.safe_min_sensory_hz:g} Hz, with every higher tested current passing; from rest, "
                       f"{cc.warmup_ms:g} ms warm-up, {cfg.safe_ms:g} ms measured"),
        "table": table,
    }
    CALIBRATION.write_text(json.dumps(cal, indent=2, default=str))
    print(f"wrote i_ext_safe to {CALIBRATION.relative_to(ROOT)}")
    return safe


# =============================================================================
# CLI
# =============================================================================

def info(path):
    drive = AudioDrive()
    times, levels, currents, ref = drive.band_currents(path)
    cfg = drive.cfg
    hop, win, n_fft = framing(cfg)
    edges = band_edges(cfg)
    print(f"{path}: {times.size} frames ({times.size * cfg.block_ms / 1e3:.2f} s), hop {hop}, window {win}, "
          f"n_fft {n_fft}, sr {cfg.sr}")
    print("bands (Hz): " + ", ".join(f"{edges[b]:.0f}-{edges[b + 1]:.0f} ({len(i)} bins)"
                                     for b, i in enumerate(band_bins(cfg))))
    rep = dynamic_range_report(levels, cfg, drive.lo, drive.hi, ref)
    a = rep["all_bands"]
    print(f"mapping {cfg.level_reference}: reference {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in ref.items()})}; "
          f"current [{drive.lo:.4g}, {drive.hi:.4g}] = {rep['window_doublings']:.2f} doublings")
    print(f"  all bands: current p5-p95 {a['current_p5']:.3g}-{a['current_p95']:.3g} = {a['doublings']:.2f} doublings, "
          f"implied sensory {a['sensory_hz_p5']:.1f}-{a['sensory_hz_p95']:.1f} Hz; clipped low {a['frac_clipped_low']:.1%}, "
          f"high {a['frac_clipped_high']:.1%}")
    for b in rep["per_band"]:
        print(f"  band {b['band']}: current p5-p95 {b['current_p5']:.3g}-{b['current_p95']:.3g} ({b['doublings']:.2f} doublings, "
              f"sensory {b['sensory_hz_p5']:.0f}-{b['sensory_hz_p95']:.0f} Hz); clipped low {b['frac_clipped_low']:.1%}, "
              f"high {b['frac_clipped_high']:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("floor")
    f.add_argument("--force", action="store_true")
    i = sub.add_parser("info")
    i.add_argument("path")
    sub.add_parser("groups")
    sf_ = sub.add_parser("safe")
    sf_.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "floor":
        measure_ignition_floor(force=args.force)
    elif args.cmd == "safe":
        measure_safe_current(force=args.force)
    elif args.cmd == "info":
        info(args.path)
    else:
        verify_groups_against_transmission()
