"""Simulate an audio section through the calibrated network once, and cache the recording.

    .venv/bin/python -m experiments.audio_run CLIP [--offset auto|S] [--mapping clip_p95|absolute] [--force]

cache/audio_runs/<stem>_<mapping>_<offset>s.npz:
  counts        (ticks, N) uint8   spikes per tick for every neuron (tick = TransmissionConfig.bin_ms
                                   = MidiConfig.tick_ms)
  signals       (ticks, n_bins)    mean band drive current per tick
  levels        (ticks, n_bins)    mean band level (dBFS) per tick
  spike_steps   int32              every spike of the voice motor neurons (hop MidiConfig.voice_hop):
  spike_idx     int32              step since section start (ms at dt 1) and matrix index
  meta          json
Test (c) and the MIDI renderer read this, so a section is simulated once per mapping.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from audio.audio_in import CALIBRATED_KEYS, AudioDrive
from config import AudioConfig, CalibConfig, MidiConfig, Paths, SimConfig, TransmissionConfig
from data.groups import hop_motor, load_neurons_and_indices, sensory_hop_distances
from sim.lif_network import LIFNetwork

CACHE = ROOT / Paths().cache
RUNS = CACHE / "audio_runs"


def tick_frames(cfg, tc, sc):
    per_tick = tc.bin_ms / cfg.block_ms
    per_frame = cfg.block_ms / sc.dt_ms
    if per_tick != int(per_tick) or per_frame != int(per_frame) or tc.bin_ms != MidiConfig().tick_ms:
        raise ValueError("need bin_ms == MidiConfig.tick_ms, a multiple of block_ms, itself a multiple of dt_ms")
    return int(per_tick), int(per_frame)


def section_candidates(clip, level_reference=None):
    """Score every candidate section of a clip under the section rule (AudioConfig section_*), audio only.

    Candidates: test_clip_seconds windows in 1 s steps, starting >= section_margin_s after the clip start
    and ending >= section_margin_s before its end. Eligible: silent frames <= section_max_silent_frac, and
    for every band the two-sample KS statistic between the section's and the whole clip's non-silent band
    levels <= section_max_ks. Score: the std of the tick-binned drive current of the least-varying band."""
    from scipy.stats import ks_2samp

    from audio.audio_in import band_levels, clip_reference, frame_total_db, levels_to_current, load_audio

    base = AudioConfig()
    cfg = base if level_reference is None else replace(base, level_reference=level_reference)
    tc = TransmissionConfig()
    drive = AudioDrive(cfg)
    y, _ = load_audio(clip, cfg.sr)
    _, levels = band_levels(y, cfg)
    ref = clip_reference(levels, cfg)
    cur = levels_to_current(levels, cfg, drive.lo, drive.hi, ref)
    silent = frame_total_db(levels) < ref["silence_threshold_db"]
    frames_per_s = round(1e3 / cfg.block_ms)
    per_tick = int(tc.bin_ms / cfg.block_ms)
    win = round(cfg.test_clip_seconds * frames_per_s)
    margin = round(cfg.section_margin_s * frames_per_s)
    starts = list(range(margin, levels.shape[0] - margin - win + 1, frames_per_s))
    if not starts:
        raise ValueError(f"{clip}: no {cfg.test_clip_seconds:g} s window fits with {cfg.section_margin_s:g} s margins")
    whole = levels[~silent]
    rows = []
    for s0 in starts:
        sl = slice(s0, s0 + win)
        sec_silent = silent[sl]
        sec = levels[sl][~sec_silent]
        ks = max(ks_2samp(sec[:, b], whole[:, b]).statistic for b in range(cfg.n_bins)) if sec.size else 1.0
        n_ticks = win // per_tick
        binned = cur[sl][:n_ticks * per_tick].reshape(n_ticks, per_tick, cfg.n_bins).mean(axis=1)
        rows.append({"offset_s": s0 / frames_per_s, "silent_frac": float(sec_silent.mean()), "max_band_ks": float(ks),
                     "score": float(binned.std(axis=0).min()),
                     "eligible": bool(sec_silent.mean() <= cfg.section_max_silent_frac and ks <= cfg.section_max_ks)})
    return rows, cfg, ref


def choose_offset(clip, level_reference=None, verbose=True):
    """The eligible section with the highest score (decided from the audio alone, before simulating)."""
    rows, cfg, ref = section_candidates(clip, level_reference)
    eligible = [r for r in rows if r["eligible"]]
    if verbose:
        print(f"section rule ({cfg.level_reference}) for {Path(clip).name}: {len(rows)} candidate windows "
              f"(margin {cfg.section_margin_s:g} s); {sum(r['silent_frac'] <= cfg.section_max_silent_frac for r in rows)} pass "
              f"silence <= {cfg.section_max_silent_frac:.0%}, {sum(r['max_band_ks'] <= cfg.section_max_ks for r in rows)} pass "
              f"max band KS <= {cfg.section_max_ks:g}, {len(eligible)} eligible; clip silent frames {ref['silent_frac']:.1%}")
    if not eligible:
        best_ks = min(rows, key=lambda r: r["max_band_ks"])
        raise RuntimeError(f"{clip}: no eligible section (lowest max band KS {best_ks['max_band_ks']:.3f} at {best_ks['offset_s']:g} s)")
    best = max(eligible, key=lambda r: r["score"])
    if verbose:
        print(f"  chosen {best['offset_s']:g}-{best['offset_s'] + cfg.test_clip_seconds:g} s: silent {best['silent_frac']:.1%}, "
              f"max band KS {best['max_band_ks']:.3f}, min band std {best['score']:.3g}")
    return best["offset_s"]


def choose_sections(clip, k, max_overlap_frac, level_reference=None, verbose=True):
    """Up to k replicate sections: eligible sections by descending score, first without any overlap with the
    sections already chosen; if that yields fewer than k, then allowing overlap <= max_overlap_frac of a section.
    The first section is always choose_offset's."""
    rows, cfg, _ = section_candidates(clip, level_reference)
    dur = cfg.test_clip_seconds
    ranked = sorted((r for r in rows if r["eligible"]), key=lambda r: -r["score"])
    if not ranked:
        raise RuntimeError(f"{clip}: no eligible section")
    chosen = []
    for limit in (0.0, max_overlap_frac):
        for r in ranked:
            if len(chosen) >= k or r in chosen:
                continue
            if all(max(0.0, dur - abs(r["offset_s"] - c["offset_s"])) / dur <= limit for c in chosen):
                chosen.append(r)
    if verbose:
        print(f"replicate sections for {Path(clip).name}: {len(chosen)} of {k} requested from {len(ranked)} eligible: "
              + ", ".join(f"{c['offset_s']:g}-{c['offset_s'] + dur:g} s (score {c['score']:.3g})" for c in chosen))
    return [c["offset_s"] for c in chosen]


def run_path(clip, offset, level_reference, tag=None):
    return RUNS / f"{Path(clip).stem}_{level_reference}_{offset:g}s{'' if tag is None else '_' + tag}.npz"


@torch.no_grad()
def run_clip(clip, offset, level_reference=None, force=False, adjacency_path=None, tag=None):
    """adjacency_path/tag: simulate on another matrix (e.g. a shuffle); tag names the cached run."""
    level_reference = AudioConfig().level_reference if level_reference is None else level_reference
    if (adjacency_path is None) != (tag is None):
        raise ValueError("give both adjacency_path and tag, or neither")
    path = run_path(clip, offset, level_reference, tag)
    if path.exists() and not force:
        d = np.load(path)
        out = {k: d[k] for k in ("counts", "signals", "levels", "spike_steps", "spike_idx")}
        out["meta"] = json.loads(str(d["meta"]))
        print(f"using cached run {path.relative_to(ROOT)}")
        return out

    cfg, tc, cc = replace(AudioConfig(), level_reference=level_reference), TransmissionConfig(), CalibConfig()
    cal = json.loads((CACHE / "calibration.json").read_text())
    sc = replace(SimConfig(), **{k: cal[k] for k in CALIBRATED_KEYS})
    per_tick, per_frame = tick_frames(cfg, tc, sc)
    drive = AudioDrive(cfg)
    times, levels, currents, ref = drive.band_currents(clip, offset, cfg.test_clip_seconds)
    n_ticks = times.size // per_tick
    n_frames = n_ticks * per_tick
    neurons, indices = load_neurons_and_indices()
    dist = sensory_hop_distances(indices)
    voices = hop_motor(indices, dist, MidiConfig().voice_hop)

    net = LIFNetwork(CACHE, cfg=sc, adjacency_path=adjacency_path)
    voices_t = torch.from_numpy(voices)
    per_sensory = torch.from_numpy(currents[:n_frames, drive.group_of_sensory].astype(np.float32))
    I = torch.zeros(net.N)
    net.reset()
    I[net.sensory_idx] = per_sensory[0]
    for _ in range(round(cc.warmup_ms / sc.dt_ms)):
        net.step(I)
    counts = np.zeros((n_ticks, net.N), dtype=np.uint8)
    acc = torch.zeros(net.N, dtype=torch.int16)
    sp_steps, sp_idx = [], []
    step = 0
    t0 = time.perf_counter()
    print(f"simulating {Path(clip).name} from {offset:g} s ({level_reference}, current "
          f"[{drive.lo:.4g}, {drive.hi:.4g}]): {n_ticks} x {tc.bin_ms:g} ms ticks", flush=True)
    for b in range(n_ticks):
        acc.zero_()
        for k in range(per_tick):
            I[net.sensory_idx] = per_sensory[b * per_tick + k]
            for _ in range(per_frame):
                spk = net.step(I)
                acc += spk
                v = spk[voices_t]
                if v.any():
                    hit = voices_t[v]
                    sp_idx.append(hit)
                    sp_steps.append(torch.full_like(hit, step))
                step += 1
        counts[b] = acc.numpy()
        if (b + 1) % 100 == 0:
            print(f"  {(b + 1) * tc.bin_ms / 1e3:>4.0f} s simulated, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    wall = time.perf_counter() - t0
    out = {
        "counts": counts,
        "signals": currents[:n_frames].reshape(n_ticks, per_tick, cfg.n_bins).mean(axis=1),
        "levels": levels[:n_frames].reshape(n_ticks, per_tick, cfg.n_bins).mean(axis=1),
        "spike_steps": (torch.cat(sp_steps).numpy() if sp_steps else np.empty(0)).astype(np.int32),
        "spike_idx": (torch.cat(sp_idx).numpy() if sp_idx else np.empty(0)).astype(np.int32),
    }
    out["meta"] = {"clip": str(clip), "offset_s": offset, "level_reference": level_reference, "reference": ref,
                   "current_range": [drive.lo, drive.hi], "n_ticks": n_ticks, "tick_ms": tc.bin_ms, "dt_ms": sc.dt_ms,
                   "seconds": n_ticks * tc.bin_ms / 1e3, "voice_idx": voices.tolist(), "tag": tag,
                   "adjacency": str(net.adjacency_path), "wall_s": wall,
                   "ms_per_step": wall * 1e3 / max(step, 1), "calibration": {k: cal[k] for k in CALIBRATED_KEYS},
                   "audio_config": asdict(cfg)}
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in out.items() if k != "meta"}, meta=json.dumps(out["meta"], default=str))
    print(f"simulated in {wall / 60:.1f} min ({out['meta']['ms_per_step']:.2f} ms/step); cached {path.relative_to(ROOT)}")
    return out


def audio_layers(run, tc, verbose=True):
    """Layers for the audio transmission test: sensory, hop 1, a hop-2 sample, motor at hop 1 and 2."""
    neurons, indices = load_neurons_and_indices()
    dist = sensory_hop_distances(indices)
    counts = run["counts"]
    total = counts.sum(axis=0)
    sens = np.asarray(indices["sensory_idx"])
    rng = np.random.default_rng(tc.seed + 1)
    layers = {}

    def add(name, idx, sample=False):
        idx = np.asarray(idx)
        active = idx[total[idx] >= tc.min_spikes]
        chosen = np.sort(rng.choice(active, tc.layer_sample, replace=False)) if sample and active.size > tc.layer_sample else active
        layers[name] = counts[:, chosen].T
        if verbose:
            print(f"  {name:<18} {idx.size:>6,} neurons, {active.size:>6,} with >= {tc.min_spikes} spikes, {chosen.size:>5} analyzed")

    add("hop 0 (sensory)", sens)
    add("hop 1", np.flatnonzero(dist == 1))
    add("hop 2", np.flatnonzero(dist == 2), sample=True)
    add("motor at hop 1", hop_motor(indices, dist, 1))
    add("motor at hop 2", hop_motor(indices, dist, 2))
    return layers


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--offset", default="auto")
    ap.add_argument("--mapping", default=AudioConfig().level_reference, choices=("clip_range", "clip_p95", "absolute"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    off = choose_offset(a.clip, a.mapping) if a.offset == "auto" else float(a.offset)
    run_clip(a.clip, off, a.mapping, a.force)
