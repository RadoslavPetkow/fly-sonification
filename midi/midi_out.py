"""Motor activity of the hop-1 descending neurons -> offline .mid file.

    .venv/bin/python -m midi.midi_out groups                         # voice groups -> cache/motor_groups.json
    .venv/bin/python -m midi.midi_out thresholds CLIP [CLIP ...]     # from the cached test (c) runs
    .venv/bin/python -m midi.midi_out render CLIP [--offset auto|S] [--scale chromatic|pentatonic] [--name NAME]

VOICES: the motor neurons at BFS hop MidiConfig.voice_hop from the sensory set, the only motor
neurons carrying measurable audio information (README). Other descending neurons are not notes.

GROUPING (fixed, audio-independent; MidiConfig.voice_grouping):
  "family" (default) one group per type family (leading letters of the type: DNge145 -> DNge)
  "type"             one group per type
Groups are ordered by name, members by bodyId; group k gets scale degree k. Which frequency bands
end up driving which note is decided by the connectome. The sensory group each voice group
correlates with best (cached OU transmission run) is stored as a DIAGNOSTIC only.

MECHANISM per group:
  "rate"   (>= 2 members) group rate = member spikes in the tick / members / tick seconds;
           note_on when not sounding and rate > thresh_on, note_off when sounding and rate <= thresh_off
           (per-group thresh_on_pct / thresh_off_pct percentiles of the pooled test (c) runs);
           velocity = rate over the group's observed rate range.
  "spike"  (1 member; a rate from one neuron in 50 ms is a single-spike detector, not a rate)
           note_on in a tick with >= 1 spike, note_off in a tick with none; velocity = its rate over the
           trailing spike_velocity_window_ms, over that signal's observed range.
CC cc_number = whole motor population mean rate over its observed range (bulk state, texture only).

Each render writes runs/<name>/out.mid, out.wav (out.mid played through FluidSynth with MidiConfig.soundfont,
no program change, i.e. preset 0), timeline.parquet (one row per tick), spikes_hop1_motor.parquet
((time_ms, bodyId) for every voice spike) and render_summary.json.
"""
import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mido
import numpy as np
import pandas as pd

from config import MidiConfig, Paths, TransmissionConfig
from data.groups import hop_motor, load_neurons_and_indices, sensory_hop_distances

CACHE = ROOT / Paths().cache
GROUPS_JSON = CACHE / "motor_groups.json"


# =============================================================================
# voice groups
# =============================================================================

def group_key(type_name, cfg):
    if cfg.voice_grouping == "type":
        return type_name
    if cfg.voice_grouping == "family":
        m = re.match(cfg.family_regex, type_name)
        if not m:
            raise ValueError(f"type {type_name!r} has no family prefix under {cfg.family_regex!r}")
        return m.group(0)
    raise ValueError(f"unknown voice_grouping {cfg.voice_grouping!r}")


def build_voice_groups(cfg=None):
    from experiments.transmission import SIM_CACHE, abs_corr

    cfg = MidiConfig() if cfg is None else cfg
    neurons, indices = load_neurons_and_indices()
    dist = sensory_hop_distances(indices)
    voices = hop_motor(indices, dist, cfg.voice_hop)
    sub = neurons.iloc[voices].copy()
    if sub["type"].isna().any():
        raise ValueError("untyped voice neurons; grouping needs a type for every voice")
    sub["group_key"] = [group_key(t, cfg) for t in sub["type"]]
    keys = sorted(sub["group_key"].unique())
    if len(keys) > cfg.max_voices:
        raise ValueError(f"{len(keys)} voice groups exceed max_voices={cfg.max_voices}; group by connectivity instead")

    d = np.load(SIM_CACHE)
    counts, drive = d["counts"], d["drive_binned"]
    groups = []
    for k, key in enumerate(keys):
        members = sub[sub["group_key"] == key].sort_values("bodyId")
        idx = members["matrix_index"].to_numpy()
        rate = counts[:, idx].sum(axis=1, keepdims=True).T.astype(np.float64)
        r = abs_corr(rate, drive.T.astype(np.float64))[0]
        groups.append({"group": k, "name": key, "mechanism": "spike" if len(idx) == 1 else "rate",
                       "types": members["type"].tolist(), "bodyIds": members["bodyId"].astype(int).tolist(),
                       "matrix_index": idx.astype(int).tolist(),
                       "diagnostic_ou": {"best_sensory_group": int(r.argmax()), "abs_r_best": float(r.max()),
                                         "abs_r_by_sensory_group": [float(v) for v in r],
                                         "spikes_in_ou_run": int(counts[:, idx].sum())}})
    out = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "voice_grouping": cfg.voice_grouping, "family_regex": cfg.family_regex,
           "rule": f"motor neurons at BFS hop {cfg.voice_hop} from the sensory set; one group per "
                   f"{cfg.voice_grouping}; groups ordered by name, members by bodyId; group k -> scale degree k; "
                   "one-member groups use the spike mechanism",
           "voice_hop": cfg.voice_hop, "n_neurons": int(voices.size), "n_groups": len(keys), "groups": groups,
           "diagnostic_note": "diagnostic_ou.* is the |r| of each group's summed rate with the 8 sensory-group OU drives "
                              "of cache/transmission_sim.npz; reported only, never used for grouping or notes"}
    GROUPS_JSON.write_text(json.dumps(out, indent=2))
    sizes = [len(g["bodyIds"]) for g in groups]
    print(f"voices: {voices.size} motor neurons at hop {cfg.voice_hop}; grouping '{cfg.voice_grouping}' -> {len(keys)} groups, "
          f"sizes {sizes}; spike mechanism: {[g['name'] for g in groups if g['mechanism'] == 'spike']}")
    for g in groups:
        dg = g["diagnostic_ou"]
        types = pd.Series(g["types"]).value_counts().sort_index()
        print(f"  {g['group']:>2} {g['name']:<9} n={len(g['bodyIds']):<3} {g['mechanism']:<5} OU spikes {dg['spikes_in_ou_run']:>6} | "
              f"diagnostic best sensory group {dg['best_sensory_group']} |r| {dg['abs_r_best']:.3f} | types "
              + ", ".join(f"{t}x{n}" if n > 1 else t for t, n in types.items()))
    print(f"wrote {GROUPS_JSON.relative_to(ROOT)}")
    return out


def load_groups(need_thresholds=True):
    if not GROUPS_JSON.exists():
        raise FileNotFoundError("cache/motor_groups.json missing; run `python -m midi.midi_out groups`")
    g = json.loads(GROUPS_JSON.read_text())
    if g.get("voice_grouping") != MidiConfig().voice_grouping:
        raise ValueError(f"motor_groups.json uses grouping {g.get('voice_grouping')!r}, config says "
                         f"{MidiConfig().voice_grouping!r}; rebuild the groups")
    if need_thresholds and "thresholds" not in g:
        raise KeyError("motor_groups.json has no thresholds; run `python -m midi.midi_out thresholds CLIP ...`")
    return g


# =============================================================================
# signals, thresholds
# =============================================================================

def group_signals(counts, groups, tick_ms, cfg):
    """(spikes, rates, velocity_signal), each (ticks, G). velocity_signal = rate for rate groups and the
    trailing spike_velocity_window_ms rate for spike groups."""
    tick_s = tick_ms / 1e3
    spikes = np.stack([counts[:, g["matrix_index"]].sum(axis=1) for g in groups["groups"]], axis=1).astype(np.float64)
    sizes = np.array([len(g["matrix_index"]) for g in groups["groups"]])
    rates = spikes / sizes / tick_s
    w = max(1, round(cfg.spike_velocity_window_ms / tick_ms))
    csum = np.cumsum(np.vstack([np.zeros((1, spikes.shape[1])), spikes]), axis=0)
    t = np.arange(1, spikes.shape[0] + 1)
    lo = np.maximum(t - w, 0)
    trailing = (csum[t] - csum[lo]) / (t - lo)[:, None] / sizes / tick_s
    spike_mask = np.array([g["mechanism"] == "spike" for g in groups["groups"]])
    velocity = np.where(spike_mask[None, :], trailing, rates)
    return spikes, rates, velocity


def motor_population_rate(counts, indices, tick_ms):
    return counts[:, indices["motor_idx"]].sum(axis=1) / len(indices["motor_idx"]) / (tick_ms / 1e3)


def derive_thresholds(clips):
    from experiments.audio_run import choose_offset, run_clip

    cfg, tc = MidiConfig(), TransmissionConfig()
    groups = load_groups(need_thresholds=False)
    _, indices = load_neurons_and_indices()
    rates, vels, pops, sources = [], [], [], []
    for clip in clips:
        off = choose_offset(clip)
        run = run_clip(clip, off)
        _, r, v = group_signals(run["counts"], groups, tc.bin_ms, cfg)
        rates.append(r)
        vels.append(v)
        pops.append(motor_population_rate(run["counts"], indices, tc.bin_ms))
        sources.append({"clip": str(clip), "offset_s": off, "ticks": int(run["meta"]["n_ticks"]),
                        "level_reference": run["meta"]["level_reference"]})
    R, V, P = np.concatenate(rates), np.concatenate(vels), np.concatenate(pops)
    rate_groups = np.array([g["mechanism"] == "rate" for g in groups["groups"]])
    on = np.where(rate_groups, np.percentile(R, cfg.thresh_on_pct, axis=0), np.nan)
    off = np.where(rate_groups, np.percentile(R, cfg.thresh_off_pct, axis=0), np.nan)
    groups["thresholds"] = {
        "sources": sources, "thresh_on_pct": cfg.thresh_on_pct, "thresh_off_pct": cfg.thresh_off_pct,
        "on_hz": [None if np.isnan(x) else float(x) for x in on], "off_hz": [None if np.isnan(x) else float(x) for x in off],
        "velocity_min_hz": V.min(axis=0).tolist(), "velocity_max_hz": V.max(axis=0).tolist(),
        "rate_min_hz": R.min(axis=0).tolist(), "rate_max_hz": R.max(axis=0).tolist(),
        "population_min_hz": float(P.min()), "population_max_hz": float(P.max()), "ticks_pooled": int(R.shape[0]),
    }
    GROUPS_JSON.write_text(json.dumps(groups, indent=2))
    print(f"thresholds from {R.shape[0]} pooled ticks of {len(clips)} runs: rate groups on p{cfg.thresh_on_pct:g} / off "
          f"p{cfg.thresh_off_pct:g}; spike groups: velocity from the trailing {cfg.spike_velocity_window_ms:g} ms rate")
    for g, a, b, vlo, vhi, rlo, rhi in zip(groups["groups"], on, off, V.min(axis=0), V.max(axis=0), R.min(axis=0), R.max(axis=0)):
        thr = f"on {a:6.3g} Hz  off {b:6.3g} Hz" if g["mechanism"] == "rate" else "spike-gated           "
        print(f"  {g['group']:>2} {g['name']:<9} n={len(g['bodyIds']):<3} {g['mechanism']:<5} {thr}  rate range {rlo:.3g}-{rhi:.3g} Hz  "
              f"velocity signal range {vlo:.3g}-{vhi:.3g} Hz; rate resolution {1e3 / tc.bin_ms / len(g['bodyIds']):.3g} Hz")
    print(f"  motor population range {P.min():.3g}-{P.max():.3g} Hz; wrote {GROUPS_JSON.relative_to(ROOT)}")
    return groups["thresholds"]


# =============================================================================
# sonification
# =============================================================================

def note_numbers(n_groups, cfg):
    if cfg.scale_mode == "chromatic":
        notes = [cfg.base_note + k for k in range(n_groups)]
    elif cfg.scale_mode == "pentatonic":
        iv = cfg.pentatonic_intervals
        notes = [cfg.pentatonic_base_note + 12 * (k // len(iv)) + iv[k % len(iv)] for k in range(n_groups)]
    else:
        raise ValueError(f"unknown scale_mode {cfg.scale_mode!r}")
    if min(notes) < 0 or max(notes) > 127:
        raise ValueError(f"{cfg.scale_mode} notes {min(notes)}..{max(notes)} fall outside MIDI 0..127")
    return notes


def scale_to(x, lo, hi, out_lo, out_hi):
    if hi <= lo:
        return out_hi
    return int(round(out_lo + (out_hi - out_lo) * min(max((x - lo) / (hi - lo), 0.0), 1.0)))


def sonify(mechanisms, spikes, rates, velocity, population, thresholds, cfg):
    """Events (tick, kind, a, b): ("on", note, velocity), ("off", note, 0), ("cc", number, value).
    Every note still sounding after the last tick is switched off at tick = n_ticks."""
    T, G = rates.shape
    notes = note_numbers(G, cfg)
    if len(mechanisms) != G or len(thresholds["on_hz"]) != G:
        raise ValueError("mechanisms / thresholds do not match the groups")
    vmin, vmax = thresholds["velocity_min_hz"], thresholds["velocity_max_hz"]
    for g, m in enumerate(mechanisms):
        if m == "rate" and not thresholds["off_hz"][g] <= thresholds["on_hz"][g]:
            raise ValueError(f"group {g}: off > on")
        if m not in ("rate", "spike"):
            raise ValueError(f"unknown mechanism {m!r}")
    sounding = np.zeros(G, dtype=bool)
    events, last_cc = [], None
    for t in range(T):
        cc = scale_to(population[t], thresholds["population_min_hz"], thresholds["population_max_hz"], 0, 127)
        if cc != last_cc:
            events.append((t, "cc", cfg.cc_number, cc))
            last_cc = cc
        for g in range(G):
            if mechanisms[g] == "rate":
                start = rates[t, g] > thresholds["on_hz"][g]
                stop = rates[t, g] <= thresholds["off_hz"][g]
            else:
                start = spikes[t, g] >= 1
                stop = spikes[t, g] < 1
            if sounding[g] and stop:
                events.append((t, "off", notes[g], 0))
                sounding[g] = False
            elif not sounding[g] and start:
                events.append((t, "on", notes[g], scale_to(velocity[t, g], vmin[g], vmax[g], cfg.velocity_min, cfg.velocity_max)))
                sounding[g] = True
    for g in np.flatnonzero(sounding):
        events.append((T, "off", notes[g], 0))
    return events


def ticks_per_tick(cfg):
    us_per_midi_tick = cfg.tempo_us_per_beat / cfg.ticks_per_beat
    n = cfg.tick_ms * 1e3 / us_per_midi_tick
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"tick_ms {cfg.tick_ms} is not a whole number of MIDI ticks at this tempo")
    return int(round(n))


def write_midi(events, n_ticks, path, cfg):
    per = ticks_per_tick(cfg)
    mf = mido.MidiFile(ticks_per_beat=cfg.ticks_per_beat, type=0)
    track = mido.MidiTrack()
    mf.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=cfg.tempo_us_per_beat, time=0))
    now = 0
    order = {"off": 0, "cc": 1, "on": 2}
    for t, kind, a, b in sorted(events, key=lambda e: (e[0], order[e[1]])):
        abs_ticks = t * per
        delta, now = abs_ticks - now, abs_ticks
        if kind == "on":
            track.append(mido.Message("note_on", channel=cfg.channel, note=a, velocity=b, time=delta))
        elif kind == "off":
            track.append(mido.Message("note_off", channel=cfg.channel, note=a, velocity=0, time=delta))
        else:
            track.append(mido.Message("control_change", channel=cfg.channel, control=a, value=b, time=delta))
    track.append(mido.MetaMessage("end_of_track", time=n_ticks * per - now))
    path.parent.mkdir(parents=True, exist_ok=True)
    mf.save(path)
    return mido.MidiFile(path).length


# =============================================================================
# render
# =============================================================================

def band_drivers(signal, bands, tc):
    """Per group: |r| of its tick signal with each band, best band, and whether that beats the group's own
    joint-shift null maximum (TransmissionConfig.n_shuffles shifts, >= min_shift_ms)."""
    from experiments.transmission import abs_corr

    T = signal.shape[0]
    R, S = signal.T.astype(np.float64), bands.T.astype(np.float64)
    real = abs_corr(R, S)
    min_shift = round(tc.min_shift_ms / tc.bin_ms)
    rng = np.random.default_rng(tc.seed + 3)
    null = np.stack([abs_corr(R, np.roll(S, int(s), axis=1)).max(axis=1)
                     for s in rng.integers(min_shift, T - min_shift + 1, size=tc.n_shuffles)])
    return real, real.max(axis=1) > null.max(axis=0), null.max(axis=0)


def render_wav(mid_path, cfg):
    """out.mid -> out.wav with FluidSynth (offline, fast-render). Raises if the file is missing, shorter than the
    MIDI, or digitally silent."""
    import soundfile as sf

    sf2 = Path(cfg.soundfont)
    if not sf2.is_file():
        raise FileNotFoundError(f"MidiConfig.soundfont {sf2} does not exist")
    wav = mid_path.with_suffix(".wav")
    cmd = [cfg.fluidsynth_bin, "-n", "-i", "-q", "-r", str(cfg.wav_sample_rate), "-T", "wav", "-F", str(wav), str(sf2), str(mid_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"fluidsynth exited {res.returncode}: {res.stderr.strip()}")
    data, sr = sf.read(wav, always_2d=True)
    midi_len = mido.MidiFile(mid_path).length
    info = {"path": str(wav.relative_to(ROOT)), "sample_rate": sr, "channels": data.shape[1], "seconds": data.shape[0] / sr,
            "midi_seconds": midi_len, "peak": float(np.abs(data).max()), "rms": float(np.sqrt((data ** 2).mean())),
            "soundfont": str(sf2), "fluidsynth_stderr": res.stderr.strip()}
    if sr != cfg.wav_sample_rate or info["seconds"] < midi_len - 0.05:
        raise AssertionError(f"wav render mismatch: {info}")
    if info["peak"] == 0.0:
        raise AssertionError(f"{wav} is silent")
    return info


def render(clip, offset, scale_mode, name=None):
    from experiments.audio_run import run_clip

    cfg, tc = replace(MidiConfig(), scale_mode=scale_mode), TransmissionConfig()
    groups = load_groups()
    neurons, indices = load_neurons_and_indices()
    run = run_clip(clip, offset)
    meta = run["meta"]
    if meta["voice_idx"] != sorted(i for g in groups["groups"] for i in g["matrix_index"]):
        raise ValueError("the run's voice neurons differ from motor_groups.json")
    name = name or f"{Path(clip).stem}_{meta['level_reference']}_{offset:g}s_{cfg.voice_grouping}_{scale_mode}"
    out_dir = ROOT / Paths().runs / name
    mechanisms = [g["mechanism"] for g in groups["groups"]]
    spikes, rates, velocity = group_signals(run["counts"], groups, tc.bin_ms, cfg)
    pop = motor_population_rate(run["counts"], indices, tc.bin_ms)
    events = sonify(mechanisms, spikes, rates, velocity, pop, groups["thresholds"], cfg)
    T = rates.shape[0]
    length = write_midi(events, T, out_dir / "out.mid", cfg)
    expected = T * cfg.tick_ms / 1e3
    if abs(length - expected) > 1e-6:
        raise AssertionError(f"MIDI length {length} s != simulated {expected} s")

    notes = note_numbers(len(groups["groups"]), cfg)
    per_tick = {t: {"note_on": [], "note_off": [], "velocity": [], "cc": None} for t in range(T + 1)}
    for t, kind, a, b in events:
        if kind == "on":
            per_tick[t]["note_on"].append(a)
            per_tick[t]["velocity"].append(b)
        elif kind == "off":
            per_tick[t]["note_off"].append(a)
        else:
            per_tick[t]["cc"] = b
    tl = pd.DataFrame({"tick": np.arange(T), "time_s": np.arange(T) * cfg.tick_ms / 1e3})
    for b in range(run["signals"].shape[1]):
        tl[f"band_{b}_current"] = run["signals"][:, b]
    for g in groups["groups"]:
        col = f"g{g['group']:02d}_{g['name']}"
        tl[f"{col}_rate_hz"] = rates[:, g["group"]]
        tl[f"{col}_spikes"] = spikes[:, g["group"]].astype(np.int32)
        if g["mechanism"] == "spike":
            tl[f"{col}_velocity_signal_hz"] = velocity[:, g["group"]]
    tl["motor_population_rate_hz"] = pop
    tl[f"cc{cfg.cc_number}"] = pd.Series([per_tick[t]["cc"] for t in range(T)], dtype="Int64")
    for col in ("note_on", "note_off", "velocity"):
        tl[col] = [per_tick[t][col] for t in range(T)]
    out_dir.mkdir(parents=True, exist_ok=True)
    tl.to_parquet(out_dir / "timeline.parquet", index=False)

    body = neurons["bodyId"].to_numpy()
    sp = pd.DataFrame({"time_ms": run["spike_steps"].astype(np.float64) * meta["dt_ms"],
                       "bodyId": body[run["spike_idx"]].astype(np.int64)})
    sp.to_parquet(out_dir / "spikes_hop1_motor.parquet", index=False)

    real, beats, null_max = band_drivers(rates, run["signals"], tc)
    n_on = np.zeros(len(groups["groups"]), dtype=int)
    for t, kind, a, b in events:
        if kind == "on":
            n_on[notes.index(a)] += 1
    summary = []
    print(f"\n{name}: {T} ticks = {expected:g} s, MIDI length {length:.3f} s, {int(n_on.sum())} notes, "
          f"{sum(1 for e in events if e[1] == 'cc')} CC{cfg.cc_number} events; notes closed at the end {len(per_tick[T]['note_off'])}")
    print(f"  {'grp':>3} {'name':<6} {'n':>3} {'mech':<5} {'note':>4} {'notes':>5} {'mean Hz':>7} {'sounding':>8} | "
          f"best band |r| (own joint-null max) -> drives?")
    sounding_frac = {}
    for g in groups["groups"]:
        k = g["group"]
        on_ticks, cur = 0, False
        for t in range(T):
            for kind in ("off", "on"):
                if notes[k] in per_tick[t]["note_on" if kind == "on" else "note_off"]:
                    cur = kind == "on"
            on_ticks += cur
        sounding_frac[k] = on_ticks / T
        bb = int(real[k].argmax())
        summary.append({"group": k, "name": g["name"], "members": len(g["bodyIds"]), "mechanism": g["mechanism"],
                        "midi_note": notes[k], "notes_played": int(n_on[k]), "sounding_frac": sounding_frac[k],
                        "mean_rate_hz": float(rates[:, k].mean()), "best_band": bb, "abs_r_best": float(real[k].max()),
                        "null_max": float(null_max[k]), "band_drives_note": bool(beats[k]),
                        "abs_r_by_band": [float(v) for v in real[k]]})
        print(f"  {k:>3} {g['name']:<6} {len(g['bodyIds']):>3} {g['mechanism']:<5} {notes[k]:>4} {n_on[k]:>5} "
              f"{rates[:, k].mean():>7.3g} {sounding_frac[k]:>8.0%} | band {bb} |r| {real[k].max():.3f} "
              f"({null_max[k]:.3f}) -> {'YES' if beats[k] else 'no'}")
    wav = render_wav(out_dir / "out.mid", cfg)
    print(f"  out.wav: {wav['seconds']:.3f} s ({wav['midi_seconds']:.3f} s MIDI), {wav['sample_rate']} Hz x {wav['channels']} ch, "
          f"peak {wav['peak']:.3f}, rms {wav['rms']:.4f}" + (f"; fluidsynth: {wav['fluidsynth_stderr']}" if wav["fluidsynth_stderr"] else ""))
    (out_dir / "render_summary.json").write_text(json.dumps({
        "clip": str(clip), "offset_s": offset, "scale_mode": scale_mode, "voice_grouping": cfg.voice_grouping,
        "ticks": T, "seconds": expected, "midi_length_s": length, "notes_total": int(n_on.sum()), "groups": summary,
        "thresholds_source": groups["thresholds"]["sources"], "midi_config": asdict(cfg), "wav": wav, "run_meta": meta},
        indent=2, default=str))
    print(f"  wrote runs/{name}/out.mid, out.wav, timeline.parquet ({len(tl)} rows), spikes_hop1_motor.parquet ({len(sp)} spikes), "
          f"render_summary.json")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("groups")
    th = sub.add_parser("thresholds")
    th.add_argument("clips", nargs="+")
    r = sub.add_parser("render")
    r.add_argument("clip")
    r.add_argument("--offset", default="auto")
    r.add_argument("--scale", default=MidiConfig().scale_mode, choices=("chromatic", "pentatonic"))
    r.add_argument("--name", default=None)
    a = ap.parse_args()
    if a.cmd == "groups":
        build_voice_groups()
    elif a.cmd == "thresholds":
        derive_thresholds(a.clips)
    else:
        from experiments.audio_run import choose_offset

        off = choose_offset(a.clip) if a.offset == "auto" else float(a.offset)
        render(a.clip, off, a.scale, a.name)
