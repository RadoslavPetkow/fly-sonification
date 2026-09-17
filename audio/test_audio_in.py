"""Tests for audio/audio_in.py.

    .venv/bin/python -m audio.test_audio_in tones
    .venv/bin/python -m audio.test_audio_in bands   CLIP [--offset auto|S] [--mapping clip_range|clip_p95|absolute]
    .venv/bin/python -m audio.test_audio_in network CLIP [--offset auto|S] [--mapping clip_p95|absolute]

(a) tones    200 Hz and 440 Hz sines (written to and read back from WAV) must peak in different
             bands, each in the band containing its frequency.
(b) bands    heatmap of band level and mapped current for the test_clip_seconds section, the
             dynamic-range report, and band-band correlation of the section and the whole clip.
(c) network  results -> cache/audio_transmission/<run>__null-<mode>.json (named by run, never
             overwritten by a different run); the section through the calibrated network (experiments/audio_run.py, cached): the 8
             band currents (tick-binned) against the binned rates of the sensory neurons, hop-1
             neurons, a hop-2 sample and motor neurons at hops 1 and 2, with the null and thresholds
             of experiments/transmission.py (TransmissionConfig.null_mode, default joint shift).

--offset auto applies experiments.audio_run.choose_offset (input-only, decided before simulating).
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.colors import LinearSegmentedColormap

from audio.audio_in import AudioDrive, band_edges, band_levels, dynamic_range_report, load_audio, verify_groups_against_transmission
from config import AudioConfig, Paths, TransmissionConfig
from experiments.audio_run import audio_layers, choose_offset, run_clip, run_path, tick_frames
from experiments.transmission import analyze
from sim.calibrate import BLUE_RAMP, SURFACE, TEXT, TEXT_2, banner, header, save

CACHE = ROOT / Paths().cache


def above_in(results, name, key, tc):
    s = results[name][key]
    return s["z"] is not None and s["z"] >= tc.null_z_min and s["mean_best"] > s["null_max"]


def band_correlation(x):
    c = np.corrcoef(x, rowvar=False)
    off = c[~np.eye(c.shape[0], dtype=bool)]
    return c, float(np.nanmean(off)), float(np.nanmax(off)), float(np.nanmin(off))


def tick_bin(x, per_tick):
    n = x.shape[0] // per_tick
    return x[:n * per_tick].reshape(n, per_tick, x.shape[1]).mean(axis=1)


def test_tones():
    header("(a) tones")
    cfg = AudioConfig()
    edges = band_edges(cfg)
    out_dir = CACHE / "test_audio"
    out_dir.mkdir(exist_ok=True)
    amp = 10 ** (cfg.test_tone_dbfs / 20)
    peaks = {}
    for f in cfg.test_tone_hz:
        t = np.arange(round(cfg.test_tone_seconds * cfg.sr)) / cfg.sr
        path = out_dir / f"sine_{f:g}hz.wav"
        sf.write(path, amp * np.sin(2 * np.pi * f * t), cfg.sr)
        y, _ = load_audio(path)
        _, levels = band_levels(y, cfg)
        mean = levels[cfg.window_mult:].mean(axis=0)            # frames whose window lies fully inside the tone
        peak = int(mean.argmax())
        expected = int(np.searchsorted(edges, f, side="right") - 1)
        peaks[f] = peak
        print(f"  {f:g} Hz at {cfg.test_tone_dbfs:g} dBFS: band levels (dB) " + " ".join(f"{v:6.1f}" for v in mean)
              + f" -> peak band {peak} [{edges[peak]:.0f}-{edges[peak + 1]:.0f} Hz], expected {expected}; "
              f"peak level {mean[peak]:.2f} dB")
        assert peak == expected, f"{f} Hz peaked in band {peak}, expected {expected}"
    f1, f2 = cfg.test_tone_hz
    assert peaks[f1] != peaks[f2], "the two tones activate the same band"
    print("  PASS: different bands")


def test_bands(clip, offset, mapping):
    from audio.audio_in import clip_reference, frame_total_db, levels_to_current

    header(f"(b) band activations: {clip} from {offset:g} s ({mapping})")
    cfg, tc = replace(AudioConfig(), level_reference=mapping), TransmissionConfig()
    drive = AudioDrive(cfg)
    per_tick = int(tc.bin_ms / cfg.block_ms)
    times, levels, currents, ref = drive.band_currents(clip, offset, cfg.test_clip_seconds)
    edges = band_edges(cfg)
    labels = [f"{edges[b]:.0f}-{edges[b + 1]:.0f}" for b in range(cfg.n_bins)]
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    extent = [offset + times[0], offset + times[-1] + cfg.block_ms / 1e3, 0, cfg.n_bins]
    for ax, data, title, vmin, vmax, unit, norm in (
            (axes[0], levels, f"band level (dBFS); clip range p{cfg.clip_range_percentiles[0]:g}-p{cfg.clip_range_percentiles[1]:g} "
             f"= {ref['p_lo_db']:.1f}..{ref['p_hi_db']:.1f} dBFS", ref["p_lo_db"], ref["p_hi_db"], "dBFS", None),
            (axes[1], currents, "mapped sensory current (log colour scale)", drive.lo, drive.hi, "current", "log")):
        from matplotlib.colors import LogNorm
        im = ax.imshow(data.T, origin="lower", aspect="auto", cmap=cmap, extent=extent, interpolation="nearest",
                       norm=LogNorm(vmin=vmin, vmax=vmax) if norm == "log" else None,
                       vmin=None if norm == "log" else vmin, vmax=None if norm == "log" else vmax)
        ax.set_yticks(np.arange(cfg.n_bins) + 0.5, labels)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=TEXT, loc="left", fontsize=10)
        ax.set_ylabel("band (Hz)", color=TEXT_2)
        ax.tick_params(colors=TEXT_2, labelsize=8)
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label(unit, color=TEXT_2)
        cb.ax.tick_params(labelsize=8, colors=TEXT_2)
    axes[1].set_xlabel("time in clip (s)", color=TEXT_2)
    fig.suptitle(f"{Path(clip).name} ({mapping}): {cfg.n_bins} log bands, {cfg.block_ms:g} ms hop, current "
                 f"[{drive.lo:.3g}, {drive.hi:.3g}]", color=TEXT, fontsize=10, x=0.01, ha="left")
    save(fig, f"audio_bands_{Path(clip).stem}_{mapping}_{offset:g}s.png")

    rep = dynamic_range_report(levels, cfg, drive.lo, drive.hi, ref)
    a = rep["all_bands"]
    print(f"mapping {mapping}: clip range {ref['p_lo_db']:.2f}..{ref['p_hi_db']:.2f} dBFS "
          f"({ref['p_hi_db'] - ref['p_lo_db']:.1f} dB, non-silent frames; clip silent {ref['silent_frac']:.1%}) -> "
          f"current {drive.lo:.4g}..{drive.hi:.4g} = {rep['window_doublings']:.2f} doublings, geometric, compression {cfg.compression:g}")
    print(f"SECTION all bands: current p5-p95 {a['current_p5']:.3g}-{a['current_p95']:.3g} = {a['doublings']:.2f} doublings; "
          f"implied sensory rate {a['sensory_hz_p5']:.1f}-{a['sensory_hz_p95']:.1f} Hz (span {a['sensory_span_hz']:.1f} Hz); "
          f"clipped low {a['frac_clipped_low']:.1%}, high {a['frac_clipped_high']:.1%}")
    for b in rep["per_band"]:
        print(f"  band {b['band']} {labels[b['band']]:>9} Hz: current p5-p95 {b['current_p5']:.3g}-{b['current_p95']:.3g} "
              f"= {b['doublings']:.2f} doublings, sensory {b['sensory_hz_p5']:.1f}-{b['sensory_hz_p95']:.1f} Hz "
              f"(span {b['sensory_span_hz']:.1f}); clipped low {b['frac_clipped_low']:.1%}, high {b['frac_clipped_high']:.1%}")

    y, _ = load_audio(clip)
    _, all_levels = band_levels(y, cfg)
    silent_all = frame_total_db(all_levels) < ref["silence_threshold_db"]
    all_cur = levels_to_current(all_levels, cfg, drive.lo, drive.hi, ref)
    whole_rep = dynamic_range_report(all_levels[~silent_all], cfg, drive.lo, drive.hi, ref)["all_bands"]
    print(f"WHOLE CLIP (non-silent frames) all bands: current p5-p95 {whole_rep['current_p5']:.3g}-{whole_rep['current_p95']:.3g} "
          f"= {whole_rep['doublings']:.2f} doublings; implied sensory {whole_rep['sensory_hz_p5']:.1f}-{whole_rep['sensory_hz_p95']:.1f} Hz")
    nonsilent_ticks = lambda x, sil: tick_bin(x[~sil], per_tick) if False else tick_bin(x, per_tick)[
        ~(tick_bin(sil[:, None].astype(float), per_tick)[:, 0] > 0)]
    sec_sil = frame_total_db(levels) < ref["silence_threshold_db"]
    sec_db = band_correlation(nonsilent_ticks(levels, sec_sil))
    sec_cur = band_correlation(nonsilent_ticks(currents, sec_sil))
    whole_db = band_correlation(nonsilent_ticks(all_levels, silent_all))
    whole_cur = band_correlation(nonsilent_ticks(all_cur, silent_all))
    print(f"band-band correlation ({tc.bin_ms:g} ms bins, ticks with any silent frame excluded): SECTION current mean "
          f"{sec_cur[1]:.3f} max {sec_cur[2]:.3f}, dB mean {sec_db[1]:.3f} | WHOLE CLIP current mean {whole_cur[1]:.3f}, "
          f"dB mean {whole_db[1]:.3f}")
    return {"report": rep, "whole_clip": whole_rep, "corr_section_current": sec_cur[1:], "corr_section_db": sec_db[1:],
            "corr_whole_current": whole_cur[1:], "corr_whole_db": whole_db[1:]}


def test_network(clip, offset, mapping, force=False):
    header(f"(c) real audio through the calibrated network: {clip} from {offset:g} s ({mapping})")
    tc = TransmissionConfig()
    verify_groups_against_transmission()
    run = run_clip(clip, offset, mapping, force)
    signals = run["signals"]
    counts = run["counts"]
    idx = json.loads((CACHE / "indices.json").read_text())
    rates = counts.sum(axis=0) / run["meta"]["seconds"]
    print(f"{run['meta']['n_ticks']} ticks ({run['meta']['seconds']:g} s); population {rates.mean():.3g} Hz, sensory "
          f"{rates[idx['sensory_idx']].mean():.3g} Hz, motor {rates[idx['motor_idx']].mean():.3g} Hz; drive current "
          f"range {run['meta']['current_range']}")
    sd = signals.std(axis=0)
    c, mean_c, max_c, min_c = band_correlation(signals)
    print(f"band current std: {np.array2string(sd, precision=3)}")
    print(f"band-band correlation of the tick-binned currents: mean {mean_c:.3f}, max {max_c:.3f}, min {min_c:.3f}")
    print(np.array2string(c, precision=2, suppress_small=True, max_line_width=160))
    if (sd == 0).any():
        raise ValueError(f"bands {np.flatnonzero(sd == 0).tolist()} are constant over the section")
    layers = audio_layers(run, tc)
    results, _ = analyze(counts, signals, layers, tc, np.random.default_rng(tc.seed + 2))
    for key, label in (("abs_r", "|r|"), ("mi", "MI (bits)")):
        print(f"\n{label}, best band per unit ({tc.n_shuffles} {tc.null_mode} circular-shift nulls, min shift {tc.min_shift_ms:g} ms):")
        print(f"  {'layer':<18} {'units':>6} {'mean':>8} {'null mean':>10} {'null max':>9} {'z':>8} {'pctile':>7} "
              f"{'units>own null max':>19}")
        for name, r in results.items():
            s = r[key]
            z = "n/a" if s["z"] is None else f"{s['z']:.2f}"
            print(f"  {name:<18} {r['n_units']:>6} {s['mean_best']:>8.4g} {s['null_mean']:>10.4g} {s['null_max']:>9.4g} "
                  f"{z:>8} {s['percentile']:>6.0f}% {s['frac_units_above_own_null_max']:>18.1%}")
    print(f"chance: {1 / (tc.n_shuffles + 1):.1%} of units above their own null max")

    def above(name, key):
        return above_in(results, name, key, tc)

    ok = {k: above("motor at hop 1", k) for k in ("abs_r", "mi")}

    trim = round(1e3 / tc.bin_ms)
    sl = slice(trim, counts.shape[0] - trim)
    interior, _ = analyze(counts[sl], signals[sl], {k: v[:, sl] for k, v in layers.items()}, tc,
                          np.random.default_rng(tc.seed + 2))
    print(f"\nrobustness: first and last {trim * tc.bin_ms / 1e3:g} s removed ({sl.stop - sl.start} ticks)")
    for name, r in interior.items():
        a, m = r["abs_r"], r["mi"]
        print(f"  {name:<18} |r| z {a['z']:6.2f} ({a['frac_units_above_own_null_max']:5.1%})   "
              f"MI z {m['z']:6.2f} ({m['frac_units_above_own_null_max']:5.1%})")
    ok_interior = {k: above_in(interior, "motor at hop 1", k, tc) for k in ("abs_r", "mi")}
    out_dir = CACHE / "audio_transmission"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{run_path(clip, offset, mapping).stem}__null-{tc.null_mode}.json"
    out_path.write_text(json.dumps({"clip": str(clip), "offset_s": offset, "mapping": mapping, "null_mode": tc.null_mode,
                                    "results": results, "band_correlation": c.tolist(), "band_corr_mean": mean_c,
                                    "band_corr_max": max_c, "motor_hop1_above_null": ok, "interior": interior,
                                    "motor_hop1_above_null_interior": ok_interior, "meta": run["meta"]}, indent=2, default=str))
    print(f"wrote {out_path.relative_to(ROOT)}")
    if not any(ok.values()) or not any(ok_interior.values()):
        banner(f"TRANSMISSION TO THE HOP-1 MOTOR NEURONS IS AT CHANCE for this audio: above null as run "
               f"{ok}, interior {ok_interior}.")
        raise SystemExit(3)
    banner(f"Transmission survives this audio at the hop-1 motor neurons: as run {ok}, interior {ok_interior}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tones")
    for name in ("bands", "network"):
        p = sub.add_parser(name)
        p.add_argument("clip")
        p.add_argument("--offset", default="auto", help="start of the analysed section (s), or 'auto'")
        p.add_argument("--mapping", default=AudioConfig().level_reference, choices=("clip_range", "clip_p95", "absolute"))
        p.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.cmd == "tones":
        test_tones()
    else:
        off = choose_offset(args.clip, args.mapping) if args.offset == "auto" else float(args.offset)
        if args.cmd == "bands":
            test_bands(args.clip, off, args.mapping)
        else:
            test_network(args.clip, off, args.mapping, args.force)
