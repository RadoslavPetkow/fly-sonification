# fly-sonification

Audio drives the auditory sensory neurons of a leaky integrate-and-fire network
built on the full *Drosophila* male CNS connectome (neuPrint `male-cns:v1.0`);
descending-neuron activity is turned into MIDI.

Standing rules for working in this repo are in `CLAUDE.md`; every tunable
constant lives in `config.py`; calibrated values live in `cache/calibration.json`.

## Pipeline

| step | module | status |
|---|---|---|
| 0 | `explore/ping.py`, `explore/recon.py` | done |
| 1 | `data/fetch_connectome.py` (`fetch`, `build`, `check`) | done |
| 2 | `sim/lif_network.py`, `sim/normalize.py` | done |
| 3 | `sim/calibrate.py` (+ `sim/hysteresis.py`, `sim/test_mechanisms.py`) | done, `validated: false` (see below) |
| - | `experiments/transmission.py` | done |
| 4 | `audio/audio_in.py` (+ `audio/test_audio_in.py`) | done |
| 5 | `midi/midi_out.py` (+ `midi/test_midi_out.py`) | done |
| 6 | `experiments/shuffle_control.py`, `experiments/shuffle_graph.py`, `experiments/structural_selectivity.py`, `experiments/matched_regime.py`, `experiments/transition_refine.py`, `experiments/partition_control.py` | done |
| 7 | realtime | not pursued: measured limitation (below) |
| 8 | plasticity | dropped (below) |

```bash
.venv/bin/python -m data.fetch_connectome fetch      # once, ~25 min
.venv/bin/python -m data.fetch_connectome build      # ~30 s
.venv/bin/python -m sim.calibrate                    # v3 protocol
.venv/bin/python -m experiments.transmission         # cached after the first run
```

## Findings to build on

### Network regime (Step 3)
- The untuned network is bistable: silent from rest until an ignition gain, then
  a tonic high-rate state (`figures/hysteresis_sqrt_in_refrac2.png`).
- Inhibitory gain (`g_inh`) is what makes the transition graded; adaptation
  suppresses the sensory input hardest and is off by default.
- Calibrated regime (`cache/calibration.json`, protocol v3): sqrt_in,
  w_scale 4.80, g_inh 2.5, b_adapt 0, noise frac 0.1, refrac 5 steps;
  ~3.9 Hz population rate, ~28 Hz per active neuron, sensory rate preserved.
- The mean-rate Stage C test fails its "drive doubled" arm (motor 9.32 -> 9.27 Hz)
  and its "drive removed" arm only shows that the network does not ignite from
  rest without input. The network is a switch with a floor, not a gain stage.

### Transmission: information reaches the motor layer MONOSYNAPTICALLY
`experiments/transmission.py`, 60 s, 8 independent OU drives into the 8 sensory
groups, 50 ms bins, 20 circular-shift nulls (`cache/transmission_results.json`,
`figures/transmission_mi_vs_hop.png`).

| layer | units | \|r\| z | units above own null | MI z |
|---|---|---|---|---|
| sensory neurons (hop 0) | 82 | 83.4 | 100% | 510 |
| hop 1 | 320 | 22.2 | 60.6% | 101 |
| hop 2 | 2000 sampled | 1.87 | 10.1% | 1.56 |
| hop 3 | 2000 sampled | 1.57 | 6.3% | 0.91 |
| **motor neurons at hop 1** | **27** | **15.9** | **66.7%** | **36.0** |
| motor neurons at hop 2 | 414 | 0.45 | 7.2% | 0.55 |
| motor neurons at hop 3 | 89 | -0.42 | 1.1% | -0.36 |
| motor, 12 type-family groups | 12 | 2.07 (verdict False) | 16.7% | 0.56 |

Chance level for "units above own null" is 4.8%.

- Stimulus information survives exactly one synapse. At hop 2 it is at chance,
  and that is where 1,057 of the 1,322 motor neurons sit.
- **Grouping motor neurons by type family destroys the signal** (z 2.07, not
  above null): the few informative neurons are diluted into ~110-neuron groups.
- **Step 5 (MIDI) grouping will be built around the hop-1 motor neurons, not
  type families.** The strongest carriers are DNge145 (x2), DNge111 (x2), DNg07,
  DNg09_a, DNge130 and DNp12.
- The pooled "motor (all)" line in `transmission_results.json` reads "information
  reaches the motor layer"; that is carried entirely by the hop-1 neurons and
  should not be read as the whole motor layer transmitting.

### Null model
All transmission statistics use a JOINT circular shift (`TransmissionConfig.null_mode = "joint"`):
every shuffle shifts all 8 input signals by the same offset, keeping each signal's
autocorrelation and the cross-band correlation. The v1 independent shift inflated the null
for correlated music bands, which is why Satie hop 2 scored *below* null.
`python -m experiments.compare_nulls` (`cache/null_comparison.json`), same spike records:

| record / layer | \|r\| z independent -> joint | MI z independent -> joint |
|---|---|---|
| OU 60 s, motor at hop 1 | 15.86 -> 24.09 | 36.03 -> 52.79 |
| OU 60 s, motor (all) | 1.36 -> 1.43 | 3.44 -> 2.92 (verdict True -> False) |
| OU 60 s, motor groups (12) | 2.07 -> 1.76 | 0.56 -> 1.19 |
| Satie 83 s (absolute mapping), motor at hop 1 | 5.54 -> 4.98 | 8.10 -> 7.13 |
| Satie 83 s, hop 2 | -3.54 -> -0.98 | -2.76 -> -0.90 |

### Audio input (Step 4)
- i_ext_floor = 0.7 (network ignites on every seed). Near it the network runs at ~4 Hz
  while the sensory neurons are held at 0-4 Hz by recurrent inhibition.
- i_ext_safe = 0.95: lowest current with the sensory set at >= 10 Hz (mean; 38-50% of
  individual sensory neurons) and the network running, on all 3 seeds.
- **With real music, hop-1 motor transmission ran at about half the OU strength** under the
  v1/v2 mappings: Satie |r| z 5.54 with 35% of units above null (independent null) vs OU
  z 15.9 and 67%.
- **Mapping v3 (`level_reference = "clip_range"`, default).** The sensory f-I curve is ~logarithmic
  in current (1 -> 18 Hz, 2 -> 35, 5 -> 59, 13 -> 105), so mapping dB (already a log) linearly onto
  current compressed twice. Now: one global range per clip, the p5-p95 of all band levels of its
  non-silent frames -> x in [0, 1] -> current = 0.95 * (11.61 / 0.95) ** x (3.61 doublings).
  Pooled current p5-p95 of the test sections, old linear (clip_p95) -> new geometric:
  kyuchek 15-35 s 0.60 -> 3.44 doublings (implied sensory span 18 -> 79 Hz); Satie 31-51 s
  1.46 -> 3.47 doublings (46 -> 80 Hz).
- **Section rule v3** (`experiments.audio_run.choose_offset`, audio only, before simulating):
  20 s windows >= 2 s from both clip ends, <= 10% silent frames, every band's level
  distribution within KS 0.30 of the whole clip's; then the v2 variability score. Chosen:
  Satie 31-51 s (113 of 181 eligible), kyuchek 15-35 s (14 of 14).
  v2 had picked clip boundaries (Satie's ending, kyuchek's start from silence), and kyuchek's
  v2 "pass" was that start-up transient.
- Test (c), joint null, hop-1 motor neurons, same sections, old vs new mapping:

| clip, section | band-band r (current) | linear clip_p95: \|r\| z / MI z | geometric clip_range: \|r\| z / MI z (units above null) | geometric, first and last 1 s removed |
|---|---|---|---|---|
| kyuchek 15-35 s | 0.34 -> 0.27 | 0.50 / -0.12 (at chance) | **6.04 / 9.22 (50% / 46%)** | 8.08 / 9.35 |
| Satie 31-51 s | 0.45 -> 0.37 | 2.47 / 1.60 (at chance) | 2.26 / 3.27 (12% / 18%) | 4.01 / 3.12 |

- **kyuchek transmits once the mapping stops compressing it.** The gain is attributable to the
  mapping: the same section under the linear mapping is at chance.
- Satie's 31-51 s section is now the WEAKER material: MI above null as run and in the interior,
  |r| only in the interior (as run z 2.26 < 3). Its v2 section (184-204 s, the ending) scored higher.
- Prediction "percussive music has higher band-band correlation" was WRONG on every measure:
  kyuchek 0.27 (section current) / 0.33 (whole clip) vs Satie 0.37 / 0.47.

### MIDI (Step 5)
- Voices: the 56 motor neurons at hop 1. Grouping (`MidiConfig.voice_grouping`):
  `"family"` (default) = leading letters of the type -> 7 groups, ordered by family name then
  bodyId: DNb 1, DNc 2, DNg 22, DNge 17, DNp 12, DNpe 1, pIP 1. `"type"` (41 groups) still selectable.
  Audio-independent; which bands drive which note is left to the connectome.
- This does not contradict "type-family grouping destroys the signal": that was all 1,322
  motor neurons, where 27 informative ones drown; here only the 56 hop-1 neurons are grouped.
- One-member groups (DNb, DNpe, pIP) have no meaningful 50 ms rate; they are spike-gated
  (note on in a tick with a spike, off in a tick without) with velocity from a 500 ms rate.
  In practice DNpe017 and pIP1 never fired in either clip, and DNb05 fires 136-155 Hz
  continuously, so its note is held for the whole render. DNc fires < 2 Hz (on = off = 0 Hz).
  The informative voices are DNg, DNge and DNp.
- Bands beating each group's own joint null: Satie 31-51 s DNg <- band 6, DNge <- band 1;
  kyuchek 15-35 s DNge <- band 5, DNp <- band 7.
- CC74 is the whole motor population's mean rate. It is the network's bulk state and carries
  no measurable audio information (motor at hop 2 and 3 at chance): texture, not signal.
- `scale_mode = "chromatic"` is what gets analysed. `"pentatonic"` is for the final
  listenable render only; it makes anything sound musical and would hide whether the
  network is doing anything.
- Each render: `runs/<name>/out.mid`, `out.wav` (FluidSynth, `MidiConfig.soundfont`), `timeline.parquet`
  (one row per 50 ms tick), `spikes_hop1_motor.parquet`, `render_summary.json`.
- Final deliverables (Satie 31-51 s and kyuchek 15-35 s, chromatic and pentatonic) are listed in `runs/README.md`.
  The pentatonic renders are the listenable ones; they are pitch-quantised to a scale we chose.

### Shuffle control (Step 6) - `experiments/control_report.md`
- 5 replicate sections (Satie 31, 79, 132 s; kyuchek 2, 15 s; kyuchek's overlap by 7 s) x degree-, weight- and
  sign-preserving shuffles (full, sensory-only, downstream-only) x 2 seeds, paired with the real runs.
- Full and downstream-only shuffles put the calibrated network into an input-independent ~35 Hz state (real:
  4-5 Hz). The prediction that downstream-only would change little was WRONG: the structure beyond hop 1 carries
  no audio information but sets the operating point.
- Control 1a (`experiments/matched_regime.py`, `experiments/transition_refine.py`): is that just a calibration
  mismatch? Identical w_scale scan at g_inh 2.5, each graph centred on its own anchor: 13 points (1.78x steps),
  then the step where the rate crosses 1 Hz re-sampled at 9 points (1.075x steps).
  - **Real graph:** window from 0.237x to between 0.562x and 1x its anchor (>= 2.37x wide), 1.8-4.4 Hz, sensory
    75-93% of reference, silent without drive.
  - **Full shuffles (2 seeds):** NO window at 1.075x resolution. They jump from 0.05 Hz to 16-21 Hz within one
    1.075x step.
  - **Downstream-only shuffles (2 seeds):** a NARROW window. 3 of 9 refined points pass all pre-registered
    conditions (1.2-5.1 Hz, silent without drive, sensory 161-187% of reference, which passes the one-sided > 70%
    rule). Width 1.155-1.33x, at most a third of the real graph's in log w_scale. The coarse grid had missed it.
  - The earlier claim "degree-matched random graphs have no working window" therefore does NOT survive. It holds
    for full shuffles at this resolution. For downstream-only shuffles (real sensory output wiring kept) the
    finding is a much narrower window, not no window.
  - Control 1b (transmission at matched rates) now applies to downstream-only and was not run. The Step 6
    full/downstream transmission differences remain regime differences, not a matched comparison.
- Sensory-only (same 56 targets, afferent groups re-assigned): no transmission advantage for the real wiring;
  real minus shuffled |r| excess -0.038 [-0.082, -0.007], negative in all 5 sections, nominal p 0.017, not
  surviving multiple comparisons.
- The real sensory wiring's structural group selectivity (input entropy 0.332 vs 0.453 shuffled, gap -0.1215,
  -10 SD) is JO TYPE structure, not band structure (Control 2, `experiments/partition_control.py`): over 200 random
  partitions of the sensory neurons into groups of the same sizes the gap is +0.0275 on average (range -0.012 to
  +0.057, none significant), and the type-sorted gap is more extreme than all 200 (0th percentile). Because the
  group-to-band assignment is ours, the wiring gives no support for a band-to-note correspondence.
- Net: the sonification depends on the real connectome through the width of the operating regime its topology
  gives (wide in the real graph, narrow in downstream-only shuffles, not found in full shuffles), not through
  frequency-specific sensory-to-descending wiring.

## Status: experimental work closed

Final renders are in `runs/` (see `runs/README.md`); the control verdicts are in `experiments/control_report.md`.
- **Real time is out of reach (measured limitation).** Simulation costs 9.1-9.4 ms of wall time per 1 ms step
  on CPU (M2): about 187 s for each 20 s render, about 9x slower than real time. The pipeline is offline only.
- **Dopamine plasticity is dropped.** Audio transmission here is monosynaptic (hop-1 descending neurons only;
  hop 2 and 3 at chance), and the rest of the network acts as a stabiliser that sets the operating regime
  (Control 1a). A reinforcement rule on the recurrent bulk would have no audio-carrying pathway to act on.
