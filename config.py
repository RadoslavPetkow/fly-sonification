# config.py — single source of truth for every tunable constant.
# These are STARTING POINTS. calibrate.py writes the tuned sim/audio values
# to cache/calibration.json; at runtime calibration.json wins over these.

from dataclasses import dataclass

NEUPRINT_SERVER  = "https://neuprint.janelia.org"
NEUPRINT_DATASET = "male-cns:v1.0"
# token: NEUPRINT_APPLICATION_CREDENTIALS env var, read by neuprint-python itself

@dataclass
class DataConfig:
    max_neurons: int = 0            # 0 = whole connectome, no subsetting

    # --- fetch -------------------------------------------------------------
    fetch_status: str = "Traced"    # only neurons with this status are cached
    fetch_batch_size: int = 1000    # source bodyIds per edge-fetch page; lower it
                                    # if the server times out (resume keeps progress)
    fetch_max_retries: int = 5      # retries of one batch on HTTP 5xx; 4xx never retried
    fetch_backoff_s: float = 2.0    # retry k (0-based) waits fetch_backoff_s * 2**k: 2,4,8,16,32s

    # --- edges -------------------------------------------------------------
    fetch_min_weight: int = 1       # what we FETCH and cache as a raw edge list
    matrix_weight_threshold: int = 2  # edges below this are dropped when BUILDING
                                      # the matrix; re-thresholding must never
                                      # require re-fetching
    threshold_report = (1, 2, 3, 5)   # thresholds whose edge/weight loss build prints

    # --- neurotransmitter -> synaptic sign ---------------------------------
    # Coverage in male-cns:v1.0 (recon, Sep 2026): acetylcholine 95,409 /
    # glutamate 28,199 / unclear 22,902 / gaba 20,357 / dopamine 4,447 /
    # histamine 2,242 / serotonin 483 / octopamine 126; 2,257 neurons have no
    # value in any NT field. consensusNt differs from predictedNt for 20,169.
    nt_field: str = "consensusNt"          # consensusNt | predictedNt | celltypePredictedNt
    nt_fallback_field: str = "predictedNt" # used only where nt_field is null
    nt_sign = {
        "acetylcholine": +1,
        "gaba": -1,
        # glutamate is the single biggest modelling assumption in this file. It
        # carries 16.97% of all synaptic weight, so flipping this one value moves
        # the E/I balance by ~17 points. GluCl makes it inhibitory at many central
        # Drosophila synapses, but not all. The Step 6 ablation flips it HERE and
        # rebuilds (28 s) - there is deliberately no second field for it.
        "glutamate": -1,
        "histamine": -1,   # HisCl1 / ort are chloride channels
    }
    nt_modulatory = ("dopamine", "octopamine", "serotonin")
    nt_unclear_values = ("unclear", "", None)
    # Measured on the real build (Sep 2026): with consensusNt on Traced neurons
    # only 1.65% of total synaptic weight has no usable sign (1.54% of the
    # final matrix), from 3,602 neurons. The earlier ~14% figure came from
    # predictedNt over all neurons and does not apply.
    nt_unknown_sign: int = +1

    # --- sensory population ------------------------------------------------
    # The dataset's own functional labels disagree with the textbook
    # "JO-A/B = sound, C/D/E = wind & gravity" split: 34 of 88 JO-B neurons are
    # subclass wind_gravity, and 8 JO-C neurons are subclass auditory. We trust
    # the curators' subclass over the type prefix.
    # Decision, Sep 2026, from the real cached neuron table:
    #   subclass=="auditory"            115 neurons,  89 with pre > 0
    #   JO-A/JO-B type prefix           138 neurons, 112 with pre > 0
    #   union of the two                151 neurons, 125 with pre > 0   <-- chosen
    #   all JO-                         672 neurons, 597 with pre > 0
    # The right side of the antennal nerve is badly truncated in this volume:
    # L has 63 neurons / 62 with output / median 160 presynapses, R has 52 / 27 /
    # median 1, and only 4 right-side neurons have presynapses in AMMC(R). So the
    # audio input is driven from the LEFT side only and is MONO. Right-side and
    # sub-threshold neurons are NOT removed from the network - they remain ordinary
    # neurons in the matrix, they are simply not injected with external current.
    sensory_mode: str = "auditory_or_jo_ab"   # auditory_or_jo_ab | subclass_auditory
                                              # | jo_ab_prefix | jo_all
    sensory_subclass: str = "auditory"
    sensory_type_prefixes = ("JO-A", "JO-B")
    sensory_jo_prefix: str = "JO-"            # sensory_mode == "jo_all"
    sensory_sides = ("L",)                    # rootSide; somaSide is empty for all JO
    sensory_min_pre: int = 5                  # a neuron with almost no presynaptic
                                              # sites transmits nothing but still
                                              # consumes a frequency-band slot
    sensory_roi_check = ("AMMC(L)", "AMMC(R)")  # cross-check only, NOT a selector

    # --- motor population --------------------------------------------------
    # superclass is populated for 166,700 of 176,422 neurons; class for only
    # 26,513 - so superclass is the reliable field. Do NOT select DNs by '^DN'
    # regex: it drags in clock neurons (DN1a, DN1pA/B, DNd01) and endocrine
    # cells (DNES1/2/3), and misses MDN, pIP1, pIP10, pMP2, aSP22, LN-DN1/2.
    motor_superclasses = ("descending_neuron", "sensory_descending")
    motor_require_type: bool = True   # untyped neurons cannot be grouped for MIDI


@dataclass
class SimConfig:
    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    delay_ms: float = 2.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_thresh: float = 1.0
    r_membrane: float = 1.0
    refrac_steps: int = 5               # was 2 (333 Hz cap). A neuron is clamped for refrac_steps
                                        # steps after the spike step, so the rate cap is
                                        # 1000 / ((refrac_steps + 1) * dt_ms) = 166.7 Hz at 5.
    normalization: str = "in_degree"    # raw | in_degree | sqrt_in
    # Measured on the real matrix (Sep 2026). The gain needed for self-sustained
    # activity follows from the connectome itself:
    #     w_anchor(f) = 1000 / (geom * median(signed row sum after norm) * f)
    #     geom = 1 / (1 - exp(-dt_ms/tau_syn_ms)) = 5.517 at dt=1, tau_syn=5
    # in_degree: ~330 for 5 Hz, ~165 for 10 Hz, ~83 for 20 Hz; ~47 to bring a
    #            strong first-layer target to threshold from sensory drive alone.
    # sqrt_in:   ~20 for 5 Hz, ~10 for 10 Hz, ~5 for 20 Hz; ~1.7 first-layer.
    # So w_scale = 1.0 is 5-300x too small depending on normalization, which is
    # exactly why the untuned network is silent. calibrate.py must derive the
    # anchor per normalization and sweep around it, never around 1.0.
    w_scale: float = 1.0                # SET BY calibrate.py
    # Membrane noise is an OU process: stationary std(V) = sigma * sqrt(dt/(2*tau_m))
    # = sigma * 0.158 here. At sigma = 0.05 that is 0.8% of V_thresh, i.e. nothing.
    # Sweep the TARGET std(V) as a fraction of threshold instead:
    #     sigma = frac * V_thresh / sqrt(dt/(2*tau_m))  ->  frac 0.02..0.5 = sigma 0.13..3.2
    sigma_noise: float = 0.05           # SET BY calibrate.py
    # Inhibitory gain: negative weights of the normalized W are multiplied by g_inh
    # once at load (positive weights untouched). 1.0 = no change.
    g_inh: float = 1.0
    # Spike-frequency adaptation: a *= exp(-dt/tau_adapt); a[spikes] += b_adapt;
    # a is subtracted from the membrane drive. b_adapt = 0 disables it.
    tau_adapt_ms: float = 150.0
    b_adapt: float = 0.0                # OFF by default: the v2 sweep showed it suppresses the
                                        # most strongly driven population (the sensory input) hardest
    # Exempt the sensory input set from adaptation (their b = 0; everyone else keeps
    # b_adapt). Johnston's organ afferents respond to sustained stimuli with weak
    # adaptation. Only matters when b_adapt > 0.
    sensory_adapt_exempt: bool = False
    force_mps: bool = False
    seed: int = 0                       # noise RNG; reset() re-seeds for reproducible runs
    rate_window_ms: float = 50.0        # length of the rolling spike-count buffers behind
                                        # get_motor_rates / get_global_rate (max window)
    test_input_rate_hz: float = 100.0   # sim/test_lif.py: Poisson event rate on the sensory
                                        # set; each event is scaled so the MEAN input current
                                        # equals AudioConfig.i_ext_max

@dataclass
class CalibConfig:
    # Protocol for sim/calibrate.py. Every value here was fixed BEFORE any
    # calibration run; none is adjusted after seeing results.
    normalizations = ("raw", "in_degree", "sqrt_in")
    anchor_rates_hz = (5.0, 10.0, 20.0)
    # Anchors measured independently (w_anchor at 5/10/20 Hz); calibrate.py stops
    # if its own numbers differ by more than anchor_tolerance.
    anchor_expected = {"in_degree": (330.0, 165.0, 83.0), "sqrt_in": (20.0, 10.0, 5.0)}
    anchor_tolerance: float = 0.20
    warmup_ms: float = 200.0            # simulated and discarded before every measurement
    # Calibration drive: constant current into the sensory input set, as a multiple
    # of the LIF rheobase (v_thresh / r_membrane). 1.1x -> ~21 Hz noise-free.
    drive_rheobase_mult: float = 1.1
    # Stage A: 1-D w_scale bracket at fixed noise
    stage_a_frac: float = 0.1
    stage_a_center_rate_hz: float = 10.0
    stage_a_log10_span: float = 1.5     # w_anchor(center) * logspace(-span, +span)
    stage_a_points: int = 13
    stage_a_ms: float = 1000.0
    # Stage B: 2-D grid inside the bracket
    stage_b_w_points: int = 7
    stage_b_fracs = (0.02, 0.05, 0.1, 0.2, 0.5)
    stage_b_ms: float = 2000.0
    # "non-zero" activity: at least this fraction of NON-sensory neurons fired
    active_nonsensory_min: float = 0.01
    # Synchrony threshold = geometric mean of the median synchrony at the silent
    # (lowest w_scale) and seizing (highest w_scale) ends of Stage A. Refuse to
    # derive one if the two ends are less than this factor apart.
    si_min_corner_separation: float = 10.0
    # Target regime
    target_rate_hz = (1.0, 8.0)
    target_active_frac: float = 0.40
    target_pick_rate_hz: float = 3.0
    # i_ext_max: the constant current that drives the sensory set to this mean rate
    # in the calibrated network, read off a measured f-I curve.
    sensory_full_scale_rate_hz: float = 100.0
    fi_rheobase_mults = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0)
    fi_ms: float = 500.0
    # Stage C: input-transmission test
    stage_c_ms: float = 5000.0
    effect_noise_mult: float = 3.0      # |effect| must exceed this x the replicate-run difference
    # Replicate seeds for the Stage C noise floor; floor = the LARGEST |mean motor-rate
    # delta| of any replicate vs baseline (v2 used a single replicate).
    stage_c_replicates: int = 3
    bootstrap_n: int = 2000
    budget_minutes: float = 90.0        # Stage A + B wall-time budget

    # --- v2 protocol --------------------------------------------------------------
    # v1 (Stage A/B above) found a tonic seizure: silent, then a jump to 14-34 Hz
    # saturating at ~75 Hz with ~36% active, sensory neurons crushed to <1 Hz
    # (runs/calibrate_v1_stageA_seizure.log). v1-only fields kept for that log's
    # provenance: stage_a_*, stage_b_*, active_nonsensory_min,
    # si_min_corner_separation, target_active_frac. v2 adds g_inh, adaptation and
    # refrac_steps=5, sweeps sqrt_in only, and replaces the target regime:
    sweep_normalization: str = "sqrt_in"
    sweep_frac: float = 0.1
    sweep_g_inh = (1.0, 2.0, 3.0, 5.0, 8.0)
    sweep_w_range = (0.5, 20.0)
    sweep_w_points: int = 7
    sweep_ms: float = 1000.0
    # b_adapt panels: b chosen so ONE spike's adaptation hyperpolarizes the free
    # membrane by this peak fraction of V_thresh (sim.lif_network.adaptation_peak_per_b)
    sweep_adapt_peak_fracs = (0.0, 0.1, 0.3)
    # Revised target (in addition to target_rate_hz). Synchrony is reported, not gated.
    target_drivable_active_frac: float = 0.50     # active among neurons WITH incoming edges and
                                                  # signed input sum >= 0 (net-inhibited ones can't fire)
    target_sensory_preservation: float = 0.50     # sensory rate >= this x its rate at w_scale = 0
    target_max_rate_per_active_hz: float = 50.0   # spikes / active neuron / s

    # --- v3 protocol ----------------------------------------------------------------
    # v2 (runs/calibrate_v2.log) found the regime at g_inh 3, w 5.848, b 0 but the
    # drivable-active gate (>50%) rejected it; sparse activity is normal, and
    # per-active rate + synchrony already measure clique firing. v3 keeps
    # target_rate_hz, replaces that gate with a sanity floor, tightens preservation
    # and per-active rate for the refine pick, and validates with Stage C.
    target_drivable_active_floor: float = 0.10
    # (1) one panel: adaptation with the sensory set exempt, same w points as v2, 1 s
    exempt_g_inh: float = 3.0
    exempt_adapt_peak_frac: float = 0.1          # b = 0.1361
    # "beats plain b_adapt=0": lower per-active rate at every w where b=0 is active
    # (rate >= target_rate_hz[0]) while sensory rate stays >= this x the b=0 value
    exempt_sensory_tolerance: float = 0.9
    # (2) refine around the v2 regime, b_adapt 0, 2 s per point
    refine_g_inh = (2.5, 3.0, 4.0)
    refine_w_range = (4.0, 12.0)
    refine_w_points: int = 7
    refine_ms: float = 2000.0
    refine_pick_rate_hz: float = 4.0
    refine_min_sensory_preservation: float = 0.70   # strictly above
    refine_max_rate_per_active_hz: float = 40.0

    # --- hysteresis test (sim/hysteresis.py) -----------------------------------
    # One continuous run: w_scale ramped log-linearly up over ramp_ms, then back
    # down over ramp_ms, state carried throughout. Rate is binned in bin_ms.
    hysteresis_normalization: str = "sqrt_in"
    hysteresis_w_range = (0.5, 20.0)
    hysteresis_ramp_ms: float = 10000.0
    hysteresis_bin_ms: float = 100.0
    hysteresis_frac: float = 0.1
    # Crossing level for comparing the two branches, and the up/down crossing
    # ratio above which the branches count as a loop. One bin of ramp moves w by
    # x1.037, so a one-bin lag on each branch alone gives x1.075.
    hysteresis_cross_rate_hz: float = 1.0
    hysteresis_loop_min_ratio: float = 1.1

@dataclass
class TransmissionConfig:
    # experiments/transmission.py: does structured input reach the motor layer?
    # Groups: sensory -> AudioConfig.n_bins, motor -> MidiConfig.n_groups (data/groups.py rule).
    duration_ms: float = 60000.0        # measured, after CalibConfig.warmup_ms
    bin_ms: float = 50.0
    ou_tau_ms: float = 200.0            # one independent OU signal per sensory group,
                                        # ZCA-whitened across groups (sample correlation 0)
    drive_low_rheobase_mult: float = 1.0  # drive current spans [this x rheobase, i_ext_max]
                                          # (monotonic part of the measured f-I curve),
                                          # log-uniform via the normal CDF of the OU signal
    max_hop: int = 3
    layer_sample: int = 2000            # neurons sampled per hop layer larger than this
    min_spikes: int = 10                # a neuron enters the MI analysis only with >= this many spikes
    mi_signal_bins: int = 8             # equiprobable signal levels
    mi_response_levels: int = 5         # rank-based response levels (ties share a level)
    n_shuffles: int = 20
    min_shift_ms: float = 1000.0        # circular shifts stay >= 5 OU time constants from zero lag
    null_z_min: float = 3.0             # layer is "above null": z >= this AND real > every shuffle
    # Null model. "joint": every shuffle shifts ALL input signals by the same random offset,
    # which keeps each signal's autocorrelation AND the cross-signal correlation (real music
    # bands correlate at r ~0.6; shifting them independently inflates the best-of-n null).
    # "independent": each signal gets its own offset (the v1 null, kept for comparison).
    null_mode: str = "joint"
    seed: int = 20260917

@dataclass
class ShuffleConfig:
    # experiments/shuffle_control.py: real connectome vs degree-, weight- and sign-preserving shuffles
    clips = ("cache/test_audio/Gymnopedie_No_1.flac", "cache/test_audio/kyuchek.mp3")
    sections_per_clip: int = 3          # by experiments.audio_run.choose_sections: highest-scoring eligible
                                        # sections, non-overlapping first; only if those run out, sections
                                        # overlapping chosen ones by <= max_section_overlap
    max_section_overlap: float = 0.5
    conditions = ("full", "sensory_only", "downstream_only")
    seeds = (1, 2)
    max_repair_iterations: int = 500
    budget_minutes: float = 120.0
    acf_max_lag_ms: float = 2000.0      # autocorrelation timescale = first lag where ACF < 1/e (capped here)
    psd_nperseg: int = 64               # ticks per Welch segment for spectral flatness (3.2 s)
    n_bootstrap: int = 10000
    alpha: float = 0.05
    # experiments/structural_selectivity.py: an ensemble of sensory-only shuffles of the sensory edges alone
    # (same procedure as shuffle_graph; structure only, no simulation), seeds structural_seed0 + i
    structural_n_shuffles: int = 1000
    structural_seed0: int = 100000
    # experiments/partition_control.py: random partitions of the sensory set into groups of the type
    # partition's sizes, each with its own sensory-only shuffle ensemble (distinct seeds)
    partition_n: int = 200
    partition_shuffles: int = 200
    partition_seed0: int = 200000
    # experiments/matched_regime.py: Stage A style scan of shuffled graphs at the calibrated g_inh and noise;
    # a point is in the target regime if population rate is in CalibConfig.target_rate_hz, sensory rate >
    # CalibConfig.refine_min_sensory_preservation x the w_scale = 0 reference, and it is input-driven: with the
    # sensory drive removed (from rest, same noise) the non-sensory rate stays <= AudioConfig.ignition_rate_hz
    regime_graph_seeds = (1, 2)
    regime_conditions = ("full", "downstream_only")
    # experiments/transition_refine.py: geometric refinement (endpoints included) of the matched_regime grid step
    # where each graph's rate first crosses into CalibConfig.target_rate_hz from below
    transition_refine_points: int = 9

@dataclass
class AudioConfig:
    sr: int = 16000
    block_ms: float = 10.0
    window_mult: int = 4
    f_min_hz: float = 80.0              # Drosophila near-field hearing range
    f_max_hz: float = 1000.0
    n_bins: int = 8                     # only 115 auditory sensory neurons exist,
                                        # so 8 bands ~= 14 neurons per band
    # level_reference:
    #  "clip_range" (default, v3): ONE global level offset and ONE global range per clip:
    #     x = (level - p_lo) / (p_hi - p_lo), clipped to [0, 1], p_lo/p_hi = clip_range_percentiles
    #     of all band levels of the whole clip's non-silent frames, pooled over bands (never per
    #     band: relative band loudness survives); current = i_ext_safe * (i_ext_max/i_ext_safe) ** x**compression.
    #     Geometric, because the sensory f-I curve is ~logarithmic in current (~20-30 Hz per
    #     doubling): rate becomes ~linear in dB instead of log(log(amplitude)).
    #  "clip_p95" (v2): dB relative to the clip's pooled 95th percentile, [db_floor, db_ceil] -> current LINEARLY.
    #  "absolute" (v1): dBFS, [db_floor, db_ceil] -> current linearly.
    level_reference: str = "clip_range"  # clip_range | clip_p95 | absolute
    clip_ref_percentile: float = 95.0
    clip_range_percentiles = (5.0, 95.0)
    # A 10 ms frame is SILENT if its total band power is more than silence_rel_db below the
    # clip_ref_percentile of frame total power over the whole clip.
    silence_rel_db: float = -60.0
    # Section rule (experiments.audio_run.choose_offset), decided from the audio alone, before simulating:
    # 20 s windows in 1 s steps, >= section_margin_s from both clip ends, <= section_max_silent_frac silent
    # frames, and every band's level distribution within section_max_ks (two-sample KS statistic,
    # non-silent frames of the section vs of the whole clip); among those, the window whose least-varying
    # band (std of the tick-binned drive current) varies most.
    section_margin_s: float = 2.0
    section_max_silent_frac: float = 0.10
    section_max_ks: float = 0.30
    db_floor: float = -60.0             # clip_p95: dB relative to the clip reference; absolute: dBFS
    db_ceil: float = 0.0                # (0 dB = full-scale sine inside the band)
    i_ext_max: float = 1.0              # SET BY calibrate.py; audio/ reads i_ext_max and
                                        # i_ext_floor from cache/calibration.json, not from here
    # dB -> current: x = (clip(dB) - db_floor) / (db_ceil - db_floor); x ** compression;
    # I = I_lo + (I_hi - I_lo) * x, I_hi = i_ext_max; I_lo = i_ext_safe for clip_p95,
    # floor_margin * i_ext_floor for absolute (v1).
    compression: float = 1.0            # 1.0 = linear in dB; < 1 lifts quiet passages
    floor_margin: float = 1.05
    fft_pad_mult: int = 4               # FFT length = next pow2 >= window * this (band edges
                                        # at 80-110 Hz need several bins; padding adds no resolution)
    # Ignition floor (python -m audio.audio_in floor): constant current into the whole sensory
    # set, each point from rest, warm-up CalibConfig.warmup_ms then ignition_ms measured, per seed.
    # A current IGNITES if the non-sensory population rate exceeds the no-input rate by
    # ignition_rate_hz for every seed. i_ext_floor = smallest current that ignites and above
    # which every tested current also ignites.
    ignition_currents = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25)
    ignition_ms: float = 500.0
    ignition_seeds = (0, 1, 2)
    ignition_rate_hz: float = 1.0       # = CalibConfig.target_rate_hz lower edge
    # tests (audio/test_audio_in.py)
    test_tone_hz = (200.0, 440.0)
    test_tone_dbfs: float = -6.0
    test_tone_seconds: float = 1.0
    test_clip_seconds: float = 20.0
    # i_ext_safe (python -m audio.audio_in safe): lowest constant current, swept upward from
    # i_ext_floor, at which on every seed the recurrent network is running (non-sensory rate
    # above the ignition threshold) AND the sensory set fires at >= safe_min_sensory_hz, with
    # every higher tested current passing too. From rest, CalibConfig.warmup_ms + safe_ms.
    safe_currents = (0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 2.0)
    safe_ms: float = 500.0
    safe_seeds = (0, 1, 2)
    safe_min_sensory_hz: float = 10.0

@dataclass
class MidiConfig:
    tick_ms: float = 50.0               # must equal TransmissionConfig.bin_ms
    n_groups: int = 12                  # v1 motor split for the transmission analysis only;
                                        # voices are the hop-1 motor neurons grouped by type
    voice_hop: int = 1                  # motor neurons at this BFS hop from the sensory set are voices
    # "family" (default): group by type family = leading letters of the type (DNge145 -> DNge,
    # DNg09_a -> DNg, pIP1 -> pIP); "type": one group per type (41 groups, mostly singletons).
    voice_grouping: str = "family"      # family | type
    family_regex: str = r"^[A-Za-z]+"
    max_voices: int = 48                # groups beyond this would need connectivity clustering
    # A one-member group has no meaningful rate in a 50 ms tick (quantized in 20 Hz steps): it is
    # driven by spikes instead: note on in a tick with >= 1 spike, off in a tick with none; its
    # velocity comes from its rate over the trailing spike_velocity_window_ms.
    spike_velocity_window_ms: float = 500.0
    channel: int = 0
    ticks_per_beat: int = 480
    tempo_us_per_beat: int = 500000     # 120 bpm: one 50 ms tick = 48 MIDI ticks exactly
    pentatonic_intervals = (0, 2, 4, 7, 9)
    pentatonic_base_note: int = 24      # 41 type voices span 8.2 octaves in pentatonic; from 48 they
                                        # would pass MIDI 127, so the pentatonic render starts lower
    scale_mode: str = "chromatic"       # chromatic | pentatonic
    base_note: int = 48
    velocity_min: int = 40
    velocity_max: int = 127
    thresh_on_pct: float = 70.0         # per-group percentile of the rates in the test (c) runs
    thresh_off_pct: float = 55.0        # note_on when rate > on; note_off when rate <= off
    cc_number: int = 74
    port_name: str = "FlyBrain"
    # SoundFont for the offline .wav render (midi_out render_wav): the one shipped with Homebrew fluid-synth 2.6.0
    soundfont: str = "/opt/homebrew/Cellar/fluid-synth/2.6.0/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2"
    fluidsynth_bin: str = "fluidsynth"
    wav_sample_rate: int = 44100

@dataclass
class AnatomyConfig:
    """The anatomical view of the web player: data/fetch_positions.py + experiments/export_anatomy.py."""
    # --- neuron set --------------------------------------------------------
    # Chosen from the GRAPH alone (BFS hop from the sensory set), never from firing rate, so the
    # picture cannot be cherry-picked: every sensory neuron, every hop-1 neuron (the hop-1 motor
    # neurons - the MIDI voices - labelled separately), and a fixed-seed uniform sample of hops 2
    # and 3. Hops 4-7 (5,551 neurons) and the 1,139 unreachable ones are left out entirely.
    hop2_sample: int = 2000
    hop3_sample: int = 1000
    sample_seed: int = 0
    layers = ("sensory", "motor", "hop1", "hop2", "hop3")
    # --- position fetch ----------------------------------------------------
    # somaLocation is null for 25,098 of the 165,122 Traced neurons. Where it is, the fallback is the
    # centroid of every synapse of that body (Neuron -Contains-> SynapseSet -Contains-> Synapse), which
    # is a real measured position, not a guess - positions.parquet records which was used per neuron.
    soma_batch: int = 2000       # bodyIds per somaLocation query
    synapse_batch: int = 100     # bodyIds per centroid query; that one aggregates every synapse of
                                 # every listed body, so the batch has to stay small
    # --- projection --------------------------------------------------------
    # Measured on the real soma cloud (140,024 points): x separates left from right (L mean 72,357 vs
    # R 24,717), z separates brain from ventral nerve cord (31,292 vs 96,787), y is dorso-ventral
    # WITHIN each part. So the view with "brain lobes on top, nerve cord below" is (x, z) - horizontal
    # x, vertical z increasing downward from brain to VNC - and y is the depth axis. (x, y) is a
    # frontal view of the brain alone, with the whole VNC projected on top of it.
    proj_horizontal: str = "x"
    proj_vertical: str = "z"
    proj_depth: str = "y"
    pos_decimals: int = 4        # of a 0..1 normalized axis: ~9 voxels ~= 0.07 um on the z extent
    # --- export ------------------------------------------------------------
    clips = ("kyuchek_clip_range_15s", "Gymnopedie_No_1_clip_range_31s")
    max_json_mb: float = 3.0     # over this, cut hop2_sample/hop3_sample - never the spikes
    figure_dpi: int = 130

@dataclass
class Paths:
    cache: str = "cache"
    figures: str = "figures"
    runs: str = "runs"
    docs: str = "docs"