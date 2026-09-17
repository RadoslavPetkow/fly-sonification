## Plain verdict and interpretation

Written after reading the numbers; the mechanical flags are in the previous section, the two follow-up controls
(matched regime, random partitions) in their own sections below. Per-run values are in
`cache/shuffle_control_results.json`, `cache/matched_regime.json` and `cache/partition_control.json`.

### What we are allowed to claim

**1. Full shuffles have no working window at g_inh 2.5; downstream-only shuffles have a narrow one (Control 1a
plus refinement).** The 13-point scan (1.78x steps) found no window for any of the 4 shuffled graphs, but they all
went from < 1 Hz to 24-27 Hz between two neighbouring points, so each of those steps was re-sampled at 9 points
(1.075x). The result section "Control 1a" below is the coarse scan; its verdict is superseded by the refinement.
- **Real graph (positive control):** passes from 0.237x its anchor on (1.8-2.7 Hz, sensory 75-93%, 0 Hz without
  drive). It also passes on the coarse grid at 0.316x and 0.562x and fails at 1x (11.7 Hz). Window width >= 2.37x
  in w_scale (upper edge not refined).
- **Full shuffle, seeds 1 and 2:** still no passing point. Each jumps from 0.045 / 0.056 Hz to 20.9 / 16.4 Hz
  within ONE 1.075x step. A window narrower than 1.075x is not excluded.
- **Downstream-only shuffle, seeds 1 and 2:** WINDOW EXISTS. 3 of 9 refined points pass all three pre-registered
  conditions each (seed 1: 1.23, 2.16, 5.13 Hz; seed 2: 1.28, 2.20, 4.96 Hz; 0 Hz without drive in all six).
  The rise is graded, not a jump. The window is between 1.155x and 1.33x wide in w_scale, at most a third of the
  real graph's width in log w_scale. The sensory neurons in these passing points fire at 161-187% of the reference
  rate (real graph: 75-93%); the pre-registered preservation condition is one-sided (> 70%), so they pass.

So the earlier claim "degree-, weight- and sign-matched random graphs do not admit a stable low-rate
input-driven state at this g_inh" does NOT survive. It holds for the two full shuffles at 1.075x resolution. It
fails for the two downstream-only shuffles, which keep the real sensory neurons' outgoing wiring and have a
narrow window. What survives: the real graph's window is much wider (>= 2.37x vs <= 1.33x) and sits with the
sensory neurons near their unperturbed rate. With 2 graphs per condition, the full vs downstream-only difference
is an observation, not a tested effect.

Control 1b (recalibrate each shuffled graph to its own window, re-run transmission at matched rates) is now
applicable to the downstream-only shuffles and was NOT run: the experimental work was closed with this
refinement. The Step 6 full- and downstream-shuffle transmission and dynamics differences are therefore still
comparisons at the real calibration (w_scale 4.80, ~1.36x the shuffles' own anchors, ~35 Hz), i.e. regime
differences, not matched-rate comparisons.

**2. The pre-registered prediction for downstream-only was wrong.** Downstream-only was expected to change little
because hop 2 and hop 3 carry no measurable audio information. At the real calibration it changes as much as
the full shuffle, because it moves the operating point: its own window lies at 0.60-0.70x its anchor, far
below the w_scale the real graph was calibrated at. The structure beyond hop 1 carries no audio information but
sets where, and how wide, the low-rate input-driven regime is.

**3. There is no evidence that the specific sensory-to-descending wiring improves transmission.** With the
regime intact (sensory-only shuffle, same 56 targets, afferent groups re-assigned), real minus shuffled |r|
excess is negative in all 5 sections (mean -0.038 [-0.082, -0.007], nominal p 0.017) and MI excess negative in
all 5 (p 0.058); nothing survives a family-wise correction (smallest attainable p 0.004 > Bonferroni 0.0014).

**4. The structural "band selectivity" of that wiring is JO type structure, not band structure (Control 2).** The
real-vs-shuffled entropy gap is -0.1215 for the type-sorted groups; over 200 random partitions of the same sizes
it is +0.0275 on average (none significantly selective), and the type-sorted gap lies at the 0th percentile. With
type-blind groups the real input is slightly LESS concentrated than shuffled. The earlier statement "the real
wiring is band-group-selective (10.5 SD)" therefore means only that different JO types contact different
descending neurons, which the sensory grouping by type builds in. Since which group is called which frequency
band is this project's assignment, not measured tonotopy, the wiring gives no support for a band-to-note
correspondence.

### What this means for the sonification
The output depends on the real connectome through the operating regime its topology gives at the calibrated gain.
That regime is wide in the real graph, narrow in downstream-only shuffles, and not found in full shuffles at
1.075x resolution. The output does not, at the level measured here, depend on a frequency-specific
sensory-to-descending wiring: the apparent selectivity is type structure, and randomizing which afferent group
contacts which target does not reduce transmission.

Limitations: 5 sections (the 2 kyuchek sections overlap by 7 s), 2 seeds per shuffle condition, 2 shuffled
graphs per condition in Control 1a, 1.075x resolution at the transition step only (the real graph's upper edge
not refined), one calibration, one g_inh, no matched-rate transmission comparison (Control 1b not run).
