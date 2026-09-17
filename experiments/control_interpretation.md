## Plain verdict and interpretation

Written after reading the numbers; the mechanical flags are in the previous section, the two follow-up controls
(matched regime, random partitions) in their own sections below. Per-run values are in
`cache/shuffle_control_results.json`, `cache/matched_regime.json` and `cache/partition_control.json`.

### What we are allowed to claim

**1. The connectome's topology admits a stable, low-rate, input-driven regime at the calibrated inhibitory gain
that degree-, weight- and sign-matched random graphs do not (Control 1a).** Under an identical 13-point w_scale
scan at g_inh 2.5, each graph centred on its own anchor, the real graph passes all three target criteria at two
points (2.6 and 4.4 Hz; sensory rate 85% and 77% of reference; 0 Hz with the drive removed). None of the four
shuffled graphs (2 full, 2 downstream-only) passes at any point: they stay silent (<= 0.75 Hz) up to 0.56x their
anchor and jump to 24-27 Hz at 1x, 41-68 Hz above that, with the sensory neurons pushed to 175-477% of reference
instead of being held near it. Limit of the statement: the grid spacing is 1.78x in w_scale, so a shuffled window
narrower than one grid step is not excluded; the real graph's window spans at least one full step on the same
relative grid. Control 1b (recalibrating each shuffled graph to a matched rate and re-running transmission) was
therefore not applicable and was not run. The full- and downstream-shuffle transmission and dynamics differences
in the main table remain comparisons between a 4-5 Hz and a ~35 Hz network: they are regime differences, and the
matched-rate comparison does not exist at this g_inh.

**2. The pre-registered prediction for downstream-only was wrong, and Control 1a explains why.** Downstream-only
was expected to change little because hop 2 and hop 3 carry no measurable audio information. It changes as much
as the full shuffle, and it too has no working window: the structure beyond hop 1 carries no audio information
but is what makes the low-rate input-driven state possible.

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
The output depends on the real connectome through its topology holding the network in a low-rate, input-driven
state that matched random graphs do not reach at the same inhibitory gain. It does not, at the level measured
here, depend on a frequency-specific sensory-to-descending wiring: the apparent selectivity is type structure,
and randomizing which afferent group contacts which target does not reduce transmission.

Limitations: 5 sections (the 2 kyuchek sections overlap by 7 s), 2 seeds per shuffle condition, 2 shuffled
graphs per condition in Control 1a, a 1.78x w_scale grid, one calibration, one g_inh.
