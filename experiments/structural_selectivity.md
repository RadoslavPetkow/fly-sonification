## Structural analysis: band selectivity of the sensory input to the hop-1 motor neurons (no simulation)

**Prediction, stated before computing:** if the real connectome gives descending neurons band-selective auditory input, the real mean normalized entropy of each hop-1 motor neuron's sensory input across the 8 band groups is LOWER than under sensory-only shuffles (selectivity = 1 - entropy and the strongest group's share HIGHER).

Method: `experiments/structural_selectivity.py`. Input weight = summed |synapses| from each sensory group. Null = sensory-only re-targeting (the procedure of `shuffle_graph.py`): the 2 matrices built for the shuffle control, and an ensemble of 1000 further seeds of the same procedure for the interval and the permutation test. Because the sensory edges' target in-degrees are preserved, the same 56 neurons keep the same number of sensory contacts; only which group contacts them changes.

56 voices; 41 have >= 2 sensory contacts (a single contact is maximally selective in real and shuffled alike).

| subset | measure | real | built seeds | shuffled mean (sd) | real - shuffled [95% over seeds] | p one-sided (predicted) | p two-sided | effect (SD) |
|---|---|---|---|---|---|---|---|---|
| all 56 | entropy | 0.332 | 0.443, 0.453 | 0.453 (0.012) | -0.122 [-0.143, -0.098] | 0.000999 | 0.000999 | -10.46 |
| all 56 | selectivity | 0.668 | 0.557, 0.547 | 0.547 (0.012) | +0.122 [+0.098, +0.143] | 0.000999 | 0.000999 | +10.46 |
| all 56 | max_share | 0.694 | 0.632, 0.611 | 0.619 (0.014) | +0.076 [+0.046, +0.103] | 0.000999 | 0.000999 | +5.25 |
| >= 2 contacts (41) | entropy | 0.453 | 0.605, 0.619 | 0.619 (0.016) | -0.166 [-0.196, -0.133] | 0.000999 | 0.000999 | -10.46 |
| >= 2 contacts (41) | selectivity | 0.547 | 0.395, 0.381 | 0.381 (0.016) | +0.166 [+0.133, +0.196] | 0.000999 | 0.000999 | +10.46 |
| >= 2 contacts (41) | max_share | 0.583 | 0.498, 0.468 | 0.479 (0.020) | +0.103 [+0.063, +0.141] | 0.000999 | 0.000999 | +5.25 |

Per neuron: 17 of 56 voices exceed the 95th percentile of their own shuffle distribution (chance 2.8); 15 fall below the 5th.

| voice family | members | selectivity | shuffled mean | p one-sided | strongest group (share) |
|---|---|---|---|---|---|
| DNb | 1 | 1.000 | 0.362 | 0.000999 | 6 (100.0%) |
| DNc | 2 | 0.672 | 0.638 | 0.407 | 1 (57.1%) |
| DNg | 22 | 0.151 | 0.065 | 0.00899 | 0 (32.9%) |
| DNge | 17 | 0.042 | 0.068 | 0.824 | 6 (21.9%) |
| DNp | 12 | 0.122 | 0.073 | 0.0689 | 1 (35.2%) |
| DNpe | 1 | 1.000 | 1.000 | 1 | 2 (100.0%) |
| pIP | 1 | 1.000 | 1.000 | 1 | 2 (100.0%) |

Render comparison (families whose note beat its joint null in the chromatic family renders): 
- Gymnopedie_No_1_clip_range_31s_family_chromatic: DNg band 6 (wiring strongest group 0), DNge band 1 (wiring strongest group 6)
- kyuchek_clip_range_15s_family_chromatic: DNge band 5 (wiring strongest group 6), DNp band 7 (wiring strongest group 1)

![structural selectivity](../figures/structural_selectivity.png)

Caveat: the 8 groups are consecutive slices of the sensory neurons sorted by type, and JO types are partly defined by projection pattern, so group-selective wiring is partly type-selective by construction; which group is called which frequency band is this project's assignment, not measured tonotopy.

**Result:** the prediction HOLDS: real input entropy is lower than shuffled (all 56: entropy 0.332 vs 0.453, difference -0.122 [-0.143, -0.098], one-sided p 0.000999; voices with >= 2 contacts: difference -0.166, p 0.000999).
