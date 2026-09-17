## Control 2: is the structural selectivity just JO type structure? (no simulation)

**Prediction, stated before computing:** if the effect comes from JO type structure, the type-sorted partition shows a large real-vs-shuffled entropy gap and random partitions show little or none. If random partitions show a comparable gap, the effect is not about type.

Method: `experiments/partition_control.py`. 200 random partitions of the 86 sensory neurons into groups of the type partition's sizes [11, 11, 11, 11, 11, 11, 10, 10]; for every partition (and for the type-sorted one) its own ensemble of 200 sensory-only re-targetings with distinct seeds. Gap = real minus shuffled mean normalized entropy over the 56 hop-1 motor neurons.

| partition | real entropy | shuffled mean (sd) | gap | z |
|---|---|---|---|---|
| type-sorted | 0.3320 | 0.4534 (0.0121) | -0.1215 | -10.07 |
| random, mean of 200 | 0.4903 | 0.4628 | +0.0275 (sd 0.0149) | +2.43 |

Random-partition gaps: min -0.0121, 5th pct -0.0003, median +0.0301, 95th pct +0.0489, max +0.0573; 0% of random partitions are individually significant (one-sided p < 0.05).

**The type-sorted gap (-0.1215) lies at the 0.0th percentile of the random-partition gaps** (0.0% of random partitions are at least as selective); it is -4.42x the random-partition mean.

![partition control](../figures/partition_control.png)

**Result:** The type-sorted gap is more extreme than every random-partition gap, and random partitions show NO selectivity at all: their mean gap is +0.0275, i.e. with type-blind groups the real input is slightly LESS concentrated than shuffled. The structural selectivity is entirely JO TYPE structure.
