# Two defects found in the study matrices

Both were found by asking whether a descriptor has a property its own definition
requires, not by looking at model performance. Neither is visible in a performance
number: a corrupted block degrades a model quietly, and the degradation is then
interpreted as a finding. Both are reported here because they change what earlier
versions of this work could claim.

## 1. The stored MACCS block does not describe these molecules

A MACCS key is a function of the molecular graph, so two rows holding the same
molecule must hold bit-for-bit identical keys.

| check | stored block | recomputed |
| --- | --- | --- |
| same-structure row pairs with identical keys | 0 of 38 | 38 of 38 |
| largest disagreement within one structure | 52 of 167 bits | 0 |
| bits set for one 444.488 Da anthraquinone | 24 against 56 | identical |
| rows agreeing with a correct recomputation | 2 of 122 | by construction |
| is the stored block a permutation of the correct one | no | n/a |

The block is therefore neither aligned nor merely reordered. `src/maccs_fix.py`
recovers a verified block for all 122 rows from the same mol2 files every other
descriptor set was computed from, and refuses to return unless within-structure
identity holds.

**Consequence.** The 89-descriptor core excludes this block and is unaffected. The
272-descriptor set contains it, so an ablation showing that 89 beats 272 was a
comparison against 167 columns of noise and could not support a conclusion about
fingerprints. Repeated against the recomputed block, the two are indistinguishable,
and the correct reading is redundancy rather than interference.

## 2. Three of the five interaction types never reached the residue descriptors

The per-residue descriptors are defined as

```
F(r) = (1/N) * sum over frames f, sum over interaction types k of  e(k) * n(r,k,f)
```

with five types k: hydrogen bond, cation pi, hydrophobic, ionic and water bridge.
Only hydrophobic and ionic ever contributed. The extraction code evaluated both
hydrogen-bond angles from a zero-length vector, so each returned undefined and every
cutoff comparison was false; cation pi had a distance cutoff defined but no detection
code.

This is provable from the stored matrix without repeating any simulation, because the
two surviving types are residue selective while the three suppressed ones are not.
Hydrophobic contacts are counted only for the eight apolar residues, ionic only for
the four charged ones, while hydrogen bonds and water bridges are counted for any
residue bearing nitrogen, oxygen or sulphur, which is all twenty.

| observation | value |
| --- | --- |
| residue descriptors identically zero in all 122 systems | 8 |
| which residues | ASN, CYS, GLN, GLY, HIS, PRO, SER, THR |
| the non-zero set equals the hydrophobic and charged sets | exactly |
| energies applied | hydrophobic 1.5, ionic 4.0 kcal per mole |
| energies never applied | hydrogen bond 5.0, cation pi 3.5, water bridge 2.0 |

`src/residue_audit.py` asserts this and will fail if the pattern is ever otherwise.
A corrected extraction is provided in `src/feature_extraction/` for future use;
recomputing the block for this dataset needs trajectories not available to us.

**Consequence.** The per-residue block encodes hydrophobic and ionic contact
frequency weighted by a constant. Reported results are the performance of that
description, not of the five-type scheme.

## Why 30 of the 89 columns are constant

| group | columns |
| --- | --- |
| residue descriptors no interaction type reached | 8 |
| interaction energy terms for water and metal coordinates absent from these systems | 21 |
| array-shape constant | 1 |

The representation carries 59 columns with variance, and that is the number to
compare against the dimensions of the alternatives. See `data/descriptors/`.
