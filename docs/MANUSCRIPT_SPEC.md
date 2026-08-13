# Specification for the final merged manuscript

Given by the author on 13 Aug 2026. This governs the **final** deliverable, which merges
the three original documents with the V2 benchmark work into one paper. Not to be built
until the author asks.

## 1. Scope of the merge

Combine into a single, continuous manuscript:

| Source | Role |
|---|---|
| `Modelling_manuscript110425_Mod_updated230425_mod.docx` | the backbone: intro, methods, results, references |
| `Hybrid model.docx` | DNN and hybrid methods/results, to be folded into the model sections |
| `Interpretation of SHAP Summary Plot.docx` | interpretability, to be folded into results/discussion |
| V2 benchmark work (this repo) | reproduction, external benchmarks, DFT, statistics |

The result must read as one paper written by one hand, not four documents stapled
together. Ideas must flow: each section should set up the next.

## 2. House style: ACS *Journal of Chemical Information and Modeling*

- ACS section structure and tone throughout.
- ACS citation style, numbered.
- **Add proper references to every sentence that needs one.** The current drafts assert
  several things without support (for example the interaction energy values in Table 1,
  the claim about architectural complexity, the comparisons to prior QSAR work).
- Correct every error found; the catalogue is `tables/TableS3_manuscript_corrections.csv`
  (13 items, including the shifted MSE columns and the mislabelled Nu-SVR CV MSE).

## 3. No AI tells

- **No em dashes anywhere in prose.** Use commas, semicolons, parentheses or a full stop.
- **No underscores in any visible text.** Descriptor and metric names must be typeset as
  English, not as code identifiers: `q_mulliken_range` becomes "Mulliken charge range",
  `E_HOMO_eV` becomes "E(HOMO), eV", `W_new` becomes "W new", `HOMO_LUMO_gap_eV` becomes
  "HOMO-LUMO gap". This applies to figure axis labels, legends, table headers and body
  text alike.
- Avoid the other common tells: no "delve", no "it is worth noting that", no tricolon
  padding, no sentence openings with "Notably," or "Importantly," used as filler, no
  bulleted lists where prose belongs.

## 4. Figures

- **Black and white only.** No colour anywhere. Series are separated by greyscale fill
  level, hatching, marker shape and line style, never by hue. Check every figure prints
  legibly on a monochrome laser printer.
- **Consolidate.** If the figure count is high, merge related figures into one
  multi-panel figure with panels labelled A, B, C. Aim for a main-text figure count a
  journal would accept (roughly 5 to 7), moving the rest to supplementary.
- **Captions are for reviewers, not for the author.** No first- or second-person, no
  "as published", no "the manuscript's", no reference to revisions, benchmarks-versus-
  original framing, or anything implying a conversation. Each caption must stand alone
  and describe only what the panel shows.
- Keep 600 dpi, TIFF and PDF, Arial/Helvetica.

## 5. Supplementary information

Produce a separate supplementary `.docx` holding everything not appropriate for the main
text: the full per-dataset metric tables, the statistical validation table, the paired
comparison table, the dataset inventory and provenance, the reproduction table, the
corrections log, and any figure moved out of the main text. **The main manuscript must
cite the supplementary material explicitly** at each point where it applies ("Table S1",
"Figure S2", and so on).

## 6. Deletions require permission

Where content in the three original documents is weak or wrong, list it and **ask the
author before removing it**. Do not delete unilaterally. Current candidates, to be put to
the author when the final build starts:

1. The duplicated DNN rows in `Hybrid model.docx` Table 1 (DNN-1 = DNN-3, DNN-2 = DNN-4).
2. The narrative DNN figures in `Hybrid model.docx` that appear in no table
   (R² train 0.8044 / 0.9532 and CV 0.7373 / 0.9323).
3. The claim that the hybrid ensemble outperforms Nu-SVR, which the statistics do not
   support.
4. The SHAP paragraph attributing importance to protein structural confidence, which
   conflates target identity with structure quality.
5. The "[Bring ref]" placeholder in `Hybrid model.docx`.
6. Table 1's tabulated interaction energies if no primary source can be attached.

## 7. Numbers

Every quoted value must continue to be read from `results/` at build time so the text,
tables and figures cannot drift apart. Nothing hard-coded.
