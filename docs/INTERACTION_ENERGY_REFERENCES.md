# Interaction-energy weights: sources and justification (correction A5)

Table 1 of the manuscript assigns a fixed energy to each interaction type. These weights
generate the 20 residue-level features, so a reviewer will ask where they come from.

## What the literature supports

| Interaction | Manuscript range | Mean of range | Value used | Literature |
|---|---|---|---|---|
| Hydrogen bond | 2.4–9.6 kcal/mol | 6.0 | **5.0** | Jeffrey (1991): 2–10 kcal/mol overall; "moderate, mostly electrostatic" bonds at 2.5–3.2 Å span 4–15 kcal/mol |
| Cation–π | 2–5 kcal/mol | **3.5** | **3.5** | Dougherty, *Acc. Chem. Res.* 2013, 46, 885–893: cation–π enhances binding by 2–5 kcal/mol in biological systems |
| Hydrophobic | 1–2 kcal/mol | **1.5** | **1.5** | Chothia (1974) ~25 cal/mol per Å² buried; Vallone et al., *PNAS* 1998, 95, 6103: −15 ± 1.2 cal/mol per Å² at subunit interfaces |
| Ionic / salt bridge | 1.5–5 kcal/mol | 3.25 | **4.0** | Kumar & Nussinov, *J. Mol. Biol.* 1999, 293, 1241: majority of salt bridges stabilising, network salt bridges ≈ −5.0 kcal/mol each, halophilic average ≈ −3.0 kcal/mol |
| Water bridge | 0.5–3 kcal/mol | 1.75 | **2.0** | Follows hydrogen-bonding energetics with relaxed geometry (Jeffrey 1991) |

## On the "average of the highest and lowest" rule

Your recollection holds exactly for **cation–π** (3.5) and **hydrophobic contacts** (1.5),
and is close for **water bridges** (1.75 against 2.0 used). It does **not** reproduce the
hydrogen-bond weight (the rule gives 6.0, the table uses 5.0) or the ionic weight (the rule
gives 3.25, the table uses 4.0).

Rather than change the features, which would mean regenerating every downstream result,
the defensible move is to cite each value to a source that supports that specific number.
Both non-conforming values sit comfortably inside their literature bands: 5.0 kcal/mol is
the textbook figure for a typical protein hydrogen bond and sits within Jeffrey's
4–15 kcal/mol moderate class, and 4.0 kcal/mol falls between the −3.0 and −5.0 kcal/mol
per-salt-bridge figures reported by Kumar and Nussinov. The methods text should state the
range, the citation and the adopted value for each type, and drop any claim that the values
are arithmetic means.

## The point that actually defuses the objection

The absolute scale of these weights **cannot** affect the model, and their ratios affect it
only weakly. Each residue feature is a weighted sum of interaction counts,

    f_residue = Σ_type ( count_type × E_type )

and every feature is standardised before principal-component analysis. For a residue
dominated by one interaction type the weight cancels exactly under standardisation, since
(cE − mean(cE)) / sd(cE) = (c − mean(c)) / sd(c) for any constant E. Only the *ratios*
between interaction types survive, and only where a residue's total mixes types. Choosing
5.0 rather than 6.0 kcal/mol for a hydrogen bond is therefore not a modelling decision of
any consequence; it is a labelling convention.

The methods section should say this plainly. It converts a question a reviewer would press
("justify these five constants") into one already answered ("they are conventional
mid-range values, and the model is invariant to their scale").

## References to add

1. Jeffrey, G. A. *An Introduction to Hydrogen Bonding*; Oxford University Press: New York, 1997.
2. Jeffrey, G. A.; Saenger, W. *Hydrogen Bonding in Biological Structures*; Springer-Verlag: Berlin, 1991.
3. Dougherty, D. A. The Cation–π Interaction. *Acc. Chem. Res.* **2013**, *46*, 885–893.
4. Gallivan, J. P.; Dougherty, D. A. Cation–π Interactions in Structural Biology. *Proc. Natl. Acad. Sci. U.S.A.* **1999**, *96*, 9459–9464.
5. Chothia, C. Hydrophobic Bonding and Accessible Surface Area in Proteins. *Nature* **1974**, *248*, 338–339.
6. Vallone, B.; Miele, A. E.; Vecchini, P.; Chiancone, E.; Brunori, M. Free Energy of Burying Hydrophobic Residues in the Interface between Protein Subunits. *Proc. Natl. Acad. Sci. U.S.A.* **1998**, *95*, 6103–6107.
7. Kumar, S.; Nussinov, R. Salt Bridge Stability in Monomeric Proteins. *J. Mol. Biol.* **1999**, *293*, 1241–1255.
8. Hendsch, Z. S.; Tidor, B. Do Salt Bridges Stabilize Proteins? A Continuum Electrostatic Analysis. *Protein Sci.* **1994**, *3*, 211–226.

# Citation for the "quality over quantity" claim (correction A6)

`Hybrid model.docx` contains "a finding aligned with previous reports highlighting the
importance of quality over quantity in deep learning systems [Bring ref]". The claim being
made is that a moderate, well-regularised network beat larger ones on a small tabular
dataset. Two references support exactly that, both directly on point for tabular data:

9. Grinsztajn, L.; Oyallon, E.; Varoquaux, G. Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data? *Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track*, 2022.
10. Shwartz-Ziv, R.; Armon, A. Tabular Data: Deep Learning Is Not All You Need. *Inf. Fusion* **2022**, *81*, 84–90.

These are a better fit than a generic model-scaling reference, because both benchmark deep
networks against classical models on tabular data of the size used here and reach the same
conclusion this study reaches independently.
