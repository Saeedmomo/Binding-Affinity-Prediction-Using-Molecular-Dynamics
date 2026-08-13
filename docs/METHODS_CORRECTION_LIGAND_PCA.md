# Corrected Methods text for the ligand-PCA descriptor block

## What the manuscript currently says (section 2.3)

> The key metrics extracted included Explained Variance Ratios (PC1, PC2, PC3) which
> represented the proportion of the total variance in the ligand's atomic displacements
> that is captured by the first three principal components, **PCA components shape that
> referred to the structural deformations associated with the first and second principal
> components and provided insights into the directions along which the ligand shows the
> most significant conformational variation**, and **Average Structure Shape that
> symbolized the mean conformation of the ligand across all frames in the trajectory**.

## Why this cannot stand

The two bolded descriptions do not match the columns in the descriptor matrix. The
extraction script recorded `pca.components_.shape` and `average_structure.shape`, which
are the dimensions of those arrays rather than any quantity computed from them:

| Column | Description given | What the column contains |
|---|---|---|
| `PCA-components-shape1` | structural deformation along PC1 | the constant 3, for all 122 rows: the number of components retained |
| `PCA-components-shape2` | structural deformation along PC2 | three times the ligand atom count |
| `Average-structure-shape` | the mean conformation | the ligand atom count, integers 24 to 72 |

A referee who opens the feature table sees three integer-valued columns, one of them
constant. The explained-variance ratios are unaffected and were always correct.

These columns are retained in the released matrices so that the published results remain
exactly reproducible, so the text must describe them for what they are.

## Suggested replacement text

> To characterise the internal conformational dynamics of the ligands, principal
> component analysis was performed on ligand-only trajectories extracted from the
> simulation. Three explained-variance ratios were retained, giving the proportion of
> the total variance in atomic displacement captured by each of the first three
> principal components. Three further columns record the dimensions of the analysis
> rather than the motion itself: the number of components retained, which is constant at
> three and is therefore removed by the zero-variance filter before modelling, and two
> columns proportional to the number of atoms in the ligand trajectory. The latter two
> act as ligand-size descriptors and are reported as such. The feature-extraction code
> released with this work additionally computes the root-mean-square atomic displacement
> along the first two principal components and the radius of gyration of the mean
> conformer, which describe the amplitude of the dominant motions directly; these were
> not available for the models reported here.

## Sentence to add to the limitations

> Two of the ligand-PCA columns scale with ligand size rather than with conformational
> amplitude, and correlate with pIC50 at r = 0.581 across the 122 systems. Their
> contribution should therefore be read as a size effect. Removing all three reduces the
> Nu-SVR test R2 on the molecular dynamics core from 0.798 to 0.706 while leaving the
> holdout R2 unchanged at 0.843, so the principal conclusions are unaffected.

## Second correction: an impossible PAE value

One protein row of the AlphaFold block reports `PAE Mean = 30.07` alongside
`PAE Q25 = 1.4`, `PAE Q50 = 1.9`, `PAE Q75 = 2.70` and `PAE Max = 30.5`. A mean cannot
approach the maximum when three quarters of the distribution lies below 2.70. The source
JSON for that structure (6N0D) gives a PAE mean of **3.07**, so a zero has been inserted
somewhere between extraction and the descriptor table.

The affected row is the TDP1 signature, which carries the seventeen most potent compounds
in the dataset, so the erroneous value is attached to the highest-activity subset. The
value should be corrected to 3.07 in any future revision of the matrix; it is left
unchanged in the released files so that the reported models remain reproducible, and is
recorded here.
