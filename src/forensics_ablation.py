"""How much of the published performance is explained by target identity alone?
The AlphaFold pLDDT/PAE block is constant within a protein, so it acts as a
target-identity indicator. PC1 (mean |SHAP| = 0.807, the dominant component in
the manuscript's SHAP analysis) is loaded almost entirely on pLDDT features."""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import NuSVR

d2  = pd.read_csv(r'D:\Chua files\MDS_analysis\data_model\2.csv', low_memory=False)
lig = pd.read_csv(r'D:\Chua files\fourth paper\benchmark_work\data\ligands_122_annotated.csv')
y   = d2['PIC50'].to_numpy()
af  = [c for c in d2.columns if 'pLDDT' in c or 'PAE' in c]
grp = lig['af_group'].to_numpy()

print("="*84)
print("BASELINE 1 : predict PIC50 from AlphaFold-group identity ALONE (8 dummies)")
print("="*84)
D = pd.get_dummies(pd.Series(grp), prefix='g').to_numpy(float)
kf = KFold(5, shuffle=True, random_state=42)
pred_cv = cross_val_predict(LinearRegression(), D, y, cv=kf)
print(f"  in-sample R2 (group means) : {r2_score(y, LinearRegression().fit(D,y).predict(D)):.4f}")
print(f"  5-fold CV R2              : {r2_score(y, pred_cv):.4f}")
print("  group mean PIC50:")
for gg in sorted(set(grp)):
    m = grp == gg
    print(f"    group {gg} (n={m.sum():3d})  mean={y[m].mean():.3f}  sd={y[m].std():.3f}"
          f"  range=[{y[m].min():.2f}, {y[m].max():.2f}]")

print()
print("="*84)
print("BASELINE 2 : the 14 AlphaFold columns only, through the published Nu-SVR recipe")
print("="*84)
idx = np.arange(122)
tmp_i, uns_i = train_test_split(idx, test_size=0.1, random_state=1)
tr_i, te_i   = train_test_split(tmp_i, test_size=0.2222, random_state=1)

def nusvr_eval(X, label, ncomp=17):
    ncomp = min(ncomp, X.shape[1], len(tr_i))
    pipe = Pipeline([('sc', StandardScaler()), ('pca', PCA(n_components=ncomp)),
                     ('m', NuSVR(C=1.0, nu=0.7, kernel='rbf'))])
    pipe.fit(X[tr_i], y[tr_i])
    out = dict(train=r2_score(y[tr_i], pipe.predict(X[tr_i])),
               test=r2_score(y[te_i], pipe.predict(X[te_i])),
               holdout=r2_score(y[uns_i], pipe.predict(X[uns_i])))
    print(f"  {label:44s} n_feat={X.shape[1]:5d} ncomp={ncomp:3d} "
          f"train={out['train']:.4f} test={out['test']:.4f} holdout={out['holdout']:.4f}")
    return out

feat_all = [c for c in d2.columns if c != 'PIC50']
maccs    = [c for c in d2.columns if c.startswith('MACCS')]
nonaf    = [c for c in feat_all if c not in af]
nonaf_nomaccs = [c for c in nonaf if c not in maccs]

nusvr_eval(d2[feat_all].to_numpy(float), 'ALL 272 features (published recipe)')
nusvr_eval(d2[af].to_numpy(float),       'AlphaFold block ONLY (14 cols)')
nusvr_eval(d2[nonaf].to_numpy(float),    'everything EXCEPT AlphaFold (258 cols)')
nusvr_eval(d2[maccs].to_numpy(float),    'MACCS fingerprints ONLY (167 cols)')
nusvr_eval(d2[nonaf_nomaccs].to_numpy(float), 'MD+energy+PCA+residue only (91 cols)')

print()
print("="*84)
print("BASELINE 3 : same, but with a STRUCTURE-DISJOINT split (no shared SMILES)")
print("="*84)
from sklearn.model_selection import GroupShuffleSplit
smi = lig['smiles'].to_numpy()
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=1)
tmp_i2, uns_i2 = next(gss1.split(idx, y, groups=smi))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2222, random_state=1)
a, b = next(gss2.split(tmp_i2, y[tmp_i2], groups=smi[tmp_i2]))
tr_i, te_i, uns_i = tmp_i2[a], tmp_i2[b], uns_i2
print(f"  sizes train={len(tr_i)} test={len(te_i)} holdout={len(uns_i)}")
print(f"  shared structures train&test={len(set(smi[tr_i])&set(smi[te_i]))}"
      f"  train&holdout={len(set(smi[tr_i])&set(smi[uns_i]))}")
nusvr_eval(d2[feat_all].to_numpy(float), 'ALL 272 features, structure-disjoint')
nusvr_eval(d2[af].to_numpy(float),       'AlphaFold block ONLY, structure-disjoint')
