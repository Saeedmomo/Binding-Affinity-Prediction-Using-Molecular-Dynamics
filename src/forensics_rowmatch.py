import pandas as pd, numpy as np

j3 = pd.read_csv(r'D:\Chua files\fourth paper\JDes_output3.csv', low_memory=False)
j2 = pd.read_csv(r'D:\Chua files\MDS_analysis\data_model\JDes_output2.csv', low_memory=False)
cols = [c for c in j3.columns if c != 'Name']

A = j3[cols].to_numpy(dtype=float)   # 122 x 1444
B = j2[cols].to_numpy(dtype=float)

# use only columns with no NaN anywhere for matching
good = ~(np.isnan(A).any(0) | np.isnan(B).any(0))
Ag, Bg = A[:, good], B[:, good]
print("matching on", good.sum(), "complete columns")

# normalise per column so scale differences don't dominate
sd = np.nanstd(np.vstack([Ag, Bg]), axis=0)
sd[sd == 0] = 1.0
An, Bn = Ag / sd, Bg / sd

# nearest neighbour of each j3 row among j2 rows
D = np.sqrt(((An[:, None, :] - Bn[None, :, :]) ** 2).sum(-1))
nn = D.argmin(1)
dmin = D.min(1)
print("exact/near matches (dist<1e-6):", int((dmin < 1e-6).sum()), "/", len(dmin))
print("dist<1e-3:", int((dmin < 1e-3).sum()), " dist<1e-1:", int((dmin < 1e-1).sum()))
print("permutation (all j2 rows used exactly once):", len(set(nn.tolist())) == len(nn))
print("identity mapping:", np.array_equal(nn, np.arange(len(nn))))
print("dmin  min/med/max: %.4g %.4g %.4g" % (dmin.min(), np.median(dmin), dmin.max()))
print("first 20 nn:", nn[:20].tolist())
print("first 20 dmin:", np.round(dmin[:20], 4).tolist())

# also: identical-value fraction per row pair on the identity mapping
same = np.isclose(Ag, Bg, rtol=1e-9, atol=1e-9)
print("\nidentity-mapping cellwise identical fraction: %.4f" % same.mean())
print("rows fully identical under identity mapping:", int(same.all(1).sum()))

# how different are they overall? correlate column-wise
cc = []
for k in range(Ag.shape[1]):
    a, b = Ag[:, k], Bg[:, k]
    if a.std() > 0 and b.std() > 0:
        cc.append(np.corrcoef(a, b)[0, 1])
cc = np.array(cc)
print("\ncolumnwise corr(J3, J2) under identity order: mean=%.4f  median=%.4f  frac>0.99=%.3f"
      % (cc.mean(), np.median(cc), (cc > 0.99).mean()))
