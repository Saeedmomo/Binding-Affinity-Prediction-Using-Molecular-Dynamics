import pandas as pd, numpy as np, joblib, os, sys, json

files = {
 'JDes_output3'          : r'D:\Chua files\fourth paper\JDes_output3.csv',
 'JDes_output2'          : r'D:\Chua files\MDS_analysis\data_model\JDes_output2.csv',
 'JDes_output2_imputed'  : r'D:\Chua files\MDS_analysis\data_model\JDes_output2_imputed.csv',
 '5_imputed'             : r'D:\Chua files\MDS_analysis\data_model\5_imputed.csv',
 '2_cleaned'             : r'D:\Chua files\MDS_analysis\data_model\2_cleaned.csv',
 '2'                     : r'D:\Chua files\MDS_analysis\data_model\2.csv',
 '1'                     : r'D:\Chua files\MDS_analysis\data_model\1.csv',
 '3'                     : r'D:\Chua files\MDS_analysis\data_model\3.csv',
 '4'                     : r'D:\Chua files\MDS_analysis\data_model\4.csv',
 'merged_desc_pic50'     : r'D:\Chua files\Said_mol2_files\merged_descriptors_with_pic50.csv',
 'merged_desc'           : r'D:\Chua files\Said_mol2_files\merged_descriptors.csv',
 'Predition_set'         : r'D:\Chua files\MDS_analysis\data_model\Predition_set.csv',
 'Predition_set2'        : r'D:\Chua files\MDS_analysis\data_model\Predition_set2.csv',
 'predicted_pic50_res3'  : r'D:\Chua files\MDS_analysis\data_model\predicted_pic50_results3.csv',
}

print("="*110)
print("CSV SHAPE / COLUMN FORENSICS")
print("="*110)
for name, p in files.items():
    if not os.path.exists(p):
        print(f"{name:24s} MISSING")
        continue
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        print(f"{name:24s} ERROR {e}")
        continue
    numcols = df.select_dtypes(include=[np.number]).shape[1]
    has_t = [c for c in df.columns if c.strip().upper() in ('PIC50','PIC_50','PIC50_','P_IC50')]
    nan_cols = int((df.isna().sum() > 0).sum())
    print(f"{name:24s} rows={df.shape[0]:6d} cols={df.shape[1]:6d} numeric={numcols:6d} "
          f"target={has_t} cols_with_nan={nan_cols}")
    print(f"{'':24s} first8={list(df.columns[:8])}")
    print(f"{'':24s} last6 ={list(df.columns[-6:])}")

print()
print("="*110)
print("PICKLED ARTEFACT FORENSICS  (padel_model)")
print("="*110)
pm = r'D:\Chua files\MDS_analysis\data_model\padel_model'
for f in ['scaler.pkl','pca.pkl','meta_model_ridge.pkl']:
    p = os.path.join(pm, f)
    try:
        o = joblib.load(p)
        print(f"\n--- {f}: {type(o)}")
        for a in ['n_features_in_','n_components_','n_samples_seen_','coef_','intercept_','alpha','mean_']:
            if hasattr(o, a):
                v = getattr(o, a)
                if a == 'mean_':
                    print(f"    mean_ shape={np.shape(v)} first5={np.ravel(v)[:5]}")
                elif a == 'coef_':
                    print(f"    coef_={np.ravel(v)}")
                else:
                    print(f"    {a}={v if not hasattr(v,'shape') else np.shape(v)}")
        if hasattr(o, 'explained_variance_ratio_'):
            evr = o.explained_variance_ratio_
            print(f"    n_comp={len(evr)}  cum_var={evr.sum():.4f}  first5evr={evr[:5]}")
        if hasattr(o, 'feature_names_in_'):
            print(f"    feature_names_in_[:10]={list(o.feature_names_in_[:10])}")
    except Exception as e:
        print(f"\n--- {f}: LOAD ERROR {e}")

print()
print("="*110)
print("GITHUB REPO ridge pkl")
print("="*110)
try:
    o = joblib.load(r'D:\Chua files\fourth paper\benchmark_work\github_repo\meta_model_ridge.pkl')
    print(type(o), 'coef_=', getattr(o,'coef_',None), 'intercept_=', getattr(o,'intercept_',None),
          'n_features_in_=', getattr(o,'n_features_in_',None))
except Exception as e:
    print('ERR', e)
