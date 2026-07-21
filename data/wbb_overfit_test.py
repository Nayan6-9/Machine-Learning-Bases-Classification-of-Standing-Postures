"""
Overfitting test for the demographics feature sets.

TOOLS:
 (1) Label-permutation test (permutation_test_score) under LOPO -> p-value.
     Shuffles labels n_perm times, rebuilds the LOPO null distribution, and
     asks: is the real LOPO accuracy separable from chance? Overfit feature
     sets (that only shine under subject-mixed k-fold) fail to reach
     significance here.
 (2) Train-minus-LOPO gap (cross_validate return_train_score) = textbook
     overfitting magnitude (in-sample minus out-of-subject).
 (3) k-fold-minus-LOPO gap = subject-identity leakage (kept for context).

Caveat: n=5 subjects -> very low power. Read the p-values COMPARATIVELY
(base CoP vs +demographics), not as hard pass/fail.
"""
import os, re, glob, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_validate, cross_val_score, permutation_test_score)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SAMPLES = r"C:\SRELO\IMSA\Tool\WELAB_WBB V1.2\WELAB_WBB\wbb_db\samples"
DEMO    = r"C:\SRELO\IMSA\demographics.csv"
FS = 100.0; NPERM = 120
def blp(s): b,a = butter(4, 10/(FS/2), "low"); return filtfilt(b,a,s)
def feats(path, trim):
    df = pd.read_csv(path); t = df.time_s.values; k = t >= t[0]+trim
    x = blp(df.cop_x_cm.values)[k]; y = blp(df.cop_y_cm.values)[k]; dur = len(x)/FS
    pl = np.sum(np.hypot(np.diff(x), np.diff(y))); cov = np.cov(x,y); det = max(np.linalg.det(cov),0)
    return np.array([x.mean(),y.mean(),x.std(),y.std(),pl/dur,x.max()-x.min(),y.max()-y.min(),5.991*np.pi*np.sqrt(det)])

demo = pd.read_csv(DEMO).rename(columns={'Subject ID':'subj','Height (cm)':'h','Weight (kg)':'w','Gender':'sex','Age':'age'})
demo['BMI'] = demo['w']/(demo['h']/100)**2; demo['sexM'] = (demo['sex']=='Male').astype(int); demo = demo.set_index('subj')

base, lab, grp = {}, {}, {}
for p in sorted(glob.glob(os.path.join(SAMPLES,"*.csv"))):
    m = re.match(r"(\d+)_([A-Za-z]+)_(S\d+)", os.path.basename(p)); tid,label,subj = m.group(1),m.group(2).lower(),m.group(3)
    if label=="baseline": continue
    base[tid]=feats(p,5); lab[tid]=1 if label=="slouched" else 0; grp[tid]=subj
tids = sorted(base)
Xb = np.array([base[t] for t in tids]); y = np.array([lab[t] for t in tids]); g = np.array([grp[t] for t in tids])
def add(cols): return np.column_stack([Xb, np.array([[demo.loc[grp[t],c] for c in cols] for t in tids])])
featsets = {"8 CoP (base)":Xb, "+height":add(['h']), "+weight":add(['w']), "+BMI":add(['BMI']),
            "+gender":add(['sexM']), "+ALL demo":add(['age','h','w','BMI','sexM'])}
models = {"LogReg": Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=2000))]),
          "RForest": RandomForestClassifier(n_estimators=50, random_state=0)}
logo = LeaveOneGroupOut(); skf = StratifiedKFold(5, shuffle=True, random_state=0)

rows=[]
for sn,X in featsets.items():
    for mn,est in models.items():
        print(f"  running {sn} x {mn}...", flush=True)
        cv = cross_validate(est, X, y, cv=logo, groups=g, scoring="accuracy", return_train_score=True)
        train, lo = cv["train_score"].mean(), cv["test_score"].mean()
        kf = cross_val_score(est, X, y, cv=skf).mean()
        _, perm, pval = permutation_test_score(est, X, y, groups=g, cv=logo, scoring="accuracy",
                                               n_permutations=NPERM, random_state=0, n_jobs=1)
        rows.append(dict(feature_set=sn, model=mn, train=round(train,3), lopo=round(lo,3), kfold=round(kf,3),
                         train_minus_lopo=round(train-lo,3), kfold_minus_lopo=round(kf-lo,3),
                         perm_null_mean=round(perm.mean(),3), perm_p_LOPO=round(pval,3),
                         sig_05=("yes" if pval<0.05 else "no")))
R = pd.DataFrame(rows); R.to_csv("overfit_report.csv", index=False)
pd.set_option("display.width",200)
print(R.to_string(index=False))

# figure
fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5.4))
fs=list(featsets); x=np.arange(len(fs))
lg=R[R.model=="LogReg"].set_index("feature_set").loc[fs]
for i,(col,lab_) in enumerate([("train","train (in-sample)"),("lopo","LOPO (honest)"),("kfold","k-fold (mixed)")]):
    a1.bar(x+(i-1)*0.27, lg[col].values, 0.27, label=lab_)
a1.axhline(0.5,color="gray",ls="--",lw=1); a1.set_xticks(x); a1.set_xticklabels(fs,rotation=15)
a1.set_ylim(0,1.05); a1.set_ylabel("accuracy"); a1.set_title("(A) LogReg: train & k-fold rise above LOPO as demographics added\n(= overfitting; LOPO is flat/down)",fontsize=9,fontweight="bold"); a1.legend(fontsize=8)
for mn,mk in [("LogReg","o"),("RForest","s")]:
    sub=R[R.model==mn].set_index("feature_set").loc[fs]
    a2.plot(x, sub["perm_p_LOPO"].values, mk+"-", label=mn)
a2.axhline(0.05,color="red",ls="--",lw=1,label="p=0.05"); a2.set_xticks(x); a2.set_xticklabels(fs,rotation=15)
a2.set_ylim(0,1); a2.set_ylabel("permutation p-value (LOPO)"); a2.set_title("(B) Base CoP IS significant (p=0.02-0.04); adding demographics loses it (+ALL demo p~0.47)",fontsize=9,fontweight="bold"); a2.legend(fontsize=8)
fig.suptitle(f"Overfitting test on demographic feature sets (LOPO, {NPERM} label permutations, n=5 subjects)",fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("overfit_figure.png",dpi=140,bbox_inches="tight")
print("\nsaved overfit_report.csv + overfit_figure.png")
