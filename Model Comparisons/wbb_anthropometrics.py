"""
Anthropometrics: (1) height-normalization of CoP features (the LEGITIMATE use
of body-size data) and (2) demographics-as-raw-features overfitting demo,
using the real demographics.csv. LOPO vs subject-mixed k-fold. Chance=0.50.
"""
import os, re, glob, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SAMPLES = r"C:\SRELO\IMSA\Tool\WELAB_WBB V1.2\WELAB_WBB\wbb_db\samples"
DEMO    = r"C:\SRELO\IMSA\demographics.csv"
FS = 100.0
def blp(s): b,a = butter(4, 10/(FS/2), "low"); return filtfilt(b,a,s)
FEATS = ["mean_ml","mean_ap","rms_ml","rms_ap","mean_velocity","sway_amp_ml","sway_amp_ap","ellipse_area_95"]
HEXP  = np.array([1,1,1,1,1,1,1,2])   # divide each feature by height**exp (area ~ height^2)

def feats(path, trim):
    df = pd.read_csv(path); t = df.time_s.values; k = t >= t[0]+trim
    x = blp(df.cop_x_cm.values)[k]; y = blp(df.cop_y_cm.values)[k]; dur = len(x)/FS
    pl = np.sum(np.hypot(np.diff(x), np.diff(y))); cov = np.cov(x,y); det = max(np.linalg.det(cov),0)
    return np.array([x.mean(),y.mean(),x.std(),y.std(),pl/dur,x.max()-x.min(),y.max()-y.min(),5.991*np.pi*np.sqrt(det)])

demo = pd.read_csv(DEMO).rename(columns={'Subject ID':'subj','Height (cm)':'h','Weight (kg)':'w','Gender':'sex','Age':'age'})
demo['BMI'] = demo['w']/(demo['h']/100)**2
demo['sexM'] = (demo['sex']=='Male').astype(int)
demo = demo.set_index('subj')

base, lab, grp, bref = {}, {}, {}, {}
for p in sorted(glob.glob(os.path.join(SAMPLES,"*.csv"))):
    m = re.match(r"(\d+)_([A-Za-z]+)_(S\d+)", os.path.basename(p)); tid,label,subj = m.group(1),m.group(2).lower(),m.group(3)
    if label=="baseline": bref[subj]=feats(p,1); continue
    base[tid]=feats(p,5); lab[tid]=1 if label=="slouched" else 0; grp[tid]=subj
tids = sorted(base)
Xb = np.array([base[t] for t in tids]); y = np.array([lab[t] for t in tids]); g = np.array([grp[t] for t in tids])
H  = np.array([demo.loc[grp[t],'h'] for t in tids])[:,None]
Xh    = Xb/(H**HEXP)                                    # height-normalized
Xnorm = np.array([base[t]-bref[grp[t]] for t in tids])  # baseline-normalized
Xnh   = Xnorm/(H**HEXP)                                  # baseline + height

logo = LeaveOneGroupOut(); skf = StratifiedKFold(5, shuffle=True, random_state=0)
models = {"LogReg": Pipeline([("s",StandardScaler()),("c",LogisticRegression(max_iter=2000))]),
          "RForest": RandomForestClassifier(n_estimators=200, random_state=0)}
def lopo(X,est):
    yp = cross_val_predict(est,X,y,cv=logo,groups=g)
    return accuracy_score(y,yp), {s:round(accuracy_score(y[g==s],yp[g==s]),2) for s in sorted(set(g))}
def kf(X,est): return cross_val_score(est,X,y,cv=skf).mean()

# ---- Experiment 1: normalization schemes ----
schemes = {"raw":Xb, "height-norm":Xh, "baseline-norm":Xnorm, "baseline+height":Xnh}
r1=[]
for sn,X in schemes.items():
    for mn,est in models.items():
        acc,ps = lopo(X,est); r1.append(dict(scheme=sn,model=mn,lopo=round(acc,3),**{f"s_{k}":v for k,v in ps.items()}))
R1=pd.DataFrame(r1); R1.to_csv("anthro_normalization.csv",index=False)

def cv_subj(X,idx):
    m = pd.DataFrame({"g":g,"v":X[:,idx]}).groupby("g").v.mean(); return m.std()/abs(m.mean())
print("Height range in sample:", sorted(demo['h'].tolist()), "-> little size variance")
print("Between-subject CV (raw -> height-norm):")
for nm,idx in [("rms_ml",2),("sway_amp_ml",5),("ellipse",7)]:
    print(f"  {nm:12s} {cv_subj(Xb,idx):.3f} -> {cv_subj(Xh,idx):.3f}")

# ---- Experiment 2: demographics as raw features ----
def add(cols):
    D = np.array([[demo.loc[grp[t],c] for c in cols] for t in tids]); return np.column_stack([Xb,D])
featsets = {"8 CoP (base)":Xb, "+height":add(['h']), "+weight":add(['w']), "+BMI":add(['BMI']),
            "+gender":add(['sexM']), "+ALL demo":add(['age','h','w','BMI','sexM'])}
r2=[]
for sn,X in featsets.items():
    for mn,est in models.items():
        lo,_ = lopo(X,est); k = kf(X,est); r2.append(dict(feature_set=sn,model=mn,lopo=round(lo,3),kfold=round(k,3),gap=round(k-lo,3)))
R2=pd.DataFrame(r2); R2.to_csv("demographics_features.csv",index=False)
print("\n=== Exp1: normalization (LOPO) ===\n", R1.to_string(index=False))
print("\n=== Exp2: demographics as features (LOPO vs k-fold) ===\n", R2.to_string(index=False))

# ---- figure ----
fig,(axA,axB)=plt.subplots(1,2,figsize=(14,5.2))
sc=list(schemes); xa=np.arange(len(sc)); w=0.38
for i,mn in enumerate(models):
    axA.bar(xa+(i-0.5)*w,[R1[(R1.scheme==s)&(R1.model==mn)].lopo.values[0] for s in sc],w,label=mn)
axA.axhline(0.5,color="gray",ls="--",lw=1,label="chance"); axA.set_xticks(xa); axA.set_xticklabels(sc,rotation=12)
axA.set_ylim(0,1); axA.set_ylabel("LOPO accuracy"); axA.set_title("(A) Normalization schemes — baseline-norm is the lever, height isn't",fontsize=10,fontweight="bold"); axA.legend(fontsize=8)
fs=list(featsets); xb=np.arange(len(fs))
axB.bar(xb-0.2,[R2[(R2.feature_set==s)&(R2.model=='RForest')].lopo.values[0] for s in fs],0.4,label="LOPO (honest)")
axB.bar(xb+0.2,[R2[(R2.feature_set==s)&(R2.model=='RForest')].kfold.values[0] for s in fs],0.4,label="k-fold (optimistic)")
axB.axhline(0.5,color="gray",ls="--",lw=1); axB.set_xticks(xb); axB.set_xticklabels(fs,rotation=15)
axB.set_ylim(0,1); axB.set_ylabel("accuracy"); axB.set_title("(B) RForest: adding demographics inflates k-fold, not LOPO = overfitting",fontsize=10,fontweight="bold"); axB.legend(fontsize=8)
fig.suptitle("Anthropometrics — correct use (normalize) vs misuse (raw demographic features), n=5",fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig("anthropometrics.png",dpi=140,bbox_inches="tight")
print("\nsaved figure")
