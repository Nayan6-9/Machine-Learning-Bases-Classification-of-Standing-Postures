"""Per-subject CoP scatter — shows the S04 slouch-direction reversal.
One panel per subject: light CoP clouds (neutral vs slouched) + per-trial
centroids + an arrow from the neutral centroid to the slouched centroid.
Sixth panel: slouch-minus-neutral AP shift per subject (S04 is the lone
forward/positive bar). Axes: x=ML (+right), y=AP (+anterior/front)."""
import os, re, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.signal import butter, filtfilt

SAMP = r"C:\SRELO\IMSA\Tool\WELAB_WBB V1.2\WELAB_WBB\wbb_db\samples"
FS = 100.0
def blp(s): b,a = butter(4, 10/(FS/2), "low"); return filtfilt(b,a,s)

C = {"neutral":"#2a9d8f", "slouched":"#e76f51", "baseline":"#9aa0a6"}
data = {}   # subj -> label -> list of (x,y) arrays
for p in sorted(glob.glob(SAMP+"/*.csv")):
    m = re.match(r"(\d+)_([A-Za-z]+)_(S\d+)", os.path.basename(p))
    tid,label,subj = m.group(1), m.group(2).lower(), m.group(3)
    df = pd.read_csv(p); t = df.time_s.values; k = t >= t[0]+5
    x = blp(df.cop_x_cm.values)[k]; y = blp(df.cop_y_cm.values)[k]
    data.setdefault(subj, {}).setdefault(label, []).append((x, y))

subs = sorted(data)
# shared limits
allx = np.concatenate([xy[0] for s in subs for l in data[s] for xy in data[s][l]])
ally = np.concatenate([xy[1] for s in subs for l in data[s] for xy in data[s][l]])
xlim = (np.percentile(allx,0.5)-0.5, np.percentile(allx,99.5)+0.5)
ylim = (np.percentile(ally,0.5)-0.5, np.percentile(ally,99.5)+0.5)

fig, axes = plt.subplots(2, 3, figsize=(14, 8.5)); axes = axes.ravel()
def centroid(arrs):
    xs = np.concatenate([a[0] for a in arrs]); ys = np.concatenate([a[1] for a in arrs])
    return xs.mean(), ys.mean()

for ax, s in zip(axes, subs):
    for label in ["baseline","neutral","slouched"]:
        for (x,y) in data[s].get(label, []):
            ax.scatter(x[::6], y[::6], s=3, c=C[label], alpha=0.10, linewidths=0)
    cen = {}
    for label in ["neutral","slouched"]:
        if data[s].get(label):
            cx,cy = centroid(data[s][label]); cen[label]=(cx,cy)
            ax.scatter([cx],[cy], s=170, c=C[label], edgecolors="black",
                       linewidths=1.4, zorder=5, label=label)
    if "neutral" in cen and "slouched" in cen:
        ar = FancyArrowPatch(cen["neutral"], cen["slouched"], arrowstyle="-|>",
                             mutation_scale=22, lw=2.6, color="black", zorder=6)
        ax.add_patch(ar)
        d_ap = cen["slouched"][1]-cen["neutral"][1]
        ax.text(0.5,0.97,("slouch → FORWARD" if d_ap>0 else "slouch → backward"),
                transform=ax.transAxes, ha="center", va="top", fontsize=10,
                fontweight="bold", color=("#c1121f" if d_ap>0 else "#333"))
    tag = "  ← REVERSED" if s=="S04" else ""
    ax.set_title(f"{s}{tag}", fontweight="bold", color=("#c1121f" if s=="S04" else "black"))
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.axhline(0,color="#ddd",lw=.8); ax.axvline(0,color="#ddd",lw=.8)
    ax.set_xlabel("ML  (cm, + = right)"); ax.set_ylabel("AP  (cm, + = anterior/front)")
    if s==subs[0]: ax.legend(loc="lower left", fontsize=8, framealpha=.9)

# summary bar panel
ax = axes[5]
deltas = []
for s in subs:
    _,ny = centroid(data[s]["neutral"]); _,sy = centroid(data[s]["slouched"])
    deltas.append(sy-ny)
colors = ["#c1121f" if d>0 else "#2a9d8f" for d in deltas]
ax.barh(subs, deltas, color=colors, edgecolor="black")
ax.axvline(0, color="black", lw=1)
ax.set_title("Slouch − Neutral AP shift  (cm)", fontweight="bold")
ax.set_xlabel("← backward (posterior)      forward (anterior) →")
for i,d in enumerate(deltas):
    ax.text(d + (0.1 if d>0 else -0.1), i, f"{d:+.2f}", va="center",
            ha=("left" if d>0 else "right"), fontsize=9, fontweight="bold")
ax.invert_yaxis()

fig.suptitle("Center-of-Pressure by posture, per subject — S04 slouches FORWARD, all others backward",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig("cop_by_subject.png", dpi=140, bbox_inches="tight")
print("saved. AP shifts:", {s:round(d,2) for s,d in zip(subs,deltas)})
