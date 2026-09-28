from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "source_data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})

NAVY = "#264C77"; BLUE = "#4D82B8"; TEAL = "#2A9D8F"
LIGHT = "#DCE8F2"; GREY = "#9AA3AA"; DARK = "#343A40"
RED = "#C44E52"; ORANGE = "#E9A23B"; GRID = "#E7EAED"

def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.6, zorder=0)
    ax.tick_params(length=2.5, width=.6)

def panel(ax, label):
    ax.text(-0.13, 1.06, label, transform=ax.transAxes, weight="bold",
            fontsize=8, va="top")

def export(fig, name):
    for ext, kwargs in {
        "svg": {}, "pdf": {}, "png": {"dpi": 300}, "tiff": {"dpi": 600}
    }.items():
        fig.savefig(FIG / f"{name}.{ext}", **kwargs)
    plt.close(fig)

def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    items = [
        (0.25, 1.45, 1.35, 1.05, "Paired CT", LIGHT, NAVY),
        (2.05, 1.25, 1.75, 1.45, "Frozen\nSelfRDB bridge", "#E8EDF2", DARK),
        (4.25, 1.45, 1.35, 1.05, "Coarse MRI", "#E8EDF2", DARK),
        (6.05, 1.15, 1.9, 1.65, "Supervised refiner\n1.3M trainable\nparameters", "#DDF2EE", TEAL),
        (8.45, 1.45, 1.3, 1.05, "Synthetic MRI", "#CBE9E4", TEAL),
    ]
    for x,y,w,h,t,fc,ec in items:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=.08",
                                    facecolor=fc,edgecolor=ec,lw=1.2))
        ax.text(x+w/2,y+h/2,t,ha="center",va="center",weight="bold" if "refiner" in t else None)
    for a,b in [(1.6,2.05),(3.8,4.25),(5.6,6.05),(7.95,8.45)]:
        ax.add_patch(FancyArrowPatch((a,1.98),(b,1.98),arrowstyle="-|>",mutation_scale=10,lw=1,color=DARK))
    ax.add_patch(FancyArrowPatch((1.0,1.45),(6.25,1.25),connectionstyle="arc3,rad=.23",
                                 arrowstyle="-|>",mutation_scale=10,lw=1,color=NAVY))
    ax.text(3.55,.62,"CT skip-conditioning",ha="center",color=NAVY)
    ax.text(2.92,3.12,"20 sampling steps; 5 recursive estimates",ha="center",color=DARK)
    ax.text(7.0,3.22,"Optimized against paired MRI with L1",ha="center",color=TEAL)
    ax.text(.05,3.83,"a",weight="bold",fontsize=8)
    export(fig,"fig1_pipeline")

def load_patient():
    p = ROOT.parent / "phase1_results" / "patient_summary.csv"
    df = pd.read_csv(p)
    shutil.copy2(p, DATA / "patient_summary.csv")
    return df

def fig2_main():
    df = load_patient()
    methods = ["SelfRDB","SynDiff","MG-CycleGAN","BridgeGAN","Bridge+L1UNet"]
    labels = ["SelfRDB","SynDiff","MG-CycleGAN","BridgeGAN","BridgeRefine"]
    colors = [GREY,"#AABFD1",BLUE,NAVY,TEAL]
    metrics = [("ssim","SSIM"),("psnr","PSNR (dB)"),("mssim","Mask SSIM"),("mpsnr","Mask PSNR (dB)")]
    fig, axs = plt.subplots(2,2,figsize=(7.2,5.2),constrained_layout=True)
    x=np.arange(len(methods))
    for i,(ax,(suffix,ylabel)) in enumerate(zip(axs.flat,metrics)):
        vals=[]; errs=[]
        for m in methods:
            s=df[f"{m}_{suffix}"]
            vals.append(s.mean()); errs.append(s.std(ddof=1))
        ax.bar(x,vals,yerr=errs,color=colors,edgecolor="white",lw=.5,capsize=2,zorder=3)
        ax.set_ylabel(ylabel); ax.set_xticks(x,labels,rotation=24,ha="right")
        clean(ax); panel(ax,chr(97+i))
        pad=(max(vals)-min(vals))*.35
        ax.set_ylim(min(vals)-pad,max(vals)+pad)
    export(fig,"fig2_main_results")

def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m-rank)*p[idx])
        out[idx] = min(running, 1.0)
    return out

def fig2_high_impact():
    """Six-panel evidence figure using only verified patient and final-summary data."""
    df = load_patient()
    methods = ["SelfRDB","SynDiff","MG-CycleGAN","BridgeGAN","Bridge+L1UNet"]
    labels = ["SelfRDB","SynDiff","MG-CycleGAN","BridgeGAN","BridgeRefine"]
    colors = [GREY,"#9FB8CD",BLUE,NAVY,TEAL]
    x = np.arange(len(methods))
    fig = plt.figure(figsize=(7.2,7.25))
    gs = fig.add_gridspec(3,2,hspace=.46,wspace=.36)
    axs = [fig.add_subplot(gs[i,j]) for i in range(3) for j in range(2)]

    # a: patient distributions with actual paired tests against BridgeRefine
    ax=axs[0]
    vals=[df[f"{m}_ssim"].to_numpy() for m in methods]
    bp=ax.boxplot(vals,positions=x,widths=.58,patch_artist=True,showfliers=False,
                  medianprops=dict(color=DARK,lw=1.1),
                  whiskerprops=dict(color=DARK,lw=.8),capprops=dict(color=DARK,lw=.8))
    for box,c in zip(bp["boxes"],colors): box.set(facecolor=c,alpha=.34,edgecolor=c,lw=1)
    rng=np.random.default_rng(42)
    for i,(v,c) in enumerate(zip(vals,colors)):
        ax.scatter(i+rng.uniform(-.16,.16,len(v)),v,s=10,color=c,alpha=.72,edgecolor="white",lw=.25,zorder=3)
        ax.scatter(i,np.mean(v),marker="D",s=25,color=c,edgecolor="white",lw=.6,zorder=4)
    raw=[wilcoxon(vals[-1],vals[i],alternative="greater").pvalue for i in range(4)]
    adj=holm_adjust(raw)
    for i,p in enumerate(adj):
        y=.898-i*.011
        ax.plot([i,i,4,4],[y-.002,y,y,y-.002],color=DARK,lw=.55)
        txt="***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns"
        ax.text((i+4)/2,y+.001,txt,ha="center",va="bottom",fontsize=6)
    ax.set_xticks(x,labels,rotation=25,ha="right"); ax.set_ylabel("Patient-level SSIM"); ax.set_ylim(.48,.91)
    clean(ax); panel(ax,"a"); ax.set_title("Patient-level reconstruction",pad=7)
    pd.DataFrame({"comparison":[f"BridgeRefine vs {l}" for l in labels[:-1]],"raw_p":raw,"holm_p":adj}).to_csv(DATA/"figure2_significance.csv",index=False)

    # b: paired patient trajectories across model stages
    ax=axs[1]
    mat=np.column_stack(vals)
    for row in mat: ax.plot(x,row,color="#B7BEC4",lw=.5,alpha=.42,zorder=1)
    means=mat.mean(axis=0)
    ax.plot(x,means,color=NAVY,lw=1.6,marker="o",ms=4,mfc="white",mew=1.2,zorder=3)
    ax.fill_between(x,means-mat.std(axis=0,ddof=1),means+mat.std(axis=0,ddof=1),color=LIGHT,alpha=.55,zorder=0)
    ax.set_xticks(x,labels,rotation=25,ha="right"); ax.set_ylabel("SSIM"); ax.set_ylim(.48,.90)
    clean(ax); panel(ax,"b"); ax.set_title("Within-patient trajectories",pad=7)

    # c: whole-image and mask-restricted paired metrics
    ax=axs[2]; w=.36
    whole=np.array([df[f"{m}_ssim"].mean() for m in methods]); mask=np.array([df[f"{m}_mssim"].mean() for m in methods])
    ew=np.array([df[f"{m}_ssim"].std(ddof=1) for m in methods]); em=np.array([df[f"{m}_mssim"].std(ddof=1) for m in methods])
    ax.bar(x-w/2,whole,w,yerr=ew,color="#D7DCE0",edgecolor=DARK,lw=.5,capsize=2,label="Whole image",zorder=3)
    ax.bar(x+w/2,mask,w,yerr=em,color=colors,edgecolor="white",lw=.4,capsize=2,label="Brain mask",zorder=3)
    ax.set_xticks(x,labels,rotation=25,ha="right"); ax.set_ylabel("SSIM"); ax.set_ylim(.50,.92)
    clean(ax); panel(ax,"c"); ax.set_title("Whole-image and mask metrics",pad=7); ax.legend(frameon=False,ncol=2,loc="upper left")

    # d: documented accuracy-efficiency relation
    ax=axs[3]
    times=[2,1264]; ss=[.8009,.8165]; pars=[1.33,15.74]
    ax.scatter(times,ss,s=np.array(pars)*9,color=[NAVY,TEAL],alpha=.9,edgecolor="white",lw=.8,zorder=3)
    for tx,sy,lab,off in zip(times,ss,["CT-only\n3 seeds","BridgeRefine\n2 seeds"],[(7,-.0015),(-610,-.004)]):
        t=ax.annotate(lab,(tx,sy),xytext=(tx+off[0],sy+off[1]),fontsize=6.5,arrowprops=dict(arrowstyle="-",lw=.6,color=DARK))
    ax.set_xscale("log"); ax.set_xlabel("Inference time per slice (ms; log scale)"); ax.set_ylabel("Cohort mean SSIM"); ax.set_ylim(.796,.821)
    clean(ax); panel(ax,"d"); ax.set_title("Accuracy--efficiency trade-off",pad=7)
    ax.text(.02,.04,"Marker area encodes total parameters",transform=ax.transAxes,fontsize=6,color=DARK)

    # e: correlation structure across 90 patient-method observations
    ax=axs[4]
    long=[]
    for m in methods:
        for _,r in df.iterrows(): long.append([r[f"{m}_ssim"],r[f"{m}_psnr"],r[f"{m}_mssim"],r[f"{m}_mpsnr"]])
    corr=pd.DataFrame(long,columns=["SSIM","PSNR","Mask SSIM","Mask PSNR"]).corr()
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap="RdBu_r")
    for i in range(4):
        for j in range(4):
            v=corr.iloc[i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=6,color="white" if abs(v)>.65 else DARK)
    ax.set_xticks(range(4),corr.columns,rotation=35,ha="right"); ax.set_yticks(range(4),corr.columns)
    cb=fig.colorbar(im,ax=ax,fraction=.047,pad=.03); cb.ax.tick_params(labelsize=5,length=2)
    panel(ax,"e"); ax.set_title("Metric correlation matrix",pad=7)
    corr.to_csv(DATA/"figure2_correlation.csv")

    # f: method-level perceptual vs through-plane behavior
    ax=axs[5]
    lp=np.array([.231,.122,.082,.080,.075]); mad=np.array([3.764,1.578,1.197,1.461,1.005]); err=np.abs(mad-1)
    ax.scatter(lp,err,s=[35,42,48,54,72],color=colors,edgecolor="white",lw=.7,zorder=3)
    label_positions=[(.219,2.87),(.128,.66),(.104,.05),(.084,.52),(.074,.18)]
    for xx,yy,lab,(tx,ty) in zip(lp,err,labels,label_positions):
        txt=ax.annotate(lab,(xx,yy),xytext=(tx,ty),fontsize=6,color=DARK,
                        arrowprops=dict(arrowstyle="-",lw=.45,color="#737A80"))
        txt.set_path_effects([pe.withStroke(linewidth=2,foreground="white")])
    ax.set_xlabel("LPIPS (lower is better)"); ax.set_ylabel(r"$|$inter-slice MAD ratio $-1|$")
    clean(ax); panel(ax,"f"); ax.set_title("Perception--continuity relation",pad=7)
    ax.annotate("desirable",xy=(.073,.02),xytext=(.13,1.8),fontsize=6,color=TEAL,
                arrowprops=dict(arrowstyle="->",lw=.7,color=TEAL))
    summary=pd.DataFrame({"method":labels,"ssim_mean":whole,"ssim_sd":ew,"mask_ssim_mean":mask,"mask_ssim_sd":em,"lpips":lp,"mad_ratio":mad})
    summary.to_csv(DATA/"figure2_method_summary.csv",index=False)
    export(fig,"fig2_evidence_landscape")

def fig3_robustness():
    fig, axs = plt.subplots(1,3,figsize=(7.2,2.45),constrained_layout=True)
    # seeds
    ax=axs[0]; ct=[.8095,.8050,.7883]; br=[.8152,.8178]
    for j,(v,c,n) in enumerate([(ct,NAVY,"CT-only\nL1 (n=3)"),(br,TEAL,"BridgeRefine\nL1+grad (n=2)")]):
        ax.scatter(np.full(len(v),j)+np.linspace(-.06,.06,len(v)),v,s=25,color=c,zorder=3)
        ax.hlines(np.mean(v),j-.18,j+.18,color=DARK,lw=1.2)
    ax.set_xticks([0,1],["CT-only","BridgeRefine"]); ax.set_ylabel("Cohort mean SSIM")
    ax.set_ylim(.782,.822); clean(ax); panel(ax,"a")
    # exploratory loss
    ax=axs[1]; labs=["Bridge\nonly","L1","L1+SSIM","L1+grad","L1+GAN"]; vals=[.598,.869,.868,.878,.803]
    ax.bar(range(5),vals,color=[GREY,BLUE,"#86A8C5",TEAL,RED],zorder=3)
    ax.set_xticks(range(5),labs,rotation=25,ha="right"); ax.set_ylabel("SSIM")
    ax.set_ylim(.55,.91); clean(ax); panel(ax,"b")
    ax.text(.5,.98,"Exploratory: 200 slices",transform=ax.transAxes,ha="center",va="top",fontsize=6.5)
    # pilot
    ax=axs[2]; labs=["SelfRDB","Adv.\n0.10","Adv.\n0.02","Supervised\npost-bridge"]; vals=[.66,.39,.43,.82]
    ax.bar(range(4),vals,color=[GREY,RED,ORANGE,TEAL],zorder=3)
    ax.set_xticks(range(4),labs,rotation=24,ha="right"); ax.set_ylabel("SSIM")
    ax.set_ylim(.32,.86); clean(ax); panel(ax,"c")
    ax.text(.5,.98,"Pilot: 3 patients",transform=ax.transAxes,ha="center",va="top",fontsize=6.5)
    pd.DataFrame({"model":["CT-only"]*3+["BridgeRefine-L1+gradient"]*2,
                  "seed":[42,123,2026,123,2026],"ssim":ct+br}).to_csv(DATA/"seed_results.csv",index=False)
    export(fig,"fig3_robustness_ablation")

def fig4_efficiency():
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.7),constrained_layout=True)
    ax=axs[0]
    ax.scatter([2,1264],[.8009,.8165],s=[60,95],color=[NAVY,TEAL],zorder=3)
    ax.set_xscale("log"); ax.set_xlabel("Inference time per slice (ms, log scale)"); ax.set_ylabel("Multi-seed cohort SSIM")
    ax.annotate("CT-only\n3 seeds",(2,.8009),xytext=(6,.799),arrowprops=dict(arrowstyle="-",color=DARK),va="top")
    ax.annotate("BridgeRefine\n2 seeds",(1264,.8165),xytext=(200,.8125),arrowprops=dict(arrowstyle="-",color=DARK))
    ax.set_ylim(.796,.820); clean(ax); panel(ax,"a")
    ax=axs[1]; labs=["CT-only","BridgeRefine"]; mem=[99,133]; bars=ax.bar(labs,mem,color=[NAVY,TEAL],zorder=3)
    ax.set_ylabel("Peak additional inference memory (MB)"); ax.set_ylim(0,150); clean(ax); panel(ax,"b")
    for b,v in zip(bars,mem): ax.text(b.get_x()+b.get_width()/2,v+3,str(v),ha="center")
    pd.DataFrame({"method":labs,"ssim":[.8009,.8165],"seeds":[3,2],"time_ms":[2,1264],"memory_mb":mem}).to_csv(DATA/"efficiency.csv",index=False)
    export(fig,"fig4_efficiency")

def fig5_images():
    src=ROOT.parent/"phase3_final"
    files=["1BA222_3d.png","1BC050_3d.png","1BB205_3d.png"]
    titles=["Higher-SSIM case: 1BA222","Median case: 1BC050","Lower-SSIM case: 1BB205"]
    fig,axs=plt.subplots(1,3,figsize=(7.2,4.5),facecolor="white")
    for i,(ax,f,t) in enumerate(zip(axs,files,titles)):
        ax.imshow(plt.imread(src/f)); ax.axis("off"); ax.set_title(t,loc="left",fontsize=8,pad=3)
        ax.text(-.025,1.01,chr(97+i),transform=ax.transAxes,weight="bold",fontsize=8,va="top")
    fig.subplots_adjust(wspace=.035,left=.01,right=.995,top=.94,bottom=.01)
    export(fig,"fig5_qualitative")

if __name__ == "__main__":
    fig1_pipeline(); fig2_main(); fig2_high_impact(); fig3_robustness(); fig4_efficiency(); fig5_images()
