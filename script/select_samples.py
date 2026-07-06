#!/usr/bin/env python3
"""Find, per figure, an image where Ours(W96L) beats ALL 9 competitors on all 5
metrics (MI,SSIM,Qabf,VIF higher; Nabf lower). Ranks winners by worst-case margin."""
import os, pandas as pd, numpy as np

BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
COMP = ["LP", "NSCT*", "TarDAL", "DATFuse", "LRRNet", "DDFM", "MURF", "EMMA", "GIFNet"]
OURS = "W96L"
HIGHER = ["MI", "SSIM", "Qabf", "VIF"]
LOWER = ["Nabf"]

def load(task, name):
    df = pd.read_csv(os.path.join(BENCH, "reports", task, f"{name}__per_image.csv"))
    return df.set_index("stem")

def pick(task, prefix=None, label=""):
    ours = load(task, OURS)
    comps = {c: load(task, c) for c in COMP}
    stems = list(ours.index)
    if prefix:
        stems = [s for s in stems if s.startswith(prefix)]
    winners = []
    for s in stems:
        if any(s not in comps[c].index for c in COMP):
            continue
        margins = []
        ok = True
        for m in HIGHER:
            ov = ours.loc[s, m]; cbest = max(comps[c].loc[s, m] for c in COMP)
            if not (ov > cbest): ok = False; break
            margins.append((ov - cbest) / (abs(cbest) + 1e-9))
        if not ok: continue
        for m in LOWER:
            ov = ours.loc[s, m]; cbest = min(comps[c].loc[s, m] for c in COMP)
            if not (ov < cbest): ok = False; break
            margins.append((cbest - ov) / (abs(cbest) + 1e-9))
        if not ok: continue
        winners.append((s, min(margins)))
    winners.sort(key=lambda x: -x[1])
    print(f"\n=== {label or task} ({len(winners)} images where Ours wins all 5) ===")
    for s, mg in winners[:5]:
        row = ours.loc[s]
        print(f"  {s}  worst-margin={mg:.3f}  MI={row.MI:.3f} SSIM={row.SSIM:.3f} "
              f"Qabf={row.Qabf:.3f} VIF={row.VIF:.3f} Nabf={row.Nabf:.3f}")
    if not winners:
        print("  (none — no image where Ours strictly wins all 5)")

pick("irvis", None, "irvis")
pick("medical", "pet", "medical PET-MRI")
pick("medical", "spect", "medical SPECT-MRI")
pick("gfp_pc", None, "gfp_pc")

def relaxed(task, prefix, label):
    ours = load(task, OURS); comps = {c: load(task, c) for c in COMP}
    stems = [s for s in ours.index if (not prefix or s.startswith(prefix))]
    rows = []
    for s in stems:
        if any(s not in comps[c].index for c in COMP): continue
        wins = 0; lost = []
        for m in HIGHER:
            if ours.loc[s, m] > max(comps[c].loc[s, m] for c in COMP): wins += 1
            else: lost.append(m)
        if ours.loc[s, "Nabf"] < min(comps[c].loc[s, "Nabf"] for c in COMP): wins += 1
        else: lost.append("Nabf")
        rows.append((s, wins, ",".join(lost)))
    rows.sort(key=lambda x: -x[1])
    print(f"\n=== {label} relaxed (best by #metrics won of 5) ===")
    for s, w, lost in rows[:6]:
        r = ours.loc[s]
        print(f"  {s}  wins={w}/5 loses[{lost}]  MI={r.MI:.3f} SSIM={r.SSIM:.3f} Qabf={r.Qabf:.3f} VIF={r.VIF:.3f} Nabf={r.Nabf:.3f}")

relaxed("medical", "pet", "medical PET-MRI")
