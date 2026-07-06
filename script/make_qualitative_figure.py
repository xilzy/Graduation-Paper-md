#!/usr/bin/env python3
"""Assemble a reference-paper-style qualitative comparison figure for one task.

For a chosen sample it renders: source A, source B, then every comparison method
and Ours, as a 2x6 grid. On each panel it draws a small red rectangle over a
fixed key region (same coordinates for all panels) and pastes a small red-bordered
zoomed-in crop of that region into the bottom-left corner — the "local enlarged
view" style of the reference paper. Each panel is captioned "(a) Name", "(b) ...".
It writes both:
  1. individual annotated panels  (for hand-assembly in PPT)
  2. a single stitched montage PNG (ready to drop into the thesis)

irvis is shown in grayscale (fused/); medical & gfp_pc are color (fused_final/).
Ours(v3) lives in folder W96L. Medical is split by modality via --subtag pet|spect,
which also sets the source-A label (PET / SPECT) and the output subfolder+name.

Usage:
  python make_qualitative_figure.py --task irvis --sample 01506D --box 0.40 0.45 0.16 0.16
  python make_qualitative_figure.py --task medical --subtag pet  --sample pet_25027 --box 0.38 0.40 0.18 0.18
  python make_qualitative_figure.py --task medical --subtag spect --sample spect_18017 --box 0.38 0.40 0.18 0.18
  python make_qualitative_figure.py --task gfp_pc --sample 05-A02 --box 0.40 0.45 0.16 0.16
"""
import os, string, argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
REPO = "/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md"

# (display label, folder). Ours(v3)=W96L. NSCT folder has a '*'.
METHODS = [
    ("LP", "LP"), ("NSCT", "NSCT*"), ("TarDAL", "TarDAL"), ("DATFuse", "DATFuse"),
    ("LRRNet", "LRRNet"), ("DDFM", "DDFM"), ("MURF", "MURF"),
    ("EMMA", "EMMA"), ("GIFNet", "GIFNet"), ("Ours", "W96L"),
]
SRC_LABELS = {"irvis": ("Visible", "Infrared"),
              "medical": ("PET/SPECT", "MRI"),
              "gfp_pc": ("GFP", "PC")}


def fused_dir(task, folder):
    root = "fused" if task == "irvis" else "fused_final"
    return os.path.join(BENCH, root, folder, task)


def load_rgb(path, ref_size=None):
    img = Image.open(path).convert("RGB")
    if ref_size is not None and img.size != ref_size:
        img = img.resize(ref_size, Image.BILINEAR)
    return img


def annotate(img, box, zoom, border, corner):
    img = img.copy()
    W, H = img.size
    x, y, w, h = box
    bx, by, bw, bh = int(x * W), int(y * H), int(w * W), int(h * H)
    bx = max(0, min(bx, W - 2)); by = max(0, min(by, H - 2))
    bw = max(2, min(bw, W - bx)); bh = max(2, min(bh, H - by))
    crop = img.crop((bx, by, bx + bw, by + bh)).resize(
        (int(bw * zoom), int(bh * zoom)), Image.BICUBIC)
    d = ImageDraw.Draw(img)
    d.rectangle([bx, by, bx + bw, by + bh], outline=(255, 0, 0), width=border)
    framed = Image.new("RGB", (crop.size[0] + 2 * border, crop.size[1] + 2 * border),
                       (255, 0, 0))
    framed.paste(crop, (border, border))
    cw, ch = framed.size
    cw = min(cw, W); ch = min(ch, H)
    framed = framed.crop((0, 0, cw, ch))
    pos = {"br": (W - cw, H - ch), "bl": (0, H - ch),
           "tr": (W - cw, 0), "tl": (0, 0)}[corner]
    img.paste(framed, pos)
    return img


def sanitize(s):
    return s.replace("/", "-").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["irvis", "medical", "gfp_pc"])
    ap.add_argument("--subtag", default="", help="medical: pet|spect (subfolder + srcA label)")
    ap.add_argument("--sample", required=True, help="stem (with or without .png)")
    ap.add_argument("--box", nargs=4, type=float, default=[0.40, 0.45, 0.16, 0.16],
                    help="key region as relative x y w h (small)")
    ap.add_argument("--zoom", type=float, default=2.2)
    ap.add_argument("--border", type=int, default=3)
    ap.add_argument("--corner", default="bl", choices=["br", "bl", "tr", "tl"])
    ap.add_argument("--ncols", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    stem = args.sample[:-4] if args.sample.endswith(".png") else args.sample
    fn = stem + ".png"
    task = args.task
    sub = args.subtag

    out_base = os.path.join(REPO, "Materials", "comparison", task, sub) if sub \
        else os.path.join(REPO, "Materials", "comparison", task)
    ind_dir = os.path.join(out_base, "individual")
    os.makedirs(ind_dir, exist_ok=True)

    # source labels (medical split by modality)
    la, lb = SRC_LABELS[task]
    if task == "medical" and sub in ("pet", "spect"):
        la = sub.upper()

    a_path = os.path.join(BENCH, "inputs", task, "colorA", fn)
    if not os.path.exists(a_path):
        a_path = os.path.join(BENCH, "inputs", task, "A", fn)
    b_path = os.path.join(BENCH, "inputs", task, "B", fn)
    srcA = Image.open(a_path).convert("RGB")
    ref = srcA.size
    srcB = load_rgb(b_path, ref)

    panels = [(la, srcA), (lb, srcB)]
    for disp, folder in METHODS:
        p = os.path.join(fused_dir(task, folder), fn)
        if not os.path.exists(p):
            print(f"[warn] missing {disp}: {p}")
            continue
        panels.append((disp, load_rgb(p, ref)))

    annotated = []
    for label, img in panels:
        ann = annotate(img, args.box, args.zoom, args.border, args.corner)
        ann.save(os.path.join(ind_dir, f"{stem}__{sanitize(label)}.png"))
        annotated.append((label, ann))

    n = len(annotated)
    ncols = min(args.ncols, n)
    nrows = int(np.ceil(n / ncols))
    letters = string.ascii_lowercase
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.25))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        label, ann = annotated[i]
        ax.imshow(np.asarray(ann))
        is_ours = (label == "Ours")
        ax.text(0.5, -0.05, f"({letters[i]}) {label}", transform=ax.transAxes,
                ha="center", va="top", fontsize=11,
                fontweight="bold" if is_ours else "normal",
                color="red" if is_ours else "black")
    fig.tight_layout(h_pad=1.6, w_pad=0.6)
    suffix = f"_{sub}" if sub else ""
    out_png = os.path.join(out_base, f"fig_{task}{suffix}_qualitative.png")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[{task}{('/'+sub) if sub else ''}] {n} panels {nrows}x{ncols} -> {out_png}")


if __name__ == "__main__":
    main()
