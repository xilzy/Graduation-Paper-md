#!/usr/bin/env python3
"""Assemble a reference-paper-style qualitative comparison figure for one task.

For a chosen sample it renders: source A, source B, then every comparison method
and Ours. On each panel it draws a red rectangle over a fixed key region (same
coordinates for all panels) and pastes a red-bordered zoomed-in crop of that
region into a corner — exactly the "local enlarged view" style of the reference
paper. It writes both:
  1. individual annotated panels  (for hand-assembly in PPT)
  2. a single stitched montage PNG (ready to drop into the thesis)

irvis is scored/shown in grayscale (output_mode=gray) so its fused images come
from fused/. medical and gfp_pc are color tasks, so their fused images come from
the RGB-final recombination in fused_final/. Ours(v3) lives in folder W96L.

Usage:
  python make_qualitative_figure.py --task irvis --sample 00004N \
      --box 0.35 0.40 0.28 0.28 --corner br --ncols 12
"""
import os, argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
REPO = "/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md"

# (display label, folder name in fused/ or fused_final/). Ours(v3)=W96L. NSCT folder has a '*'.
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
    """Draw red rect over `box` (relative x,y,w,h) and paste a zoomed crop in a corner."""
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
    cw, ch = crop.size
    framed = Image.new("RGB", (cw + 2 * border, ch + 2 * border), (255, 0, 0))
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
    ap.add_argument("--sample", required=True, help="stem (with or without .png)")
    ap.add_argument("--box", nargs=4, type=float, default=[0.35, 0.40, 0.28, 0.28],
                    help="key region as relative x y w h")
    ap.add_argument("--zoom", type=float, default=2.6)
    ap.add_argument("--border", type=int, default=4)
    ap.add_argument("--corner", default="br", choices=["br", "bl", "tr", "tl"])
    ap.add_argument("--ncols", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    stem = args.sample[:-4] if args.sample.endswith(".png") else args.sample
    fn = stem + ".png"
    task = args.task
    out_base = os.path.join(REPO, "Materials", "comparison", task)
    ind_dir = os.path.join(out_base, "individual")
    os.makedirs(ind_dir, exist_ok=True)

    # sources: A in color (colorA), B in grayscale
    a_path = os.path.join(BENCH, "inputs", task, "colorA", fn)
    if not os.path.exists(a_path):
        a_path = os.path.join(BENCH, "inputs", task, "A", fn)
    b_path = os.path.join(BENCH, "inputs", task, "B", fn)
    srcA = Image.open(a_path).convert("RGB")
    ref = srcA.size
    srcB = load_rgb(b_path, ref)

    la, lb = SRC_LABELS[task]
    panels = [(la, srcA), (lb, srcB)]
    for disp, folder in METHODS:
        p = os.path.join(fused_dir(task, folder), fn)
        if not os.path.exists(p):
            print(f"[warn] missing {disp}: {p}")
            continue
        panels.append((disp, load_rgb(p, ref)))

    # annotate + save individual panels
    annotated = []
    for label, img in panels:
        ann = annotate(img, args.box, args.zoom, args.border, args.corner)
        ann.save(os.path.join(ind_dir, f"{stem}__{sanitize(label)}.png"))
        annotated.append((label, ann))

    # montage
    n = len(annotated)
    ncols = min(args.ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.9, nrows * 2.05))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        label, ann = annotated[i]
        ax.imshow(np.asarray(ann))
        is_ours = (label == "Ours")
        ax.set_title(label, fontsize=11, fontweight="bold" if is_ours else "normal",
                     color="red" if is_ours else "black", pad=3)
    fig.suptitle(f"Qualitative comparison on {task} (sample {stem})",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(out_base, f"fig_{task}_qualitative.png")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[{task}] {n} panels -> {out_png}")
    print(f"       individual panels -> {ind_dir}")


if __name__ == "__main__":
    main()
