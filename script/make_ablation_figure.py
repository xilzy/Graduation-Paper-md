#!/usr/bin/env python3
"""Qualitative ABLATION figure for one task: full v3 vs each innovation removed.

Columns: source A, source B, Full(v3), -MoE, -DecisionHead, -WindowAttn, -maxfuse,
-TaskCond  (2x4 grid). Same red-box + bottom-left zoom-inset style, (a)-(h) captions.
IR-VIS shown in color (visible-chroma recombination); medical/gfp_pc from fused_final.
Only INNOVATION-point ablations here (NOT hyperparameter sweeps).

Usage:
  python make_ablation_figure.py --task irvis  --sample 00778N     --box 0.40 0.45 0.16 0.16
  python make_ablation_figure.py --task gfp_pc --sample 05-A02     --box 0.40 0.45 0.16 0.16
  python make_ablation_figure.py --task medical --sample spect_18017 --box 0.38 0.40 0.18 0.18
"""
import os, string, argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
REPO = "/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md"

# (display label, folder). Innovation ablations only.
CONFIGS = [
    ("Full (v3)", "W96L"), ("-MoE", "abNoMoE"), ("-Decision head", "abDirect"),
    ("-Window attn", "abWs1"), ("-maxfuse", "abOrig"), ("-Task cond", "abNoTC"),
]
SRC_LABELS = {"irvis": ("Visible", "Infrared"),
              "medical": ("PET/SPECT", "MRI"), "gfp_pc": ("GFP", "PC")}


def fused_dir(task, folder):
    root = "fused" if task == "irvis" else "fused_final"
    return os.path.join(BENCH, root, folder, task)


def load_rgb(path, ref=None):
    img = Image.open(path).convert("RGB")
    if ref is not None and img.size != ref:
        img = img.resize(ref, Image.BILINEAR)
    return img


def recolor_irvis(fused_gray, visible_rgb):
    y = fused_gray.convert("L")
    _, cb, cr = visible_rgb.convert("YCbCr").split()
    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")


def annotate(img, box, zoom, border, corner):
    img = img.copy(); W, H = img.size
    x, y, w, h = box
    bx, by, bw, bh = int(x * W), int(y * H), int(w * W), int(h * H)
    bx = max(0, min(bx, W - 2)); by = max(0, min(by, H - 2))
    bw = max(2, min(bw, W - bx)); bh = max(2, min(bh, H - by))
    crop = img.crop((bx, by, bx + bw, by + bh)).resize(
        (int(bw * zoom), int(bh * zoom)), Image.BICUBIC)
    ImageDraw.Draw(img).rectangle([bx, by, bx + bw, by + bh], outline=(255, 0, 0), width=border)
    framed = Image.new("RGB", (crop.size[0] + 2 * border, crop.size[1] + 2 * border), (255, 0, 0))
    framed.paste(crop, (border, border))
    cw, ch = min(framed.size[0], W), min(framed.size[1], H)
    framed = framed.crop((0, 0, cw, ch))
    pos = {"br": (W - cw, H - ch), "bl": (0, H - ch), "tr": (W - cw, 0), "tl": (0, 0)}[corner]
    img.paste(framed, pos); return img


def sanitize(s):
    return s.replace("/", "-").replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["irvis", "medical", "gfp_pc"])
    ap.add_argument("--sample", required=True)
    ap.add_argument("--box", nargs=4, type=float, default=[0.40, 0.45, 0.16, 0.16])
    ap.add_argument("--zoom", type=float, default=2.2)
    ap.add_argument("--border", type=int, default=3)
    ap.add_argument("--corner", default="bl", choices=["br", "bl", "tr", "tl"])
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    stem = args.sample[:-4] if args.sample.endswith(".png") else args.sample
    fn = stem + ".png"; task = args.task
    out_base = os.path.join(REPO, "Materials", "ablation", task)
    ind_dir = os.path.join(out_base, "individual"); os.makedirs(ind_dir, exist_ok=True)

    la, lb = SRC_LABELS[task]
    if task == "medical":
        la = "SPECT" if stem.startswith("spect") else "PET"
    a_path = os.path.join(BENCH, "inputs", task, "colorA", fn)
    if not os.path.exists(a_path):
        a_path = os.path.join(BENCH, "inputs", task, "A", fn)
    srcA = Image.open(a_path).convert("RGB"); ref = srcA.size
    srcB = load_rgb(os.path.join(BENCH, "inputs", task, "B", fn), ref)

    panels = [(la, srcA), (lb, srcB)]
    for disp, folder in CONFIGS:
        p = os.path.join(fused_dir(task, folder), fn)
        if not os.path.exists(p):
            print(f"[warn] missing {disp}: {p}"); continue
        if task == "irvis":
            fg = Image.open(p).convert("L").resize(ref, Image.BILINEAR)
            panels.append((disp, recolor_irvis(fg, srcA)))
        else:
            panels.append((disp, load_rgb(p, ref)))

    letters = string.ascii_lowercase
    annotated = []
    for label, img in panels:
        ann = annotate(img, args.box, args.zoom, args.border, args.corner)
        ann.save(os.path.join(ind_dir, f"{stem}__{sanitize(label)}.png"))
        annotated.append((label, ann))

    n = len(annotated); ncols = min(args.ncols, n); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.25))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        label, ann = annotated[i]
        ax.imshow(np.asarray(ann))
        ax.text(0.5, -0.05, f"({letters[i]}) {label}", transform=ax.transAxes,
                ha="center", va="top", fontsize=11, color="black")
    fig.tight_layout(h_pad=1.6, w_pad=0.6)
    out_png = os.path.join(out_base, f"fig_{task}_ablation.png")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
    print(f"[{task}] {n} panels {nrows}x{ncols} -> {out_png}")


if __name__ == "__main__":
    main()
