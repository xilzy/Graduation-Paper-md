#!/usr/bin/env python3
"""Qualitative HYPERPARAMETER-sweep figure for one param on one representative task.

For a given hyperparameter (e.g. n_routed = 4/8/12/16), shows source A, source B,
then the fused result at each scanned value (v3 value labelled "(v3)"), in a grid.
Same red-box + bottom-left zoom-inset style as make_ablation_figure.py, (a)-(?)
captions. IR-VIS shown in color (visible-chroma recombination); medical/gfp_pc from
fused_final. This is for PARAMETER-VALUE sweeps (NOT innovation ablations, which
live in make_ablation_figure.py).

Representative modality per param is chosen where the effect is clearest (see
section-hyperparam.md §4.4.2). Output: Materials/hyperparam/<param>/.

Usage:
  python make_hyperparam_figure.py --param depth       --task irvis   --sample 00147D     --box 0.40 0.45 0.16 0.16
  python make_hyperparam_figure.py --param n_routed    --task medical --subtag spect --sample spect_15009 --box 0.38 0.40 0.18 0.18
"""
import os, string, argparse
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH = "/ytech_m2v4_hdd/lizhongyin/fusion_bench"
REPO = "/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md"

# param -> [(display label, fused folder), ...]; folder W96L == full v3.
PARAMS = {
    "n_routed":    [("n=4", "hpNr4"), ("n=8", "hpNr8"), ("n=12 (v3)", "W96L"), ("n=16", "hpNr16")],
    "topk":        [("k=1", "hpK1"), ("k=2 (v3)", "W96L"), ("k=4", "hpK4")],
    "n_shared":    [("ns=0", "hpNs0"), ("ns=1 (v3)", "W96L"), ("ns=2", "hpNs2")],
    "depth":       [("d=2", "hpD2"), ("d=3", "abD3"), ("d=4 (v3)", "W96L"), ("d=5", "hpD5")],
    "out_channel": [("oc=64", "hpOc64"), ("oc=96 (v3)", "W96L"), ("oc=128", "hpOc128")],
    "window_size": [("ws=4", "hpWs4"), ("ws=8 (v3)", "W96L"), ("ws=16", "hpWs16")],
    "aux_weight":  [("aux=0.001", "hpAux001"), ("aux=0.01 (v3)", "W96L"), ("aux=0.1", "hpAux1")],
    "routing":     [("softmax (v3)", "W96L"), ("deepseek", "abDeep")],
}
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
    ap.add_argument("--param", required=True, choices=list(PARAMS.keys()))
    ap.add_argument("--task", required=True, choices=["irvis", "medical", "gfp_pc"])
    ap.add_argument("--subtag", default="", help="medical: pet|spect (subfolder + srcA label)")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--box", nargs=4, type=float, default=[0.40, 0.45, 0.16, 0.16])
    ap.add_argument("--zoom", type=float, default=2.2)
    ap.add_argument("--border", type=int, default=3)
    ap.add_argument("--corner", default="bl", choices=["br", "bl", "tr", "tl"])
    ap.add_argument("--ncols", type=int, default=0, help="0 = auto (2 if panels<=4 else 3)")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    stem = args.sample[:-4] if args.sample.endswith(".png") else args.sample
    fn = stem + ".png"; task = args.task; sub = args.subtag
    out_base = os.path.join(REPO, "Materials", "hyperparam", args.param)
    ind_dir = os.path.join(out_base, "individual"); os.makedirs(ind_dir, exist_ok=True)

    la, lb = SRC_LABELS[task]
    if task == "medical":
        la = sub.upper() if sub in ("pet", "spect") else \
            ("SPECT" if stem.startswith("spect") else "PET")
    a_path = os.path.join(BENCH, "inputs", task, "colorA", fn)
    if not os.path.exists(a_path):
        a_path = os.path.join(BENCH, "inputs", task, "A", fn)
    srcA = Image.open(a_path).convert("RGB"); ref = srcA.size
    srcB = load_rgb(os.path.join(BENCH, "inputs", task, "B", fn), ref)

    panels = [(la, srcA), (lb, srcB)]
    for disp, folder in PARAMS[args.param]:
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

    n = len(annotated)
    ncols = args.ncols if args.ncols > 0 else (2 if n <= 4 else 3)
    ncols = min(ncols, n); nrows = int(np.ceil(n / ncols))
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
    out_png = os.path.join(out_base, f"fig_{args.param}_{task}{('_'+sub) if sub else ''}.png")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight"); plt.close(fig)
    print(f"[{args.param}/{task}{('/'+sub) if sub else ''}] {n} panels {nrows}x{ncols} -> {out_png}")


if __name__ == "__main__":
    main()
