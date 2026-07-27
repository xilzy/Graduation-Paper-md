"""Shared publication font configuration for thesis figures."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path


def _bootstrap_mpl_cache() -> None:
    """Avoid an implicit recursive system-font scan on shared storage."""
    cache = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis_figure_fonts_mpl_v2"))
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    cache.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("matplotlib")
    cache_file = cache / f"fontlist-v{version}.json"
    if cache_file.exists():
        return

    def entry(
        filename: str,
        weight: int = 400,
        style: str = "normal",
        name: str = "DejaVu Sans",
    ) -> dict:
        return {
            "fname": f"fonts/ttf/{filename}",
            "index": 0,
            "name": name,
            "style": style,
            "variant": "normal",
            "weight": weight,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry",
        }

    payload = {
        "_version": version,
        "_FontManager__default_weight": "normal",
        "default_size": None,
        "defaultFamily": {"ttf": "DejaVu Sans", "afm": "Helvetica"},
        "afmlist": [],
        "ttflist": [
            entry("DejaVuSans.ttf"),
            entry("DejaVuSans-Bold.ttf", 700),
            entry("DejaVuSans-Oblique.ttf", 400, "oblique"),
            entry("DejaVuSans-BoldOblique.ttf", 700, "oblique"),
            entry("cmex10.ttf", name="cmex10"),
        ],
        "__class__": "FontManager",
    }
    cache_file.write_text(json.dumps(payload), encoding="utf-8")


_bootstrap_mpl_cache()

import matplotlib as mpl
from matplotlib import font_manager


DEFAULT_TIMES_DIR = Path(
    os.environ.get(
        "TIMES_NEW_ROMAN_DIR",
        "/ytech_m2v4_hdd/lizhongyin/.cache/fonts/times-new-roman",
    )
)


def setup_times_new_roman(font_dir: Path | str = DEFAULT_TIMES_DIR) -> str:
    """Register all Times New Roman faces and apply them to text and mathtext."""
    os.environ.pop("MPL_IGNORE_SYSTEM_FONTS", None)
    font_dir = Path(font_dir)
    paths = [
        font_dir / "Times.TTF",
        font_dir / "Timesbd.TTF",
        font_dir / "Timesi.TTF",
        font_dir / "Timesbi.TTF",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Times New Roman files are required. Missing: " + ", ".join(missing)
        )
    for path in paths:
        font_manager.fontManager.addfont(path)

    family = font_manager.FontProperties(fname=paths[0]).get_name()
    if family != "Times New Roman":
        raise RuntimeError(f"Expected Times New Roman, found {family}")

    mpl.rcParams.update(
        {
            "font.family": family,
            "font.serif": [family],
            "font.sans-serif": [family],
            "font.cursive": [family],
            "font.monospace": [family],
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "mathtext.cal": f"{family}:italic",
            "mathtext.sf": family,
            "mathtext.tt": family,
            "mathtext.fallback": None,
            "mathtext.default": "regular",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return family
