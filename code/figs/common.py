"""Shared paths, names and plot style for the result figures.

Numbers are read only from the result tables (results/, or FIG_SRC for regenerated tables).
Palette: orange = calibration / training-platform side, teal = disjoint reference / discrimination, grey otherwise.
"""
import os
import pathlib

import matplotlib as mpl

ROOT = pathlib.Path(__file__).resolve().parents[2]          # repository root
TAB = pathlib.Path(os.environ.get("FIG_SRC", ROOT / "results"))
OUT = pathlib.Path(os.environ.get("OUT_DIR", ROOT / "out"))
PREVIEW = OUT / "preview"
OUT.mkdir(parents=True, exist_ok=True)
PREVIEW.mkdir(parents=True, exist_ok=True)

NAME = {"B0'": "B0′"}   # tables use the paper names; only the prime is typographic
ORDER = ["B0", "B0′", "B0+0", "C0", "C0+0", "C1", "M0", "SpecTTTra", "ArtifactNet", "Fakeprint LR"]
EXCLUDE = {x.strip() for x in os.environ.get("SAC_EXCLUDE", "").split(",") if x.strip()}
FINETUNED = {"B0", "B0′", "B0+0", "C0", "C0+0", "C1", "M0"}
# Family groups (for separators and marker shapes): B0 family, C0 family, C1, M0, public.
FAMILY = {"B0": "B0", "B0′": "B0", "B0+0": "B0", "C0": "C0", "C0+0": "C0", "C1": "C1", "M0": "M0",
          "SpecTTTra": "pub", "ArtifactNet": "pub", "Fakeprint LR": "pub"}
FAMILY_LABEL = {"B0": "B0 family", "C0": "C0 family", "C1": "C1", "M0": "M0", "pub": "public"}

COHORT = {
    "jamendo_B": "Jamendo-B", "ava_sm5": "AVA-SM", "audioset_sm": "AudioSet-SM",
    "openbmat_human": "AI-OpenBMAT-H (all)", "openbmat_human_musicbearing": "AI-OpenBMAT-H",
    "musdb18": "MUSDB18", "gtzan": "GTZAN", "ava_mo": "AVA-MO", "ava_mo_long": "AVA-MO (≥10 s)",
    "audioset_mo": "AudioSet-MO", "librispeech_tc": "LibriSpeech", "esc50": "ESC-50", "fma_dev800": "FMA-dev",
}

# Palette (matches fig0_overview_provisional.png)
ORANGE, ORANGE_FILL = "#B4541F", "#FBEEE4"
TEAL, TEAL_FILL = "#1D6E78", "#E3F0F1"
GREY, GREY_L, GREY_FILL = "#555555", "#BBBBBB", "#F1F1F1"
INK = "#222222"

COL_W_IN = 3.33     # sigconf column width (inches)
TEXT_W_IN = 7.0     # sigconf text width (inches)
FPR_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]


def style():
    """Sans-serif, ≥7 pt at final size, embedded TrueType, light grid."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "lines.linewidth": 1.0, "lines.markersize": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "xtick.color": "#444444", "ytick.color": "#444444",
        "axes.labelcolor": INK, "text.color": INK,
        "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
        "mathtext.fontset": "dejavusans",
    })


def fpr_axis(ax, lo=0.07, hi=110, ticks=FPR_TICKS, label_floor=None, axis="y"):
    """Log FPR axis with dense ticks (0.1 … 100 %), light major grid, minor ticks shown."""
    import matplotlib.ticker as mt
    set_scale = ax.set_yscale if axis == "y" else ax.set_xscale
    set_lim = ax.set_ylim if axis == "y" else ax.set_xlim
    axobj = ax.yaxis if axis == "y" else ax.xaxis
    set_scale("log")
    set_lim(lo, hi)
    tk = [t for t in ticks if lo <= t <= hi]
    labels = [("%g" % t) for t in tk]
    if label_floor is not None:
        tk = [label_floor] + tk
        labels = ["0"] + labels
    axobj.set_major_locator(mt.FixedLocator(tk))
    axobj.set_major_formatter(mt.FixedFormatter(labels))
    axobj.set_minor_locator(mt.LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=20))
    axobj.set_minor_formatter(mt.NullFormatter())
    ax.grid(axis=axis, which="major", zorder=0)


def alpha_band(ax, alpha=2.0, mult=3.0, axis="y", label=True, where="right"):
    """Dashed nominal-α line and a light band up to mult·α."""
    if axis == "y":
        ax.axhspan(alpha, mult * alpha, color="#F3F3F3", zorder=0, lw=0)
        ax.axhline(alpha, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.axhline(mult * alpha, color=GREY_L, lw=0.5, ls=(0, (2, 2)), zorder=1)
    else:
        ax.axvspan(alpha, mult * alpha, color="#F3F3F3", zorder=0, lw=0)
        ax.axvline(alpha, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.axvline(mult * alpha, color=GREY_L, lw=0.5, ls=(0, (2, 2)), zorder=1)


def name(det):
    n = NAME.get(det)
    return None if n in EXCLUDE else n


def save(fig, stem, preview_dpi=300):
    """Vector PDF (fonts embedded, no metadata) to the manuscript folder + 300 dpi PNG previews
    (colour and grayscale) to figs_src_v2/preview/ for visual checking."""
    path = OUT / f"{stem}.pdf"
    meta = {"Creator": None, "Producer": None, "Author": None, "Title": None}
    fig.savefig(path, metadata=meta, bbox_inches=None)      # exact figsize: LaTeX will not rescale the fonts
    png = PREVIEW / f"{stem}.png"
    fig.savefig(png, dpi=preview_dpi, bbox_inches=None)
    try:
        from PIL import Image
        Image.open(png).convert("L").save(PREVIEW / f"{stem}_gray.png")
    except Exception as e:  # noqa: BLE001
        print("grayscale preview failed:", e)
    print("wrote", (path.relative_to(ROOT) if str(path).startswith(str(ROOT)) else path), "and", (png.relative_to(ROOT) if str(png).startswith(str(ROOT)) else png))
