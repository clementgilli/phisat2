"""
Shared matplotlib style for every figure in the paper.

Import `apply_style()` once at the top of each plotting script, then use
COLOR_MIM / COLOR_TERRAMIND (or PALETTE) for series colors and the
CVPR_TEXTWIDTH_IN / CVPR_COLUMNWIDTH_IN constants to size figures. Keeping
every script on this one module means a single edit here (e.g. swapping
COLOR_MIM to match Figure 1 exactly) propagates everywhere.
"""
import matplotlib as mpl

# ---- CVPR page geometry (cvpr.sty, US letter, two-column) -----------
CVPR_TEXTWIDTH_IN = 6.75       # full page width -- figures spanning both columns (figure*)
CVPR_COLUMNWIDTH_IN = 3.25     # single-column width

# ---- Shared palette ---------------------------------------------------
# TODO: replace with the *exact* hex used in Figure 1's PowerPoint blocks
# (right-click shape -> Format Shape -> Fill -> More Colors -> shows hex).
# Matching these two colors to Fig.1's "frozen" (blue) / "trainable"
# (red/pink) blocks makes the whole paper read as one visual language
# instead of "figure 1 was made by someone else than figures 3/5/9".
COLOR_MIM = "mediumblue"          # placeholder -- match Fig.1 frozen/blue
COLOR_TERRAMIND = "crimson"    # placeholder -- match Fig.1 trainable/red-pink
PALETTE = [COLOR_MIM, COLOR_TERRAMIND, "#059669", "#7c3aed", "#ea580c", "#0891b2"]

LINESTYLE_BEFORE_DA = ":"
LINESTYLE_REFERENCE = "--"
LINESTYLE_AFTER_DA = "-"


def apply_style():
    mpl.rcParams.update({
        # Serif to match the CVPR body text (Times-like), not the
        # matplotlib default sans -- this is the single biggest thing
        # that makes a figure look "pasted from a slide" vs. "typeset".
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        # Small base sizes: these figures are shrunk to ~6.75in wide in
        # the PDF, so anything above ~9-10pt here reads oversized on the
        # page. Tuned to stay legible at print size, not on-screen size.
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        # Flat, quiet axes -- consistent with Figure 1's uncluttered style
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e5e5e5",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,

        # Lines / markers
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,

        # Vector export with embedded (non-Type3) fonts -- some venues'
        # PDF checkers flag Type3 fonts as "not embedded"/unsearchable.
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })