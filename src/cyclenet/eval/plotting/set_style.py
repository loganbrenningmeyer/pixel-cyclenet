import matplotlib as mpl

COLORS = {
    "sim": "#6b7280",
    "translated": "#2563eb",
    "real": "#000000",
    "highlight": "#111827",
}

CHECKPOINT_MARKERS = {
    10000: "o",
    20000: "D",
    30000: "^",
    40000: "s",
    50000: "p",
}

MODEL_COLORS = {
    "oem_only": "#3B82F6",                     # RGB, No SPADE
    "oem_only_rgb_only_spade": "#14B8A6",     # RGB + SPADE
    "oem_only_rgb_only_spade_mid_skips": "#8B5CF6",  # RGB + SPADE, Mid-Skips
    "oem_only_seg_only": "#E9C46A",           # Seg, No SPADE
    "oem_only_seg_only_spade": "#F28E2B",     # Seg + SPADE
}


MODEL_NAMES = {
    "oem_only": "RGB, No SPADE",
    "oem_only_rgb_only_spade": "RGB + SPADE",
    "oem_only_rgb_only_spade_mid_skips": "RGB + SPADE, Mid-Skips",
    "oem_only_seg_only": "Seg, No SPADE",
    "oem_only_seg_only_spade": "Seg + SPADE",
}

def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Font / LaTeX rendering
        "text.usetex": True,
        "font.family": "serif",
        
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "font.size": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Figure styling
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.18,

        # Vector-friendly font embedding
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
