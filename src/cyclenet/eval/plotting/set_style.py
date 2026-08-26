import matplotlib as mpl

COLORS = {
    "sim": "#6b7280",
    "translated": "#7c7c7c",
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
    "oem_only": "#68ACD6",
    "oem_only_rgb_only_spade": "#5B6FC0",
    "oem_only_rgb_only_spade_mid_skips": "#9C6ADE",
    "oem_only_seg_only": "#E3B23C",
    "oem_only_seg_only_spade": "#F28E2B",
}

MODEL_NAMES = {
    "oem_only": "RGB",
    "oem_only_rgb_only_spade": "RGB + SPADE",
    "oem_only_rgb_only_spade_mid_skips": "RGB + SPADE (BN Only)",
    "oem_only_seg_only": "Seg",
    "oem_only_seg_only_spade": "Seg + SPADE",
}

CLASS_NAMES = {
    "bareland": "Bareland",
    "rangeland": "Rangeland",
    "developed_space": "Developed Space",
    "road": "Road",
    "trees": "Trees",
    "water": "Water",
    "agriculture_land": "Agriculture Land",
    "buildings": "Buildings",
}

CLASS_LABELS = {
    1: "bareland",
    2: "rangeland",
    3: "developed_space",
    4: "road",
    5: "trees",
    6: "water",
    7: "agriculture_land",
    8: "buildings",
}

CLASS_COLORS = {
    "bareland": "#ac4848",
    "rangeland": "#7af289",
    "developed_space": "#bebebe",
    "road": "#e6e6e6",
    "trees": "#519855",
    "water": "#6191f2",
    "agriculture_land": "#eac82b",
    "buildings": "#f26161",
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
