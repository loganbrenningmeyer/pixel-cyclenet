import matplotlib as mpl

COLORS = {
    "sim": "#6b7280",
    "translated": "#2563eb",
    "real": "#dc2626",
    "highlight": "#111827",
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
