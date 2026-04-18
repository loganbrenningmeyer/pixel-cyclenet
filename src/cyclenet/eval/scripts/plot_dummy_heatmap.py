import argparse
from pathlib import Path

import pandas as pd

from cyclenet.eval.plotting.heatmap import plot_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and save a dummy FID heatmap over noise strength and CFG weight."
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/heatmaps/dummy_fid_heatmap.pdf"),
        help="Output path for the generated heatmap image.",
    )
    return parser.parse_args()


def build_dummy_fid_dataframe() -> pd.DataFrame:
    cfg_weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    noise_strengths = [0.1, 0.2, 0.3, 0.4, 0.5]

    rows = []
    for strength in noise_strengths:
        for cfg in cfg_weights:
            # Simple hand-crafted pattern: best region near moderate cfg/strength,
            # worse values toward the edges.
            fid = (
                34.0
                + 18.0 * abs(strength - 0.25)
                + 3.2 * abs(cfg - 3.0)
                + 5.0 * max(0.0, strength - 0.35)
            )
            rows.append(
                {
                    "noise_strength": strength,
                    "cfg_weight": cfg,
                    "real_fid": round(fid, 2),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    df = build_dummy_fid_dataframe()
    grid_df = df.pivot(index="noise_strength", columns="cfg_weight", values="real_fid")

    plot_heatmap(
        grid_df=grid_df,
        title="Dummy FID Heatmap",
        xlabel="CFG Weight",
        ylabel="Noise Strength",
        save_path=args.out_path,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        cbar_label="FID",
    )

    print(f"Saved heatmap to {args.out_path}")


if __name__ == "__main__":
    main()
