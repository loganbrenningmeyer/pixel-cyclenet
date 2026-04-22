#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


MERGE_KEYS = ["step", "noise_strength", "cfg_weight"]
REQUIRED_LPIPS_COLUMNS = set(MERGE_KEYS) | {"lpips_mean", "lpips_std"}
REQUIRED_FID_COLUMNS = set(MERGE_KEYS) | {"fid"}
REQUIRED_DEEPLAB_FD_COLUMNS = set(MERGE_KEYS) | {"deeplab_fd"}


def validate_columns(df: pd.DataFrame, required_columns: set[str], csv_label: str) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{csv_label} CSV is missing required columns: {', '.join(missing)}")


def load_checkpoint_metrics(
    checkpoint_name: str,
    train_split: str,
    lpips_csv_path: str | Path,
    fid_csv_path: str | Path,
    deeplab_fd_csv_path: str | Path,
) -> pd.DataFrame:
    lpips_csv_path = Path(lpips_csv_path)
    fid_csv_path = Path(fid_csv_path)
    deeplab_fd_csv_path = Path(deeplab_fd_csv_path)

    for path, label in [
        (lpips_csv_path, "LPIPS"),
        (fid_csv_path, "FID"),
        (deeplab_fd_csv_path, "DeepLab FD"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} CSV does not exist: {path}")

    lpips_df = pd.read_csv(lpips_csv_path)
    fid_df = pd.read_csv(fid_csv_path)
    deeplab_df = pd.read_csv(deeplab_fd_csv_path)

    validate_columns(lpips_df, REQUIRED_LPIPS_COLUMNS, csv_label="LPIPS stats")
    validate_columns(fid_df, REQUIRED_FID_COLUMNS, csv_label="FID stats")
    validate_columns(deeplab_df, REQUIRED_DEEPLAB_FD_COLUMNS, csv_label="DeepLab FD stats")

    merged = lpips_df[list(MERGE_KEYS) + ["lpips_mean", "lpips_std"]].merge(
        fid_df[list(MERGE_KEYS) + ["fid"]],
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        deeplab_df[list(MERGE_KEYS) + ["deeplab_fd"]],
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(
            f"No shared `(step, noise_strength, cfg_weight)` rows were found across "
            f"{lpips_csv_path}, {fid_csv_path}, and {deeplab_fd_csv_path}."
        )

    merged["checkpoint_name"] = checkpoint_name
    merged["train_split"] = train_split
    return merged.sort_values(MERGE_KEYS).reset_index(drop=True)


def compute_pareto_mask(df: pd.DataFrame, metric_cols: list[str]) -> pd.Series:
    """
    Returns a boolean mask where True means the row is non-dominated under
    minimization of every metric in `metric_cols`.
    """
    values = df[metric_cols].to_numpy(dtype=float)
    is_pareto = [True] * len(df)

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            if (values[j] <= values[i]).all() and (values[j] < values[i]).any():
                is_pareto[i] = False
                break

    return pd.Series(is_pareto, index=df.index)


def select_operating_point(
    checkpoint_df: pd.DataFrame,
    selection_mode: str,
) -> pd.Series:
    mode = selection_mode.lower()

    if mode == "deeplab_lpips_pareto_then_fid":
        candidates = checkpoint_df.loc[checkpoint_df["is_pareto_deeplab_lpips"]].copy()
        if candidates.empty:
            raise ValueError("No DeepLab/LPIPS Pareto candidates were found.")
        return candidates.sort_values(
            ["fid", "deeplab_fd", "lpips_mean", "noise_strength", "cfg_weight"],
            ascending=[True, True, True, True, True],
        ).iloc[0]

    if mode == "three_metric_pareto_then_deeplab":
        candidates = checkpoint_df.loc[checkpoint_df["is_pareto_3d"]].copy()
        if candidates.empty:
            raise ValueError("No 3-metric Pareto candidates were found.")
        return candidates.sort_values(
            ["deeplab_fd", "fid", "lpips_mean", "noise_strength", "cfg_weight"],
            ascending=[True, True, True, True, True],
        ).iloc[0]

    if mode == "fid_deeplab_pareto_then_lpips":
        candidates = checkpoint_df.loc[checkpoint_df["is_pareto_fid_deeplab"]].copy()
        if candidates.empty:
            raise ValueError("No FID/DeepLab Pareto candidates were found.")
        return candidates.sort_values(
            ["lpips_mean", "deeplab_fd", "fid", "noise_strength", "cfg_weight"],
            ascending=[True, True, True, True, True],
        ).iloc[0]

    if mode == "best_deeplab_fd":
        return checkpoint_df.nsmallest(1, "deeplab_fd").iloc[0]

    raise ValueError(
        f"Unsupported selection_mode '{selection_mode}'. "
        "Expected 'deeplab_lpips_pareto_then_fid', 'three_metric_pareto_then_deeplab', "
        "'fid_deeplab_pareto_then_lpips', or 'best_deeplab_fd'."
    )


def summarize_checkpoint(
    checkpoint_df: pd.DataFrame,
    selection_mode: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    checkpoint_df = checkpoint_df.copy()
    checkpoint_df["is_pareto_deeplab_lpips"] = compute_pareto_mask(
        checkpoint_df,
        ["deeplab_fd", "lpips_mean"],
    )
    checkpoint_df["is_pareto_fid_deeplab"] = compute_pareto_mask(
        checkpoint_df,
        ["fid", "deeplab_fd"],
    )
    checkpoint_df["is_pareto_3d"] = compute_pareto_mask(
        checkpoint_df,
        ["fid", "lpips_mean", "deeplab_fd"],
    )

    best_fid_row = checkpoint_df.nsmallest(1, "fid").iloc[0]
    best_deeplab_row = checkpoint_df.nsmallest(1, "deeplab_fd").iloc[0]
    best_lpips_row = checkpoint_df.nsmallest(1, "lpips_mean").iloc[0]
    selected_row = select_operating_point(checkpoint_df, selection_mode=selection_mode)
    checkpoint_df["is_selected"] = checkpoint_df.index == selected_row.name

    summary_row = {
        "checkpoint_name": str(selected_row["checkpoint_name"]),
        "train_split": str(selected_row["train_split"]),
        "step": int(selected_row["step"]),
        "n_settings": int(len(checkpoint_df)),
        "pareto_deeplab_lpips_count": int(checkpoint_df["is_pareto_deeplab_lpips"].sum()),
        "pareto_fid_deeplab_count": int(checkpoint_df["is_pareto_fid_deeplab"].sum()),
        "pareto_3d_count": int(checkpoint_df["is_pareto_3d"].sum()),
        "selection_mode": selection_mode,
        "best_fid": float(best_fid_row["fid"]),
        "best_fid_gamma": float(best_fid_row["noise_strength"]),
        "best_fid_cfg_weight": float(best_fid_row["cfg_weight"]),
        "best_fid_lpips": float(best_fid_row["lpips_mean"]),
        "best_fid_deeplab_fd": float(best_fid_row["deeplab_fd"]),
        "best_deeplab_fd": float(best_deeplab_row["deeplab_fd"]),
        "best_deeplab_gamma": float(best_deeplab_row["noise_strength"]),
        "best_deeplab_cfg_weight": float(best_deeplab_row["cfg_weight"]),
        "best_deeplab_lpips": float(best_deeplab_row["lpips_mean"]),
        "best_deeplab_fid": float(best_deeplab_row["fid"]),
        "best_lpips": float(best_lpips_row["lpips_mean"]),
        "best_lpips_gamma": float(best_lpips_row["noise_strength"]),
        "best_lpips_cfg_weight": float(best_lpips_row["cfg_weight"]),
        "best_lpips_fid": float(best_lpips_row["fid"]),
        "best_lpips_deeplab_fd": float(best_lpips_row["deeplab_fd"]),
        "selected_gamma": float(selected_row["noise_strength"]),
        "selected_cfg_weight": float(selected_row["cfg_weight"]),
        "selected_fid": float(selected_row["fid"]),
        "selected_lpips": float(selected_row["lpips_mean"]),
        "selected_deeplab_fd": float(selected_row["deeplab_fd"]),
    }
    return summary_row, checkpoint_df


def build_latex_summary_table(summary_df: pd.DataFrame, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    table_df = summary_df[
        [
            "checkpoint_name",
            "train_split",
            "best_fid",
            "best_fid_gamma",
            "best_fid_cfg_weight",
            "best_deeplab_fd",
            "best_deeplab_gamma",
            "best_deeplab_cfg_weight",
            "selected_gamma",
            "selected_cfg_weight",
            "selected_fid",
            "selected_lpips",
            "selected_deeplab_fd",
        ]
    ].copy()
    table_df = table_df.rename(
        columns={
            "checkpoint_name": "Checkpoint",
            "train_split": "Split",
            "best_fid": "Best FID",
            "best_fid_gamma": "Best FID $\\gamma$",
            "best_fid_cfg_weight": "Best FID $w$",
            "best_deeplab_fd": "Best DeepLab FD",
            "best_deeplab_gamma": "Best DeepLab $\\gamma$",
            "best_deeplab_cfg_weight": "Best DeepLab $w$",
            "selected_gamma": "Selected $\\gamma$",
            "selected_cfg_weight": "Selected $w$",
            "selected_fid": "Selected FID",
            "selected_lpips": "Selected LPIPS",
            "selected_deeplab_fd": "Selected DeepLab FD",
        }
    )

    latex = table_df.to_latex(
        index=False,
        escape=False,
        float_format=lambda value: f"{value:.3f}",
    )
    save_path.write_text(latex)


def main() -> None:
    # Output directory for merged per-setting metrics and checkpoint summary tables.
    out_dir = Path("/tmp/checkpoint_metric_analysis")

    # Operating-point selection rule:
    # - `deeplab_lpips_pareto_then_fid`: default segmentation-oriented rule
    # - `fid_deeplab_pareto_then_lpips`: realism/task-aware alignment first,
    #   then lowest preservation cost among those candidates
    # - `three_metric_pareto_then_deeplab`: stricter 3-metric non-dominated rule
    # - `best_deeplab_fd`: purely task-aware minimum
    selection_mode = "deeplab_lpips_pareto_then_fid"

    # Per-checkpoint metric CSVs. Each entry should point at the LPIPS, FID, and
    # DeepLab FD CSVs produced from the same translated `step-*` directory.
    checkpoint_specs = [
        {
            "checkpoint_name": "2.5k",
            "train_split": "OEM only",
            "lpips_csv_path": "/path/to/step-2500/lpips_stats.csv",
            "fid_csv_path": "/path/to/step-2500/fid_stats.csv",
            "deeplab_fd_csv_path": "/path/to/step-2500/deeplab_fd_stats.csv",
        },
        {
            "checkpoint_name": "10k",
            "train_split": "All real",
            "lpips_csv_path": "/path/to/step-10000/lpips_stats.csv",
            "fid_csv_path": "/path/to/step-10000/fid_stats.csv",
            "deeplab_fd_csv_path": "/path/to/step-10000/deeplab_fd_stats.csv",
        },
        {
            "checkpoint_name": "20k",
            "train_split": "All real",
            "lpips_csv_path": "/path/to/step-20000/lpips_stats.csv",
            "fid_csv_path": "/path/to/step-20000/fid_stats.csv",
            "deeplab_fd_csv_path": "/path/to/step-20000/deeplab_fd_stats.csv",
        },
        {
            "checkpoint_name": "30k",
            "train_split": "All real",
            "lpips_csv_path": "/path/to/step-30000/lpips_stats.csv",
            "fid_csv_path": "/path/to/step-30000/fid_stats.csv",
            "deeplab_fd_csv_path": "/path/to/step-30000/deeplab_fd_stats.csv",
        },
    ]

    merged_dfs: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for spec in checkpoint_specs:
        checkpoint_df = load_checkpoint_metrics(**spec)
        summary_row, annotated_df = summarize_checkpoint(
            checkpoint_df=checkpoint_df,
            selection_mode=selection_mode,
        )
        merged_dfs.append(annotated_df)
        summary_rows.append(summary_row)

    all_metrics_df = pd.concat(merged_dfs, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("step").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_csv_path = out_dir / "checkpoint_metrics_merged.csv"
    summary_csv_path = out_dir / "checkpoint_summary.csv"
    latex_table_path = out_dir / "checkpoint_summary.tex"

    all_metrics_df.to_csv(merged_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    build_latex_summary_table(summary_df, save_path=latex_table_path)

    print(f"Saved merged checkpoint metrics CSV to {merged_csv_path}")
    print(f"Saved checkpoint summary CSV to {summary_csv_path}")
    print(f"Saved checkpoint summary LaTeX table to {latex_table_path}")


if __name__ == "__main__":
    main()
