#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MERGE_KEYS = ["step", "noise_strength", "cfg_weight"]
REQUIRED_LPIPS_COLUMNS = set(MERGE_KEYS) | {"lpips_mean", "lpips_std"}
REQUIRED_FID_COLUMNS = set(MERGE_KEYS) | {"fid"}
REQUIRED_DEEPLAB_FD_COLUMNS = set(MERGE_KEYS) | {"deeplab_fd"}
REQUIRED_BOUNDARY_ALIGN_COLUMNS = set(MERGE_KEYS) | {
    "boundary_edge_ratio_mean",
    "boundary_edge_inverse_ratio_mean",
    "boundary_edge_contrast_mean",
}

PARETO_METRIC_SPECS = {
    "deeplab_lpips": {
        "metric_cols": ["deeplab_fd", "lpips_mean"],
        "pareto_col": "is_pareto_deeplab_lpips",
        "candidate_col": "is_candidate_deeplab_lpips",
        "pareto_count_col": "pareto_deeplab_lpips_count",
        "candidate_count_col": "candidate_deeplab_lpips_count",
    },
    "fid_lpips": {
        "metric_cols": ["fid", "lpips_mean"],
        "pareto_col": "is_pareto_fid_lpips",
        "candidate_col": "is_candidate_fid_lpips",
        "pareto_count_col": "pareto_fid_lpips_count",
        "candidate_count_col": "candidate_fid_lpips_count",
    },
    "fid_deeplab": {
        "metric_cols": ["fid", "deeplab_fd"],
        "pareto_col": "is_pareto_fid_deeplab",
        "candidate_col": "is_candidate_fid_deeplab",
        "pareto_count_col": "pareto_fid_deeplab_count",
        "candidate_count_col": "candidate_fid_deeplab_count",
    },
    "deeplab_boundary": {
        "metric_cols": ["deeplab_fd", "boundary_edge_inverse_ratio_mean"],
        "pareto_col": "is_pareto_deeplab_boundary",
        "candidate_col": "is_candidate_deeplab_boundary",
        "pareto_count_col": "pareto_deeplab_boundary_count",
        "candidate_count_col": "candidate_deeplab_boundary_count",
    },
    "fid_boundary": {
        "metric_cols": ["fid", "boundary_edge_inverse_ratio_mean"],
        "pareto_col": "is_pareto_fid_boundary",
        "candidate_col": "is_candidate_fid_boundary",
        "pareto_count_col": "pareto_fid_boundary_count",
        "candidate_count_col": "candidate_fid_boundary_count",
    },
    "3d": {
        "metric_cols": ["fid", "lpips_mean", "deeplab_fd"],
        "pareto_col": "is_pareto_3d",
        "candidate_col": "is_candidate_3d",
        "pareto_count_col": "pareto_3d_count",
        "candidate_count_col": "candidate_3d_count",
    },
    "3d_boundary": {
        "metric_cols": ["fid", "deeplab_fd", "boundary_edge_inverse_ratio_mean"],
        "pareto_col": "is_pareto_3d_boundary",
        "candidate_col": "is_candidate_3d_boundary",
        "pareto_count_col": "pareto_3d_boundary_count",
        "candidate_count_col": "candidate_3d_boundary_count",
    },
}

SELECTION_MODE_SPECS = {
    "fid_lpips_pareto_then_deeplab": {
        "pareto_key": "fid_lpips",
        "sort_cols": ["deeplab_fd", "fid", "lpips_mean", "noise_strength", "cfg_weight"],
    },
    "deeplab_lpips_pareto_then_fid": {
        "pareto_key": "deeplab_lpips",
        "sort_cols": ["fid", "deeplab_fd", "lpips_mean", "noise_strength", "cfg_weight"],
    },
    "three_metric_pareto_then_deeplab": {
        "pareto_key": "3d",
        "sort_cols": ["deeplab_fd", "fid", "lpips_mean", "noise_strength", "cfg_weight"],
    },
    "fid_deeplab_pareto_then_lpips": {
        "pareto_key": "fid_deeplab",
        "sort_cols": ["lpips_mean", "deeplab_fd", "fid", "noise_strength", "cfg_weight"],
    },
    "best_deeplab_fd": {
        "direct_metric": "deeplab_fd",
    },
    "deeplab_boundary_pareto_then_fid": {
        "pareto_key": "deeplab_boundary",
        "sort_cols": [
            "fid",
            "deeplab_fd",
            "boundary_edge_inverse_ratio_mean",
            "noise_strength",
            "cfg_weight",
        ],
    },
    "deeplab_boundary_pareto_then_knee": {
        "pareto_key": "deeplab_boundary",
        "selection_kind": "pareto_knee",
        "knee_metric_cols": ["boundary_edge_inverse_ratio_mean", "deeplab_fd"],
    },
    "fid_boundary_pareto_then_knee": {
        "pareto_key": "fid_boundary",
        "selection_kind": "pareto_knee",
        "knee_metric_cols": ["boundary_edge_inverse_ratio_mean", "fid"],
    },
    "fid_deeplab_boundary_pareto_then_knee": {
        "pareto_key": "3d_boundary",
        "selection_kind": "pareto_ideal",
        "ideal_metric_cols": ["boundary_edge_inverse_ratio_mean", "fid", "deeplab_fd"],
    },
}

DEFAULT_SELECTION_MODES = [
    "fid_lpips_pareto_then_deeplab",
    "fid_deeplab_pareto_then_lpips",
    "deeplab_lpips_pareto_then_fid",
    "deeplab_boundary_pareto_then_fid",
    "deeplab_boundary_pareto_then_knee",
    "fid_boundary_pareto_then_knee",
    "fid_deeplab_boundary_pareto_then_knee",
]


def validate_columns(df: pd.DataFrame, required_columns: set[str], csv_label: str) -> None:
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{csv_label} CSV is missing required columns: {', '.join(missing)}")


def load_checkpoint_metrics(
    checkpoint_name: str,
    lpips_csv_path: str | Path,
    fid_csv_path: str | Path,
    deeplab_fd_csv_path: str | Path,
    boundary_align_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    lpips_csv_path = Path(lpips_csv_path)
    fid_csv_path = Path(fid_csv_path)
    deeplab_fd_csv_path = Path(deeplab_fd_csv_path)
    boundary_align_csv_path = (
        Path(boundary_align_csv_path) if boundary_align_csv_path is not None else None
    )

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

    if boundary_align_csv_path is not None:
        if not boundary_align_csv_path.exists():
            raise FileNotFoundError(f"Boundary align CSV does not exist: {boundary_align_csv_path}")
        boundary_df = pd.read_csv(boundary_align_csv_path)
        validate_columns(
            boundary_df,
            REQUIRED_BOUNDARY_ALIGN_COLUMNS,
            csv_label="Boundary edge alignment stats",
        )
        merged = merged.merge(
            boundary_df[
                list(MERGE_KEYS)
                + [
                    "boundary_edge_ratio_mean",
                    "boundary_edge_inverse_ratio_mean",
                    "boundary_edge_contrast_mean",
                ]
            ],
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


def compute_pareto_candidate_mask(
    df: pd.DataFrame,
    metric_cols: list[str],
    pareto_threshold_pct: float | None,
) -> pd.Series:
    exact_pareto_mask = compute_pareto_mask(df, metric_cols)

    if pareto_threshold_pct is None or pareto_threshold_pct <= 0:
        return exact_pareto_mask

    values = df[metric_cols].to_numpy(dtype=float)
    exact_front = values[exact_pareto_mask.to_numpy()]
    tolerance_factor = 1.0 + (pareto_threshold_pct / 100.0)
    is_candidate: list[bool] = []

    for row in values:
        row_is_candidate = False
        for front_point in exact_front:
            tolerance = np.where(
                front_point == 0.0,
                front_point,
                front_point * tolerance_factor,
            )
            if (row <= tolerance).all():
                row_is_candidate = True
                break
        is_candidate.append(row_is_candidate)

    return pd.Series(is_candidate, index=df.index)


def _normalized_knee_score(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns `(knee_distance, ideal_distance)` for minimization points.

    The knee distance is the Euclidean distance to the line through the
    normalized endpoint Pareto points. The ideal distance is a fallback / tie
    breaker measuring Euclidean distance to the normalized ideal point.
    """
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins
    safe_spans = np.where(spans > 0.0, spans, 1.0)
    normalized = (points - mins) / safe_spans

    ideal_distance = np.sqrt((normalized**2).sum(axis=1))

    if len(normalized) < 3:
        return np.zeros(len(normalized), dtype=float), ideal_distance

    start = normalized[0]
    end = normalized[-1]
    line = end - start
    line_norm = float(np.linalg.norm(line))

    if line_norm <= 1e-12:
        return np.zeros(len(normalized), dtype=float), ideal_distance

    rel = normalized - start
    proj_scale = (rel @ line) / float(line @ line)
    proj = np.outer(proj_scale, line)
    knee_distance = np.linalg.norm(rel - proj, axis=1)
    return knee_distance, ideal_distance


def _normalized_ideal_distance(points: np.ndarray) -> np.ndarray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins
    safe_spans = np.where(spans > 0.0, spans, 1.0)
    normalized = (points - mins) / safe_spans
    return np.sqrt((normalized**2).sum(axis=1))


def select_pareto_knee_point(
    checkpoint_df: pd.DataFrame,
    pareto_col: str,
    metric_cols: list[str],
) -> pd.Series:
    front = checkpoint_df.loc[checkpoint_df[pareto_col]].copy()
    if front.empty:
        raise ValueError(f"No exact Pareto-front points were found for column '{pareto_col}'.")

    if len(metric_cols) < 2:
        raise ValueError(
            f"Pareto-knee selection expects at least 2 metrics, got {metric_cols}."
        )

    front = front.sort_values(
        metric_cols + ["noise_strength", "cfg_weight"],
        ascending=[True] * (len(metric_cols) + 2),
    ).copy()

    points = front[metric_cols].to_numpy(dtype=float)
    knee_distance, ideal_distance = _normalized_knee_score(points)

    front["knee_distance"] = knee_distance
    front["ideal_distance"] = ideal_distance

    return front.sort_values(
        [
            "knee_distance",
            "ideal_distance",
            metric_cols[0],
            metric_cols[1],
            "noise_strength",
            "cfg_weight",
        ],
        ascending=[False, True, True, True, True, True],
    ).iloc[0]


def select_pareto_ideal_point(
    checkpoint_df: pd.DataFrame,
    pareto_col: str,
    metric_cols: list[str],
) -> pd.Series:
    front = checkpoint_df.loc[checkpoint_df[pareto_col]].copy()
    if front.empty:
        raise ValueError(f"No exact Pareto-front points were found for column '{pareto_col}'.")

    if len(metric_cols) < 2:
        raise ValueError(
            f"Pareto-ideal selection expects at least 2 metrics, got {metric_cols}."
        )

    front = front.sort_values(
        metric_cols + ["noise_strength", "cfg_weight"],
        ascending=[True] * (len(metric_cols) + 2),
    ).copy()

    points = front[metric_cols].to_numpy(dtype=float)
    front["ideal_distance"] = _normalized_ideal_distance(points)

    return front.sort_values(
        ["ideal_distance"] + metric_cols + ["noise_strength", "cfg_weight"],
        ascending=[True] * (len(metric_cols) + 3),
    ).iloc[0]


def selected_col_name(selection_mode: str) -> str:
    return f"is_selected__{selection_mode.lower()}"


def normalize_selection_modes(selection_modes: list[str]) -> list[str]:
    normalized_modes: list[str] = []

    for mode in selection_modes:
        mode_key = mode.lower()
        if mode_key not in SELECTION_MODE_SPECS:
            supported = ", ".join(sorted(SELECTION_MODE_SPECS))
            raise ValueError(
                f"Unsupported selection_mode '{mode}'. Supported modes: {supported}"
            )
        if mode_key not in normalized_modes:
            normalized_modes.append(mode_key)

    if not normalized_modes:
        raise ValueError("selection_modes cannot be empty.")

    return normalized_modes


def get_selection_mode_spec(selection_mode: str) -> dict[str, object]:
    mode_key = selection_mode.lower()
    if mode_key not in SELECTION_MODE_SPECS:
        supported = ", ".join(sorted(SELECTION_MODE_SPECS))
        raise ValueError(
            f"Unsupported selection_mode '{selection_mode}'. Supported modes: {supported}"
        )

    spec = dict(SELECTION_MODE_SPECS[mode_key])
    pareto_key = spec.get("pareto_key")
    if pareto_key is not None:
        pareto_spec = PARETO_METRIC_SPECS[str(pareto_key)]
        spec["pareto_col"] = pareto_spec["pareto_col"]
        spec["candidate_col"] = pareto_spec["candidate_col"]
    spec["selected_col"] = selected_col_name(mode_key)
    return spec


def annotate_pareto_columns(
    df: pd.DataFrame,
    pareto_threshold_pct: float | None,
) -> pd.DataFrame:
    annotated = df.copy()

    for spec in PARETO_METRIC_SPECS.values():
        annotated[spec["pareto_col"]] = compute_pareto_mask(
            annotated,
            metric_cols=spec["metric_cols"],
        )
        annotated[spec["candidate_col"]] = compute_pareto_candidate_mask(
            annotated,
            metric_cols=spec["metric_cols"],
            pareto_threshold_pct=pareto_threshold_pct,
        )

    annotated["pareto_threshold_pct"] = (
        0.0 if pareto_threshold_pct is None else float(pareto_threshold_pct)
    )
    return annotated


def select_operating_point(
    checkpoint_df: pd.DataFrame,
    selection_mode: str,
) -> pd.Series:
    spec = get_selection_mode_spec(selection_mode)

    direct_metric = spec.get("direct_metric")
    if direct_metric is not None:
        return checkpoint_df.nsmallest(1, str(direct_metric)).iloc[0]

    selection_kind = spec.get("selection_kind")
    if selection_kind == "pareto_knee":
        return select_pareto_knee_point(
            checkpoint_df=checkpoint_df,
            pareto_col=str(spec["pareto_col"]),
            metric_cols=list(spec["knee_metric_cols"]),
        )
    if selection_kind == "pareto_ideal":
        return select_pareto_ideal_point(
            checkpoint_df=checkpoint_df,
            pareto_col=str(spec["pareto_col"]),
            metric_cols=list(spec["ideal_metric_cols"]),
        )

    candidate_col = str(spec["candidate_col"])
    candidates = checkpoint_df.loc[checkpoint_df[candidate_col]].copy()
    if candidates.empty:
        raise ValueError(
            f"No candidates were found for selection_mode '{selection_mode}'."
        )

    sort_cols = list(spec["sort_cols"])
    return candidates.sort_values(
        sort_cols,
        ascending=[True] * len(sort_cols),
    ).iloc[0]


def filter_candidates(
    df: pd.DataFrame,
    max_lpips: float | None = None,
    max_noise_strength: float | None = None,
    min_boundary_edge_ratio: float | None = None,
    max_boundary_edge_inverse_ratio: float | None = None,
) -> pd.DataFrame:
    filtered = df.copy()

    if max_lpips is not None:
        filtered = filtered.loc[filtered["lpips_mean"] <= max_lpips].copy()

    if max_noise_strength is not None:
        filtered = filtered.loc[filtered["noise_strength"] <= max_noise_strength].copy()

    if min_boundary_edge_ratio is not None:
        if "boundary_edge_ratio_mean" not in filtered.columns:
            raise ValueError(
                "min_boundary_edge_ratio was set, but boundary_edge_ratio_mean is not available. "
                "Provide boundary_align_csv_path in checkpoint_specs."
            )
        filtered = filtered.loc[filtered["boundary_edge_ratio_mean"] >= min_boundary_edge_ratio].copy()

    if max_boundary_edge_inverse_ratio is not None:
        if "boundary_edge_inverse_ratio_mean" not in filtered.columns:
            raise ValueError(
                "max_boundary_edge_inverse_ratio was set, but boundary_edge_inverse_ratio_mean "
                "is not available. Provide boundary_align_csv_path in checkpoint_specs."
            )
        filtered = filtered.loc[
            filtered["boundary_edge_inverse_ratio_mean"] <= max_boundary_edge_inverse_ratio
        ].copy()

    if filtered.empty:
        filters = []
        if max_lpips is not None:
            filters.append(f"lpips_mean <= {max_lpips}")
        if max_noise_strength is not None:
            filters.append(f"noise_strength <= {max_noise_strength}")
        if min_boundary_edge_ratio is not None:
            filters.append(f"boundary_edge_ratio_mean >= {min_boundary_edge_ratio}")
        if max_boundary_edge_inverse_ratio is not None:
            filters.append(
                f"boundary_edge_inverse_ratio_mean <= {max_boundary_edge_inverse_ratio}"
            )
        filter_str = ", ".join(filters) if filters else "no filters"
        raise ValueError(f"No candidates remain after applying filters: {filter_str}")

    return filtered.reset_index(drop=True)


def summarize_candidate_pool(
    candidate_df: pd.DataFrame,
    selection_modes: list[str],
    selection_scope: str,
    pareto_threshold_pct: float | None = None,
    max_lpips: float | None = None,
    max_noise_strength: float | None = None,
    min_boundary_edge_ratio: float | None = None,
    max_boundary_edge_inverse_ratio: float | None = None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    selection_modes = normalize_selection_modes(selection_modes)
    original_count = int(len(candidate_df))

    candidate_df = filter_candidates(
        candidate_df,
        max_lpips=max_lpips,
        max_noise_strength=max_noise_strength,
        min_boundary_edge_ratio=min_boundary_edge_ratio,
        max_boundary_edge_inverse_ratio=max_boundary_edge_inverse_ratio,
    )
    candidate_df = annotate_pareto_columns(
        candidate_df,
        pareto_threshold_pct=pareto_threshold_pct,
    )
    candidate_df["selection_scope"] = selection_scope

    selected_rows: dict[str, pd.Series] = {}
    for selection_mode in selection_modes:
        selected_col = selected_col_name(selection_mode)
        candidate_df[selected_col] = False
        selected_row = select_operating_point(candidate_df, selection_mode=selection_mode)
        candidate_df.loc[selected_row.name, selected_col] = True
        selected_rows[selection_mode] = selected_row

    candidate_df["is_selected"] = False
    if len(selection_modes) == 1:
        candidate_df["is_selected"] = candidate_df[selected_col_name(selection_modes[0])]

    best_fid_row = candidate_df.nsmallest(1, "fid").iloc[0]
    best_deeplab_row = candidate_df.nsmallest(1, "deeplab_fd").iloc[0]
    best_lpips_row = candidate_df.nsmallest(1, "lpips_mean").iloc[0]
    summary_rows: list[dict[str, object]] = []

    if candidate_df["checkpoint_name"].nunique() == 1:
        scope_checkpoint_name = str(candidate_df["checkpoint_name"].iloc[0])
        scope_step = int(candidate_df["step"].iloc[0])
    else:
        scope_checkpoint_name = ""
        scope_step = pd.NA

    base_summary = {
        "selection_scope": selection_scope,
        "checkpoint_name": scope_checkpoint_name,
        "step": scope_step,
        "n_settings_before_filter": original_count,
        "n_settings": int(len(candidate_df)),
        "n_checkpoints": int(candidate_df["checkpoint_name"].nunique()),
        "max_lpips": max_lpips,
        "max_noise_strength": max_noise_strength,
        "min_boundary_edge_ratio": min_boundary_edge_ratio,
        "max_boundary_edge_inverse_ratio": max_boundary_edge_inverse_ratio,
        "pareto_threshold_pct": 0.0 if pareto_threshold_pct is None else float(pareto_threshold_pct),
        "best_fid": float(best_fid_row["fid"]),
        "best_fid_checkpoint_name": str(best_fid_row["checkpoint_name"]),
        "best_fid_step": int(best_fid_row["step"]),
        "best_fid_gamma": float(best_fid_row["noise_strength"]),
        "best_fid_cfg_weight": float(best_fid_row["cfg_weight"]),
        "best_fid_lpips": float(best_fid_row["lpips_mean"]),
        "best_fid_deeplab_fd": float(best_fid_row["deeplab_fd"]),
        "best_deeplab_fd": float(best_deeplab_row["deeplab_fd"]),
        "best_deeplab_checkpoint_name": str(best_deeplab_row["checkpoint_name"]),
        "best_deeplab_step": int(best_deeplab_row["step"]),
        "best_deeplab_gamma": float(best_deeplab_row["noise_strength"]),
        "best_deeplab_cfg_weight": float(best_deeplab_row["cfg_weight"]),
        "best_deeplab_lpips": float(best_deeplab_row["lpips_mean"]),
        "best_deeplab_fid": float(best_deeplab_row["fid"]),
        "best_lpips": float(best_lpips_row["lpips_mean"]),
        "best_lpips_checkpoint_name": str(best_lpips_row["checkpoint_name"]),
        "best_lpips_step": int(best_lpips_row["step"]),
        "best_lpips_gamma": float(best_lpips_row["noise_strength"]),
        "best_lpips_cfg_weight": float(best_lpips_row["cfg_weight"]),
        "best_lpips_fid": float(best_lpips_row["fid"]),
        "best_lpips_deeplab_fd": float(best_lpips_row["deeplab_fd"]),
    }

    for pareto_spec in PARETO_METRIC_SPECS.values():
        base_summary[pareto_spec["pareto_count_col"]] = int(candidate_df[pareto_spec["pareto_col"]].sum())
        base_summary[pareto_spec["candidate_count_col"]] = int(
            candidate_df[pareto_spec["candidate_col"]].sum()
        )

    if "boundary_edge_ratio_mean" in candidate_df.columns:
        base_summary["best_boundary_edge_ratio"] = float(candidate_df["boundary_edge_ratio_mean"].max())
        base_summary["best_boundary_edge_inverse_ratio"] = float(
            candidate_df["boundary_edge_inverse_ratio_mean"].min()
        )
        best_boundary_row = candidate_df.nsmallest(1, "boundary_edge_inverse_ratio_mean").iloc[0]
        base_summary["best_boundary_checkpoint_name"] = str(best_boundary_row["checkpoint_name"])
        base_summary["best_boundary_step"] = int(best_boundary_row["step"])
        base_summary["best_boundary_gamma"] = float(best_boundary_row["noise_strength"])
        base_summary["best_boundary_cfg_weight"] = float(best_boundary_row["cfg_weight"])
        base_summary["best_boundary_fid"] = float(best_boundary_row["fid"])
        base_summary["best_boundary_deeplab_fd"] = float(best_boundary_row["deeplab_fd"])

    for selection_mode in selection_modes:
        spec = get_selection_mode_spec(selection_mode)
        selected_row = selected_rows[selection_mode]
        summary_row = dict(base_summary)
        summary_row.update(
            {
                "selection_mode": selection_mode,
                "selection_selected_col": str(spec["selected_col"]),
                "selected_checkpoint_name": str(selected_row["checkpoint_name"]),
                "selected_step": int(selected_row["step"]),
                "selected_gamma": float(selected_row["noise_strength"]),
                "selected_cfg_weight": float(selected_row["cfg_weight"]),
                "selected_fid": float(selected_row["fid"]),
                "selected_lpips": float(selected_row["lpips_mean"]),
                "selected_deeplab_fd": float(selected_row["deeplab_fd"]),
            }
        )

        candidate_col = spec.get("candidate_col")
        pareto_col = spec.get("pareto_col")
        if candidate_col is not None:
            summary_row["selection_candidate_col"] = str(candidate_col)
            summary_row["selection_candidate_count"] = int(candidate_df[str(candidate_col)].sum())
            summary_row["selected_in_candidate_pool"] = bool(selected_row[str(candidate_col)])
        else:
            summary_row["selection_candidate_col"] = ""
            summary_row["selection_candidate_count"] = pd.NA
            summary_row["selected_in_candidate_pool"] = pd.NA

        if pareto_col is not None:
            summary_row["selection_pareto_col"] = str(pareto_col)
            summary_row["selected_on_exact_pareto_front"] = bool(selected_row[str(pareto_col)])
        else:
            summary_row["selection_pareto_col"] = ""
            summary_row["selected_on_exact_pareto_front"] = pd.NA

        if "boundary_edge_ratio_mean" in candidate_df.columns:
            summary_row["selected_boundary_edge_ratio"] = float(selected_row["boundary_edge_ratio_mean"])
            summary_row["selected_boundary_edge_inverse_ratio"] = float(
                selected_row["boundary_edge_inverse_ratio_mean"]
            )

        summary_rows.append(summary_row)

    return summary_rows, candidate_df


def build_latex_summary_table(summary_df: pd.DataFrame, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    display_columns = [
        "selection_mode",
        "checkpoint_name",
        "selected_checkpoint_name",
        "selected_step",
        "pareto_threshold_pct",
        "selection_candidate_count",
        "selected_gamma",
        "selected_cfg_weight",
        "selected_fid",
        "selected_lpips",
        "selected_deeplab_fd",
    ]
    existing_columns = [column for column in display_columns if column in summary_df.columns]
    table_df = summary_df[existing_columns].copy().rename(
        columns={
            "selection_mode": "Selection Mode",
            "checkpoint_name": "Scope Checkpoint",
            "selected_checkpoint_name": "Selected Checkpoint",
            "selected_step": "Selected Step",
            "pareto_threshold_pct": "Pareto % Threshold",
            "selection_candidate_count": "Candidate Count",
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


def pareto_sweep(
    cyclenet_sim_dir: Path | str,
    steps: list[int],
    out_dir: Path | str,
    selection_modes: list[str] = [
        "fid_lpips_pareto_then_deeplab",
        "fid_deeplab_pareto_then_lpips",
        "deeplab_lpips_pareto_then_fid",
    ],
    pct_thresholds: list[float] = [0.0, 5.0, 10.0, 25.0],
    select_by_checkpoint: bool = False,
    max_lpips: float | None = None,
):
    # -------------------------
    # Prepare checkpoint data paths
    # -------------------------
    checkpoint_specs = [
        {
            "checkpoint_name": str(step),
            "lpips_csv_path": Path(cyclenet_sim_dir) / f"step-{step}" / "lpips_stats.csv",
            "fid_csv_path": Path(cyclenet_sim_dir) / f"step-{step}" / "fid_stats.csv",
            "deeplab_fd_csv_path": Path(cyclenet_sim_dir) / f"step-{step}" / "deeplab_fd_stats.csv",
            "boundary_align_csv_path": Path(cyclenet_sim_dir)
            / f"step-{step}"
            / "boundary_edge_align_stats.csv",
        }
        for step in steps
    ]

    # -------------------------
    # Merge checkpoint data
    # -------------------------
    merged_dfs: list[pd.DataFrame] = []
    for spec in checkpoint_specs:
        checkpoint_df = load_checkpoint_metrics(**spec)
        merged_dfs.append(checkpoint_df)

    all_metrics_df = pd.concat(merged_dfs, axis=0, ignore_index=True)

    # -------------------------
    # Select models for each pareto front % threshold
    # -------------------------
    for pareto_threshold_pct in pct_thresholds:

        if pareto_threshold_pct == 0.0:
            pct_out_dir = Path(out_dir) / "pareto-front"
        else:
            pct_out_dir = Path(out_dir) / f"{int(pareto_threshold_pct)}-pct-pareto-front"

        # -------------------------
        # Select model for each checkpoint
        # -------------------------
        if select_by_checkpoint:
            summary_rows: list[dict[str, object]] = []
            annotated_dfs: list[pd.DataFrame] = []

            for _, checkpoint_df in all_metrics_df.groupby("checkpoint_name", sort=False):
                checkpoint_summary_rows, annotated_df = summarize_candidate_pool(
                    candidate_df=checkpoint_df.reset_index(drop=True),
                    selection_modes=selection_modes,
                    selection_scope="checkpoint",
                    pareto_threshold_pct=pareto_threshold_pct,
                    max_lpips=max_lpips,
                )
                summary_rows.extend(checkpoint_summary_rows)
                annotated_dfs.append(annotated_df)

            all_metrics_df = pd.concat(annotated_dfs, axis=0, ignore_index=True)
        # -------------------------
        # Select model across all checkpoints
        # -------------------------
        else:
            summary_rows, all_metrics_df = summarize_candidate_pool(
                candidate_df=all_metrics_df,
                selection_modes=selection_modes,
                selection_scope="overall",
                pareto_threshold_pct=pareto_threshold_pct,
                max_lpips=max_lpips,
            )

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            sort_cols = [column for column in ["selection_mode", "step", "checkpoint_name"] if column in summary_df.columns]
            summary_df = summary_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

        # -------------------------
        # Save model selection csvs
        # -------------------------
        pct_out_dir.mkdir(parents=True, exist_ok=True)

        merged_csv_path = pct_out_dir / "checkpoint_metrics_merged.csv"
        summary_csv_path = pct_out_dir / "checkpoint_summary.csv"
        latex_table_path = pct_out_dir / "checkpoint_summary.tex"

        all_metrics_df.to_csv(merged_csv_path, index=False)
        summary_df.to_csv(summary_csv_path, index=False)
        build_latex_summary_table(summary_df, save_path=latex_table_path)

        print(f"Saved merged checkpoint metrics CSV to {merged_csv_path}")
        print(f"Saved checkpoint summary CSV to {summary_csv_path}")
        print(f"Saved checkpoint summary LaTeX table to {latex_table_path}")
