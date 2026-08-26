from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from cyclenet.eval.plotting.set_style import CLASS_NAMES, MODEL_NAMES


PER_CLASS_ORDER = list(CLASS_NAMES.keys())
COLUMN_LABELS = {
    "display_name": "Model",
    "model_name": "Model ID",
    "model_category": "Category",
    "cond_mode": "Cond.",
    "use_spade": "SPADE",
    "mid_skips_only": "BN Only",
    "checkpoint": "Checkpoint",
    "noise_strength": "$s$",
    "cfg_weight": "$w$",
    "num_seg_runs": "$N$ Runs",
    "fid": "FID",
    "deeplab_fd": "DeepLab-FD",
    "ber": "BER",
    "ber_std": "BER Std.",
    "lpips_mean": "LPIPS",
    "lpips_std": "LPIPS Std.",
    "miou_mean": "mIoU",
    "miou_std": "mIoU Std.",
    "delta_miou_vs_sim": "$\\Delta$ mIoU vs Sim",
    "miou_gap_closed": "Gap Closed",
    "pixel_acc_mean": "Pixel Acc.",
    "pixel_acc_std": "Pixel Acc. Std.",
    "lpips": "LPIPS",
    "miou": "mIoU",
    "pixel_acc": "Pixel Acc.",
}


def _display_model_name(model_name: str) -> str:
    display_name = MODEL_NAMES.get(model_name, "")
    return display_name if display_name else model_name.replace("_", " ")


def _parse_float(value: str | float | int) -> float:
    return float(value)


def _parse_int(value: str | float | int) -> int:
    return int(float(value))


def _format_bool_like(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "yes"
    if text in {"false", "0", "no"}:
        return "no"
    return str(value)


def _format_float(value: object, decimals: int = 3) -> str:
    if value == "":
        return ""
    return f"{float(value):.{decimals}f}"


def _format_mean_std(mean: object, std: object, decimals: int = 3) -> str:
    if mean == "" or std == "":
        return ""
    return f"{float(mean):.{decimals}f} $\\pm$ {float(std):.{decimals}f}"


def _escape_latex(text: object) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def _load_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV does not exist: {csv_path}")

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")
    return rows


def _load_selected_models(selected_models_csv: str | Path) -> list[dict[str, str]]:
    rows = _load_csv_rows(selected_models_csv)
    required = {
        "model_name",
        "cond_mode",
        "use_spade",
        "mid_skips_only",
        "checkpoint",
        "noise_strength",
        "cfg_weight",
        "lpips_stats_path",
        "fid_stats_path",
        "deeplab_fd_stats_path",
        "boundary_edge_align_stats_path",
    }
    missing = sorted(required - set(rows[0].keys()))
    if missing:
        raise ValueError(
            f"Selected-models CSV is missing required columns: {', '.join(missing)}"
        )
    return rows


def _find_matching_metric_row(
    rows: list[dict[str, str]],
    checkpoint: int,
    noise_strength: float,
    cfg_weight: float,
    csv_label: str,
) -> dict[str, str]:
    matches: list[dict[str, str]] = []
    for row in rows:
        try:
            row_step = _parse_int(row["step"])
            row_strength = _parse_float(row["noise_strength"])
            row_cfg = _parse_float(row["cfg_weight"])
        except KeyError as exc:
            raise ValueError(f"{csv_label} is missing expected key: {exc}") from exc

        if (
            row_step == checkpoint
            and abs(row_strength - noise_strength) <= 1e-8
            and abs(row_cfg - cfg_weight) <= 1e-8
        ):
            matches.append(row)

    if not matches:
        raise ValueError(
            f"No row in {csv_label} matched step={checkpoint}, "
            f"noise_strength={noise_strength}, cfg_weight={cfg_weight}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple rows in {csv_label} matched step={checkpoint}, "
            f"noise_strength={noise_strength}, cfg_weight={cfg_weight}."
        )
    return matches[0]


def _load_segmentation_summary(summary_csv_path: str | Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    rows = _load_csv_rows(summary_csv_path)
    by_model_name: dict[str, dict[str, str]] = {}
    for row in rows:
        model_name = str(row["model_name"])
        if model_name in by_model_name:
            raise ValueError(f"Duplicate segmentation summary row for model_name '{model_name}'.")
        by_model_name[model_name] = row

    if "sim" not in by_model_name:
        raise ValueError("Segmentation summary CSV is missing the sim baseline row.")
    return by_model_name, by_model_name["sim"]


def _collect_per_class_metrics(
    summary_row: dict[str, str],
    sim_row: dict[str, str],
    real_row: dict[str, str] | None = None,
) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for class_name in PER_CLASS_ORDER:
        mean_key = f"mean_iou_{class_name}"
        std_key = f"std_iou_{class_name}"
        if mean_key not in summary_row or std_key not in summary_row:
            continue
        translated_miou = _parse_float(summary_row[mean_key])
        out[f"miou_{class_name}"] = translated_miou
        out[f"miou_std_{class_name}"] = _parse_float(summary_row[std_key])
        if mean_key in sim_row:
            sim_miou = _parse_float(sim_row[mean_key])
            out[f"delta_miou_vs_sim_{class_name}"] = translated_miou - sim_miou
            out[f"miou_gap_closed_{class_name}"] = _compute_gap_closed(
                translated_miou=translated_miou,
                sim_miou=sim_miou,
                real_miou=(
                    _parse_float(real_row[mean_key])
                    if real_row is not None and mean_key in real_row
                    else None
                ),
            )
    return out


def _compute_gap_closed(
    translated_miou: float,
    sim_miou: float,
    real_miou: float | None,
) -> float | str:
    if real_miou is None:
        return ""
    denom = real_miou - sim_miou
    if abs(denom) <= 1e-12:
        return ""
    return (translated_miou - sim_miou) / denom


def build_final_results_rows(
    selected_models_csv: str | Path,
    segmentation_summary_csv: str | Path,
    include_baselines: bool = True,
) -> list[dict[str, object]]:
    selected_rows = _load_selected_models(selected_models_csv)
    segmentation_by_model, sim_row = _load_segmentation_summary(segmentation_summary_csv)
    real_row = segmentation_by_model.get("real")

    results: list[dict[str, object]] = []
    sim_miou = _parse_float(sim_row["mean_miou"])
    real_miou = _parse_float(real_row["mean_miou"]) if real_row is not None else None

    for selected in selected_rows:
        model_name = str(selected["model_name"])
        if model_name not in segmentation_by_model:
            raise ValueError(
                f"Selected model '{model_name}' is missing from segmentation summary CSV."
            )

        checkpoint = _parse_int(selected["checkpoint"])
        noise_strength = _parse_float(selected["noise_strength"])
        cfg_weight = _parse_float(selected["cfg_weight"])

        lpips_row = _find_matching_metric_row(
            _load_csv_rows(selected["lpips_stats_path"]),
            checkpoint=checkpoint,
            noise_strength=noise_strength,
            cfg_weight=cfg_weight,
            csv_label=f"LPIPS stats for {model_name}",
        )
        fid_row = _find_matching_metric_row(
            _load_csv_rows(selected["fid_stats_path"]),
            checkpoint=checkpoint,
            noise_strength=noise_strength,
            cfg_weight=cfg_weight,
            csv_label=f"FID stats for {model_name}",
        )
        deeplab_fd_row = _find_matching_metric_row(
            _load_csv_rows(selected["deeplab_fd_stats_path"]),
            checkpoint=checkpoint,
            noise_strength=noise_strength,
            cfg_weight=cfg_weight,
            csv_label=f"DeepLab-FD stats for {model_name}",
        )
        ber_row = _find_matching_metric_row(
            _load_csv_rows(selected["boundary_edge_align_stats_path"]),
            checkpoint=checkpoint,
            noise_strength=noise_strength,
            cfg_weight=cfg_weight,
            csv_label=f"Boundary-edge stats for {model_name}",
        )

        seg = segmentation_by_model[model_name]
        translated_miou = _parse_float(seg["mean_miou"])
        row: dict[str, object] = {
            "model_name": model_name,
            "display_name": _display_model_name(model_name),
            "model_category": seg["model_category"],
            "cond_mode": selected["cond_mode"],
            "use_spade": selected["use_spade"],
            "mid_skips_only": selected["mid_skips_only"],
            "checkpoint": checkpoint,
            "noise_strength": noise_strength,
            "cfg_weight": cfg_weight,
            "num_seg_runs": _parse_int(seg["num_sub_runs"]),
            "lpips_mean": _parse_float(lpips_row["lpips_mean"]),
            "lpips_std": _parse_float(lpips_row["lpips_std"]),
            "fid": _parse_float(fid_row["fid"]),
            "deeplab_fd": _parse_float(deeplab_fd_row["deeplab_fd"]),
            "ber": _parse_float(ber_row["boundary_edge_inverse_ratio_mean"]),
            "ber_std": _parse_float(ber_row["boundary_edge_inverse_ratio_std"]),
            "pixel_acc_mean": _parse_float(seg["mean_pixel_acc"]),
            "pixel_acc_std": _parse_float(seg["std_pixel_acc"]),
            "miou_mean": translated_miou,
            "miou_std": _parse_float(seg["std_miou"]),
            "delta_miou_vs_sim": translated_miou - sim_miou,
            "miou_gap_closed": _compute_gap_closed(
                translated_miou=translated_miou,
                sim_miou=sim_miou,
                real_miou=real_miou,
            ),
        }
        row.update(
            _collect_per_class_metrics(
                summary_row=seg,
                sim_row=sim_row,
                real_row=real_row,
            )
        )
        results.append(row)

    if include_baselines:
        for baseline_name in ["sim", "real"]:
            baseline = segmentation_by_model.get(baseline_name)
            if baseline is None:
                continue
            baseline_miou = _parse_float(baseline["mean_miou"])
            row = {
                "model_name": baseline_name,
                "display_name": _display_model_name(baseline_name),
                "model_category": baseline["model_category"],
                "cond_mode": "",
                "use_spade": "",
                "mid_skips_only": "",
                "checkpoint": "",
                "noise_strength": "",
                "cfg_weight": "",
                "num_seg_runs": _parse_int(baseline["num_sub_runs"]),
                "lpips_mean": "",
                "lpips_std": "",
                "fid": "",
                "deeplab_fd": "",
                "ber": "",
                "ber_std": "",
                "pixel_acc_mean": _parse_float(baseline["mean_pixel_acc"]),
                "pixel_acc_std": _parse_float(baseline["std_pixel_acc"]),
                "miou_mean": baseline_miou,
                "miou_std": _parse_float(baseline["std_miou"]),
                "delta_miou_vs_sim": baseline_miou - sim_miou,
                "miou_gap_closed": _compute_gap_closed(
                    translated_miou=baseline_miou,
                    sim_miou=sim_miou,
                    real_miou=real_miou,
                ),
            }
            row.update(
                _collect_per_class_metrics(
                    summary_row=baseline,
                    sim_row=sim_row,
                    real_row=real_row,
                )
            )
            results.append(row)

    return results


def _results_field_order(rows: list[dict[str, object]]) -> list[str]:
    base_fields = [
        "display_name",
        "model_name",
        "model_category",
        "cond_mode",
        "use_spade",
        "mid_skips_only",
        "checkpoint",
        "noise_strength",
        "cfg_weight",
        "num_seg_runs",
        "fid",
        "deeplab_fd",
        "ber",
        "ber_std",
        "lpips_mean",
        "lpips_std",
        "miou_mean",
        "miou_std",
        "delta_miou_vs_sim",
        "miou_gap_closed",
        "pixel_acc_mean",
        "pixel_acc_std",
    ]
    per_class_fields: list[str] = []
    for class_name in PER_CLASS_ORDER:
        per_class_fields.extend(
            [
                f"miou_{class_name}",
                f"miou_std_{class_name}",
                f"delta_miou_vs_sim_{class_name}",
                f"miou_gap_closed_{class_name}",
            ]
        )
    extras = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in set(base_fields) and key not in set(per_class_fields)
        }
    )
    return base_fields + per_class_fields + extras


def _column_label(column_key: str) -> str:
    if column_key in COLUMN_LABELS:
        return COLUMN_LABELS[column_key]
    if column_key.startswith("miou_") and not column_key.startswith("miou_std_"):
        class_name = column_key.removeprefix("miou_")
        return f"{CLASS_NAMES.get(class_name, class_name.replace('_', ' ').title())} IoU"
    if column_key.startswith("delta_miou_vs_sim_"):
        class_name = column_key.removeprefix("delta_miou_vs_sim_")
        return f"$\\Delta$ {CLASS_NAMES.get(class_name, class_name.replace('_', ' ').title())} IoU"
    if column_key.startswith("miou_gap_closed_"):
        class_name = column_key.removeprefix("miou_gap_closed_")
        return f"{CLASS_NAMES.get(class_name, class_name.replace('_', ' ').title())} Gap Closed"
    return column_key.replace("_", " ").title()


def _column_formatter(column_key: str) -> tuple[str, Callable[[dict[str, object]], str]]:
    if column_key == "display_name":
        return _column_label(column_key), lambda row: _escape_latex(row.get(column_key, ""))
    if column_key in {"cond_mode", "model_category"}:
        return _column_label(column_key), lambda row: _escape_latex(row.get(column_key, ""))
    if column_key in {"use_spade", "mid_skips_only"}:
        return _column_label(column_key), lambda row: _escape_latex(_format_bool_like(row.get(column_key, "")))
    if column_key == "checkpoint":
        return _column_label(column_key), lambda row: _escape_latex("" if row.get(column_key, "") == "" else str(int(row[column_key])))
    if column_key in {
        "noise_strength",
        "cfg_weight",
        "fid",
        "deeplab_fd",
        "ber",
        "lpips_mean",
        "delta_miou_vs_sim",
        "miou_gap_closed",
    }:
        return _column_label(column_key), lambda row: _format_float(row.get(column_key, ""), decimals=3)
    if column_key in {"miou", "pixel_acc", "lpips"}:
        mean_key = f"{column_key}_mean"
        std_key = f"{column_key}_std"
        return _column_label(column_key), lambda row: (
            _format_mean_std(row.get(mean_key, ""), row.get(std_key, ""), decimals=3)
        )
    if column_key in {"miou_mean", "miou_std", "pixel_acc_mean", "pixel_acc_std", "lpips_std", "ber_std"}:
        return _column_label(column_key), lambda row: _format_float(row.get(column_key, ""), decimals=3)
    if (
        column_key.startswith("miou_std_")
        or column_key.startswith("delta_miou_vs_sim_")
        or column_key.startswith("miou_gap_closed_")
        or column_key.startswith("miou_")
    ):
        return _column_label(column_key), lambda row: _format_float(row.get(column_key, ""), decimals=3)
    return _column_label(column_key), lambda row: _escape_latex(row.get(column_key, ""))


def write_final_results_table_csv(
    selected_models_csv: str | Path,
    segmentation_summary_csv: str | Path,
    save_path: str | Path,
    include_baselines: bool = True,
) -> Path:
    rows = build_final_results_rows(
        selected_models_csv=selected_models_csv,
        segmentation_summary_csv=segmentation_summary_csv,
        include_baselines=include_baselines,
    )
    fieldnames = _results_field_order(rows)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return save_path


def write_final_results_table_tex(
    selected_models_csv: str | Path,
    segmentation_summary_csv: str | Path,
    save_path: str | Path,
    columns: list[str],
    include_baselines: bool = True,
) -> Path:
    if not columns:
        raise ValueError("columns cannot be empty for LaTeX table export.")

    rows = build_final_results_rows(
        selected_models_csv=selected_models_csv,
        segmentation_summary_csv=segmentation_summary_csv,
        include_baselines=include_baselines,
    )
    header_specs = [_column_formatter(column_key) for column_key in columns]
    col_spec = "l" + "c" * max(0, len(columns) - 1)

    lines: list[str] = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(header for header, _ in header_specs) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        values = [formatter(row) for _, formatter in header_specs]
        lines.append(" & ".join(values) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n")
    return save_path


def main() -> None:
    # CSV listing selected translation models and their chosen checkpoint / sampling settings.
    selected_models_csv = "/develop/code/eval/thesis/selected_models.csv"
    # CSV aggregating downstream segmentation test metrics across the 3 runs per condition.
    segmentation_summary_csv = "/develop/code/eval/thesis/test_run_dir_mean_std.csv"
    # Output path for the joined final-results table CSV.
    save_path = "/develop/code/eval/thesis/final_results_table.csv"
    # Output path for the LaTeX table body.
    tex_save_path = "/develop/code/eval/thesis/final_results_table.tex"
    # Whether to append sim and real baseline segmentation rows with blank translation-metric fields.
    include_baselines = True
    # Columns to include in the LaTeX table, in display order.
    tex_columns = [
        "display_name",
        # "checkpoint",
        # "noise_strength",
        # "cfg_weight",
        "fid",
        "deeplab_fd",
        "ber",
        # "lpips",
        "miou",
        "delta_miou_vs_sim",
        "miou_gap_closed",
        "pixel_acc",
    ]

    saved_path = write_final_results_table_csv(
        selected_models_csv=selected_models_csv,
        segmentation_summary_csv=segmentation_summary_csv,
        save_path=save_path,
        include_baselines=include_baselines,
    )
    tex_path = write_final_results_table_tex(
        selected_models_csv=selected_models_csv,
        segmentation_summary_csv=segmentation_summary_csv,
        save_path=tex_save_path,
        columns=tex_columns,
        include_baselines=include_baselines,
    )
    print(f"Saved final results table CSV to {saved_path}")
    print(f"Saved final results table LaTeX to {tex_path}")


if __name__ == "__main__":
    main()
