from __future__ import annotations

import csv
from pathlib import Path


CLASS_COLUMNS = [
    ("Bareland", "miou_bareland"),
    ("Rangeland", "miou_rangeland"),
    ("Developed Space", "miou_developed_space"),
    ("Road", "miou_road"),
    ("Trees", "miou_trees"),
    ("Water", "miou_water"),
    ("Agriculture Land", "miou_agriculture_land"),
    ("Buildings", "miou_buildings"),
]


def _normalize_identifier(value: object) -> str:
    return str(value).strip().lower()


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


def _find_row(rows: list[dict[str, str]], identifier: str) -> dict[str, str]:
    normalized_identifier = _normalize_identifier(identifier)
    matches = [
        row
        for row in rows
        if _normalize_identifier(row.get("model_name", "")) == normalized_identifier
        or _normalize_identifier(row.get("display_name", "")) == normalized_identifier
    ]
    if not matches:
        raise ValueError(f"No CSV row matched identifier '{identifier}'.")
    if len(matches) > 1:
        raise ValueError(f"Multiple CSV rows matched identifier '{identifier}'.")
    return matches[0]


def _format_model_label(row: dict[str, str]) -> str:
    model_name = str(row.get("model_name", "")).strip()
    display_name = str(row.get("display_name", "")).strip()
    if model_name == "sim":
        return "Sim"
    if display_name:
        return display_name
    return model_name


def _format_metric(row: dict[str, str], column_name: str, decimals: int) -> str:
    if column_name not in row:
        raise ValueError(f"Missing required column '{column_name}' in CSV row '{row.get('model_name', '')}'.")
    return f"{float(row[column_name]):.{decimals}f}"


def build_per_class_latex_table(
    csv_path: str | Path,
    selected_models: list[str],
    output_path: str | Path,
    caption: str,
    label: str,
    decimals: int = 3,
) -> Path:
    rows = _load_csv_rows(csv_path)
    selected_rows = [_find_row(rows, identifier=model_id) for model_id in selected_models]

    header_cells = ["Model", *[class_label for class_label, _ in CLASS_COLUMNS]]
    body_lines: list[str] = []
    for row in selected_rows:
        cell_values = [_escape_latex(_format_model_label(row))]
        for _, column_name in CLASS_COLUMNS:
            cell_values.append(_format_metric(row, column_name=column_name, decimals=decimals))
        body_lines.append("        " + " & ".join(cell_values) + r" \\")

    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"    \centering",
            r"    \small",
            r"    \setlength{\tabcolsep}{4pt}",
            rf"    \caption{{{_escape_latex(caption)}}}",
            rf"    \label{{{label}}}",
            r"    \resizebox{\textwidth}{!}{%",
            r"    \begin{tabular}{lcccccccc}",
            r"        \toprule",
            "        " + " & ".join(_escape_latex(cell) for cell in header_cells) + r" \\",
            r"        \midrule",
            *body_lines,
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table*}",
            "",
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex)
    return output_path


def main() -> None:
    # Final results CSV containing the per-class mIoU columns for each model.
    csv_path = "/home/logan/projects/pixel-cyclenet/final_results_table.csv"
    # Model identifiers to include. Each entry may match either `model_name` or `display_name`.
    selected_models = ["sim", "Seg"]
    # Output `.tex` file path for the generated LaTeX table.
    output_path = "eval/thesis/tables/sim_seg_per_class_miou.tex"
    # Table caption written into the LaTeX output.
    caption = "Per-class mIoU for the simulated baseline and the Seg translation model."
    # LaTeX label written into the table environment.
    label = "tab:sim_seg_per_class_miou"
    # Number of decimal places to use for each mIoU value.
    decimals = 3

    saved_path = build_per_class_latex_table(
        csv_path=csv_path,
        selected_models=selected_models,
        output_path=output_path,
        caption=caption,
        label=label,
        decimals=decimals,
    )
    print(f"Saved LaTeX table to {saved_path}")


if __name__ == "__main__":
    main()
