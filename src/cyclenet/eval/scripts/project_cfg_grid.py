import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from omegaconf import DictConfig, OmegaConf
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cyclenet.eval.plotting.set_style import apply_style
from cyclenet.eval.scripts.project_translated import (
    compute_reference_axis_limits,
    load_or_fit_reference_projection,
    load_config,
    save_config,
    title_names,
)

apply_style()


def cfg_select(config: DictConfig, key: str, default=None):
    value = OmegaConf.select(config, key)
    return default if value is None else value


def sample_coords(xy: np.ndarray, max_points: int | None, seed: int) -> np.ndarray:
    if max_points is None or len(xy) <= max_points:
        return xy

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xy), size=max_points, replace=False)
    return xy[idx]


def should_rasterize_points(config: DictConfig, point_count: int) -> bool:
    rasterize = cfg_select(config, "plotting.points.rasterize", None)
    if rasterize is not None:
        return bool(rasterize)

    threshold = cfg_select(config, "plotting.points.rasterize_threshold", 3000)
    return point_count > int(threshold)


def translated_embed_path(
    projection_root: Path,
    model: str,
    step: int | str,
    strength: float,
    cfg_weight: float,
) -> Path:
    return (
        projection_root
        / f"step-{step}"
        / f"strength-{strength}"
        / f"cfg-{cfg_weight}"
        / f"{model}_translated_embed.npy"
    )


def axis_labels(method: str) -> tuple[str, str]:
    if method == "umap":
        return "UMAP 1", "UMAP 2"
    if method == "pca":
        return "PCA 1", "PCA 2"
    raise ValueError(f"Unsupported projection method: {method}")


def add_marginal_axes(ax, config: DictConfig):
    if not bool(cfg_select(config, "plotting.marginals.show", False)):
        return None, None

    divider = make_axes_locatable(ax)
    top_ax = divider.append_axes(
        "top",
        size=str(cfg_select(config, "plotting.marginals.size", "18%")),
        pad=float(cfg_select(config, "plotting.marginals.pad", 0.04)),
        sharex=ax,
    )
    right_ax = divider.append_axes(
        "right",
        size=str(cfg_select(config, "plotting.marginals.size", "18%")),
        pad=float(cfg_select(config, "plotting.marginals.pad", 0.04)),
        sharey=ax,
    )
    top_ax.set_xticks([])
    top_ax.set_yticks([])
    right_ax.set_xticks([])
    right_ax.set_yticks([])
    for marginal_ax in (top_ax, right_ax):
        for spine in marginal_ax.spines.values():
            spine.set_visible(False)
    return top_ax, right_ax


def draw_group_scatter(
    ax,
    coords: np.ndarray,
    color: str,
    alpha: float,
    size: float,
    rasterized: bool,
):
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=size,
        alpha=alpha,
        color=color,
        linewidths=0,
        edgecolors="none",
        rasterized=rasterized,
    )


def draw_group_kde(
    ax,
    coords: np.ndarray,
    color: str,
    fill: bool,
    alpha: float,
    levels: int,
    linewidth: float,
):
    sns.kdeplot(
        x=coords[:, 0],
        y=coords[:, 1],
        ax=ax,
        fill=fill,
        levels=levels,
        color=color,
        alpha=alpha,
        linewidths=linewidth,
    )


def draw_group_marginals(
    top_ax,
    right_ax,
    coords: np.ndarray,
    color: str,
    alpha: float,
    linewidth: float,
    fill: bool,
):
    if top_ax is None or right_ax is None:
        return

    sns.kdeplot(
        x=coords[:, 0],
        ax=top_ax,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        fill=fill,
    )
    sns.kdeplot(
        y=coords[:, 1],
        ax=right_ax,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        fill=fill,
    )


def style_panel_axes(ax, config: DictConfig):
    if bool(cfg_select(config, "plotting.panels.force_square", True)):
        ax.set_box_aspect(float(cfg_select(config, "plotting.panels.box_aspect", 1.0)))

    if bool(cfg_select(config, "plotting.panels.show_box", True)):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(str(cfg_select(config, "plotting.panels.box_color", "black")))
            spine.set_linewidth(float(cfg_select(config, "plotting.panels.box_linewidth", 0.8)))
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if bool(cfg_select(config, "plotting.panels.show_facecolor", False)):
        ax.set_facecolor(str(cfg_select(config, "plotting.panels.facecolor", "white")))


def draw_projection_panel(
    ax,
    sim_coords: np.ndarray,
    real_coords: np.ndarray,
    translated_coords: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    config: DictConfig,
    seed: int,
    title: str,
    xlabel: str,
    ylabel: str,
    show_xlabel: bool,
    show_ylabel: bool,
):
    top_ax, right_ax = add_marginal_axes(ax, config)

    ref_max_points = cfg_select(config, "plotting.reference.points.max_points_per_group", None)
    trans_max_points = cfg_select(config, "plotting.translated.points.max_points_per_group", None)
    ref_kde_points = cfg_select(config, "plotting.reference.kde.max_points_per_group", None)
    trans_kde_points = cfg_select(config, "plotting.translated.kde.max_points_per_group", None)

    sim_points = sample_coords(sim_coords, ref_max_points, seed)
    real_points = sample_coords(real_coords, ref_max_points, seed + 1)
    translated_points = sample_coords(translated_coords, trans_max_points, seed + 2)

    sim_kde = sample_coords(sim_coords, ref_kde_points, seed + 10)
    real_kde = sample_coords(real_coords, ref_kde_points, seed + 11)
    translated_kde = sample_coords(translated_coords, trans_kde_points, seed + 12)

    colors = config.plotting.colors
    ref_points_show = bool(cfg_select(config, "plotting.reference.points.show", True))
    ref_kde_show = bool(cfg_select(config, "plotting.reference.kde.show", False))
    trans_points_show = bool(cfg_select(config, "plotting.translated.points.show", True))
    trans_kde_show = bool(cfg_select(config, "plotting.translated.kde.show", False))
    marginals_show = bool(cfg_select(config, "plotting.marginals.show", False))

    if ref_kde_show:
        draw_group_kde(
            ax=ax,
            coords=sim_kde,
            color=str(colors.sim),
            fill=bool(cfg_select(config, "plotting.reference.kde.fill", False)),
            alpha=float(cfg_select(config, "plotting.reference.kde.alpha", 0.25)),
            levels=int(cfg_select(config, "plotting.reference.kde.levels", 6)),
            linewidth=float(cfg_select(config, "plotting.reference.kde.linewidth", 1.0)),
        )
        draw_group_kde(
            ax=ax,
            coords=real_kde,
            color=str(colors.real),
            fill=bool(cfg_select(config, "plotting.reference.kde.fill", False)),
            alpha=float(cfg_select(config, "plotting.reference.kde.alpha", 0.25)),
            levels=int(cfg_select(config, "plotting.reference.kde.levels", 6)),
            linewidth=float(cfg_select(config, "plotting.reference.kde.linewidth", 1.0)),
        )
    if trans_kde_show:
        draw_group_kde(
            ax=ax,
            coords=translated_kde,
            color=str(colors.translated),
            fill=bool(cfg_select(config, "plotting.translated.kde.fill", False)),
            alpha=float(cfg_select(config, "plotting.translated.kde.alpha", 0.35)),
            levels=int(cfg_select(config, "plotting.translated.kde.levels", 6)),
            linewidth=float(cfg_select(config, "plotting.translated.kde.linewidth", 1.1)),
        )

    if ref_points_show:
        draw_group_scatter(
            ax=ax,
            coords=sim_points,
            color=str(colors.sim),
            alpha=float(cfg_select(config, "plotting.reference.points.alpha", 0.08)),
            size=float(cfg_select(config, "plotting.reference.points.size", 8.0)),
            rasterized=should_rasterize_points(config, len(sim_points)),
        )
        draw_group_scatter(
            ax=ax,
            coords=real_points,
            color=str(colors.real),
            alpha=float(cfg_select(config, "plotting.reference.points.alpha", 0.08)),
            size=float(cfg_select(config, "plotting.reference.points.size", 8.0)),
            rasterized=should_rasterize_points(config, len(real_points)),
        )
    if trans_points_show:
        draw_group_scatter(
            ax=ax,
            coords=translated_points,
            color=str(colors.translated),
            alpha=float(cfg_select(config, "plotting.translated.points.alpha", 0.20)),
            size=float(cfg_select(config, "plotting.translated.points.size", 8.0)),
            rasterized=should_rasterize_points(config, len(translated_points)),
        )

    if marginals_show:
        marginal_fill = bool(cfg_select(config, "plotting.marginals.fill", False))
        marginal_alpha = float(cfg_select(config, "plotting.marginals.alpha", 0.8))
        marginal_linewidth = float(cfg_select(config, "plotting.marginals.linewidth", 1.0))
        if ref_points_show or ref_kde_show:
            draw_group_marginals(top_ax, right_ax, sim_kde, str(colors.sim), marginal_alpha, marginal_linewidth, marginal_fill)
            draw_group_marginals(top_ax, right_ax, real_kde, str(colors.real), marginal_alpha, marginal_linewidth, marginal_fill)
        if trans_points_show or trans_kde_show:
            draw_group_marginals(
                top_ax,
                right_ax,
                translated_kde,
                str(colors.translated),
                marginal_alpha,
                marginal_linewidth,
                marginal_fill,
            )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    if bool(cfg_select(config, "plotting.titles.show_panel_titles", False)):
        ax.set_title(title, fontsize=float(cfg_select(config, "plotting.titles.panel_fontsize", 9.0)))
    ax.set_xlabel(xlabel if show_xlabel else "")
    ax.set_ylabel(ylabel if show_ylabel else "")
    style_panel_axes(ax, config)


def draw_trajectory_panel(
    ax,
    sim_coords: np.ndarray,
    real_coords: np.ndarray,
    centroids: np.ndarray,
    cfg_weights: list[float],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    config: DictConfig,
    seed: int,
    xlabel: str,
    ylabel: str,
    show_xlabel: bool,
    show_ylabel: bool,
):
    ref_points = cfg_select(config, "plotting.summary.reference.max_points_per_group", cfg_select(config, "plotting.reference.points.max_points_per_group", 1000))
    sim_points = sample_coords(sim_coords, ref_points, seed)
    real_points = sample_coords(real_coords, ref_points, seed + 1)

    if bool(cfg_select(config, "plotting.summary.reference.show_points", True)):
        draw_group_scatter(
            ax=ax,
            coords=sim_points,
            color=str(config.plotting.colors.sim),
            alpha=float(cfg_select(config, "plotting.summary.reference.alpha", 0.08)),
            size=float(cfg_select(config, "plotting.summary.reference.size", 8.0)),
            rasterized=should_rasterize_points(config, len(sim_points)),
        )
        draw_group_scatter(
            ax=ax,
            coords=real_points,
            color=str(config.plotting.colors.real),
            alpha=float(cfg_select(config, "plotting.summary.reference.alpha", 0.08)),
            size=float(cfg_select(config, "plotting.summary.reference.size", 8.0)),
            rasterized=should_rasterize_points(config, len(real_points)),
        )

    traj_color = str(cfg_select(config, "plotting.trajectory.color", str(config.plotting.colors.translated)))
    traj_alpha = float(cfg_select(config, "plotting.trajectory.alpha", 1.0))
    traj_line_width = float(cfg_select(config, "plotting.trajectory.line_width", 2.0))

    ax.plot(
        centroids[:, 0],
        centroids[:, 1],
        color=traj_color,
        linewidth=traj_line_width,
        alpha=traj_alpha,
        zorder=3,
    )

    default_marker_size = float(cfg_select(config, "plotting.trajectory.marker_size", 45.0))
    marker_size = float(cfg_select(config, "plotting.trajectory.intermediate_marker_size", default_marker_size))
    start_marker_size = float(cfg_select(config, "plotting.trajectory.start_marker_size", marker_size))
    end_marker_size = float(cfg_select(config, "plotting.trajectory.end_marker_size", marker_size * 1.15))
    marker_style = str(cfg_select(config, "plotting.trajectory.marker", "o"))
    start_marker = str(cfg_select(config, "plotting.trajectory.start_marker", marker_style))
    end_marker = str(cfg_select(config, "plotting.trajectory.end_marker", marker_style))
    edgecolor = str(cfg_select(config, "plotting.trajectory.edgecolor", "white"))
    edgewidth = float(cfg_select(config, "plotting.trajectory.edgewidth", 0.8))
    start_edgecolor = str(cfg_select(config, "plotting.trajectory.start_edgecolor", edgecolor))
    start_edgewidth = float(cfg_select(config, "plotting.trajectory.start_edgewidth", edgewidth))
    end_edgecolor = str(cfg_select(config, "plotting.trajectory.end_edgecolor", edgecolor))
    end_edgewidth = float(cfg_select(config, "plotting.trajectory.end_edgewidth", edgewidth))
    start_facecolor = str(cfg_select(config, "plotting.trajectory.start_facecolor", "white"))
    end_facecolor = str(cfg_select(config, "plotting.trajectory.end_facecolor", traj_color))

    if len(centroids) > 2:
        ax.scatter(
            centroids[1:-1, 0],
            centroids[1:-1, 1],
            s=marker_size,
            marker=marker_style,
            color=traj_color,
            edgecolors=edgecolor,
            linewidths=edgewidth,
            alpha=traj_alpha,
            zorder=4,
        )

    if len(centroids) >= 1:
        ax.scatter(
            centroids[0, 0],
            centroids[0, 1],
            s=start_marker_size,
            marker=start_marker,
            facecolors=start_facecolor,
            edgecolors=start_edgecolor,
            linewidths=start_edgewidth,
            alpha=traj_alpha,
            zorder=5,
        )

    if len(centroids) >= 2:
        ax.scatter(
            centroids[-1, 0],
            centroids[-1, 1],
            s=end_marker_size,
            marker=end_marker,
            facecolors=end_facecolor,
            edgecolors=end_edgecolor,
            linewidths=end_edgewidth,
            alpha=traj_alpha,
            zorder=5,
        )

    if bool(cfg_select(config, "plotting.trajectory.show_arrow", True)) and len(centroids) >= 2:
        arrow_on_last_segment_only = bool(cfg_select(config, "plotting.trajectory.arrow_on_last_segment_only", True))
        arrowstyle = str(cfg_select(config, "plotting.trajectory.arrowstyle", "-|>"))
        mutation_scale = float(cfg_select(config, "plotting.trajectory.arrow_mutation_scale", 11.0))
        arrow_linewidth = float(cfg_select(config, "plotting.trajectory.arrow_linewidth", traj_line_width))

        if arrow_on_last_segment_only:
            segments = [(centroids[-2], centroids[-1])]
        else:
            segments = list(zip(centroids[:-1], centroids[1:]))

        for start_pt, end_pt in segments:
            ax.add_patch(
                FancyArrowPatch(
                    posA=(float(start_pt[0]), float(start_pt[1])),
                    posB=(float(end_pt[0]), float(end_pt[1])),
                    arrowstyle=arrowstyle,
                    mutation_scale=mutation_scale,
                    linewidth=arrow_linewidth,
                    color=traj_color,
                    alpha=traj_alpha,
                    shrinkA=float(cfg_select(config, "plotting.trajectory.arrow_shrink_a", 0.0)),
                    shrinkB=float(cfg_select(config, "plotting.trajectory.arrow_shrink_b", 0.0)),
                    zorder=6,
                )
            )

    if bool(cfg_select(config, "plotting.trajectory.annotate_cfg", True)):
        dx = float(cfg_select(config, "plotting.trajectory.annotation_dx", 0.02))
        dy = float(cfg_select(config, "plotting.trajectory.annotation_dy", 0.02))
        for centroid, cfg_weight in zip(centroids, cfg_weights):
            ax.text(
                centroid[0] + dx,
                centroid[1] + dy,
                f"{cfg_weight:g}",
                color=traj_color,
                ha="left",
                va="bottom",
                fontsize=float(cfg_select(config, "plotting.trajectory.annotation_fontsize", 8.0)),
            )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    if bool(cfg_select(config, "plotting.titles.show_panel_titles", False)):
        ax.set_title(
            str(cfg_select(config, "plotting.titles.summary_title", "Trajectory")),
            fontsize=float(cfg_select(config, "plotting.titles.panel_fontsize", 9.0)),
        )
    ax.set_xlabel(xlabel if show_xlabel else "")
    ax.set_ylabel(ylabel if show_ylabel else "")
    style_panel_axes(ax, config)


def add_shared_column_titles(fig, main_axes, cfg_weights: list[float], config: DictConfig):
    if not main_axes or not bool(cfg_select(config, "plotting.titles.show_shared_column_titles", True)):
        return []

    title_template = str(cfg_select(config, "plotting.titles.cfg_label_template", "{cfg:g}"))
    y = min(ax.get_position().y0 for ax in main_axes[-1]) - float(
        cfg_select(config, "plotting.titles.bottom_cfg_label_pad", 0.018)
    )
    fontsize = float(cfg_select(config, "plotting.titles.column_title_fontsize", 9.0))
    title_artists = []

    for ax, cfg_weight in zip(main_axes[-1], cfg_weights):
        x = (ax.get_position().x0 + ax.get_position().x1) * 0.5
        title_artists.append(
            fig.text(x, y, title_template.format(cfg=cfg_weight), ha="center", va="top", fontsize=fontsize)
        )

    return title_artists


def add_summary_column_title(fig, summary_axes, config: DictConfig):
    if not summary_axes or not bool(cfg_select(config, "plotting.titles.show_shared_column_titles", True)):
        return []

    top_ax = summary_axes[0]
    return [
        fig.text(
            (top_ax.get_position().x0 + top_ax.get_position().x1) * 0.5,
            top_ax.get_position().y1 + float(cfg_select(config, "plotting.titles.column_title_pad", 0.008)),
            str(cfg_select(config, "plotting.titles.summary_title", "Trajectory")),
            ha="center",
            va="bottom",
            fontsize=float(cfg_select(config, "plotting.titles.column_title_fontsize", 9.0)),
        )
    ]


def add_cfg_xlabel(fig, main_axes, config: DictConfig):
    if not main_axes or not bool(cfg_select(config, "plotting.labels.show_cfg_xlabel", True)):
        return None

    left = min(ax.get_position().x0 for row_axes in main_axes for ax in row_axes)
    right = max(ax.get_position().x1 for row_axes in main_axes for ax in row_axes)
    bottom = min(ax.get_position().y0 for row_axes in main_axes for ax in row_axes)
    y = bottom - float(cfg_select(config, "plotting.labels.cfg_xlabel_pad", 0.055))

    return fig.text(
        (left + right) * 0.5,
        y,
        str(cfg_select(config, "plotting.labels.cfg_xlabel", r"CFG weight ($w$)")),
        ha="center",
        va="top",
        fontsize=float(cfg_select(config, "plotting.labels.cfg_xlabel_fontsize", 11.0)),
    )


def artist_bbox_in_figure_coords(fig, artist, renderer):
    return artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())


def bboxes_overlap_horizontally(bbox_a, bbox_b, pad: float = 0.0) -> bool:
    return (bbox_a.x0 - pad) < bbox_b.x1 and (bbox_b.x0 - pad) < bbox_a.x1


def add_figure_legend(fig, legend_handles, all_axes, config: DictConfig, title_artist=None, header_artists=None):
    if not bool(cfg_select(config, "plotting.legend.show", True)):
        return None

    legend_position = str(cfg_select(config, "plotting.legend.position", "top_right")).lower()
    plot_right = max(ax.get_position().x1 for ax in all_axes) if all_axes else 1.0

    legend_kwargs = {
        "handles": legend_handles,
        "bbox_transform": fig.transFigure,
        "ncol": int(cfg_select(config, "plotting.legend.ncol", 4)),
        "fontsize": float(cfg_select(config, "plotting.legend.fontsize", 10.0)),
        "handletextpad": float(cfg_select(config, "plotting.legend.handletextpad", 0.4)),
        "handlelength": float(cfg_select(config, "plotting.legend.handlelength", 1.2)),
        "columnspacing": float(cfg_select(config, "plotting.legend.columnspacing", 1.2)),
        "labelspacing": float(cfg_select(config, "plotting.legend.labelspacing", 0.5)),
        "borderaxespad": float(cfg_select(config, "plotting.legend.borderaxespad", 0.0)),
        "frameon": bool(cfg_select(config, "plotting.legend.frameon", False)),
    }

    if legend_position == "top_center":
        default_top_y = float(cfg_select(config, "plotting.legend.top_y", 0.945))
        legend_anchor = list(cfg_select(config, "plotting.legend.bbox_to_anchor", [0.5, default_top_y]))
        if len(legend_anchor) < 2:
            raise ValueError("plotting.legend.bbox_to_anchor must have at least two values.")
        legend_x = float(
            cfg_select(config, "plotting.legend.center_x", compute_header_center_x(all_axes, config))
        )
        legend_top = float(cfg_select(config, "plotting.legend.top_y", legend_anchor[1]))
        legend = fig.legend(
            loc=str(cfg_select(config, "plotting.legend.loc", "upper center")),
            bbox_to_anchor=(legend_x, legend_top),
            **legend_kwargs,
        )
    elif legend_position != "top_right":
        legend_anchor = list(cfg_select(config, "plotting.legend.bbox_to_anchor", [0.5, 0.995]))
        if len(legend_anchor) < 2:
            raise ValueError("plotting.legend.bbox_to_anchor must have at least two values.")
        legend_anchor[0] = float(cfg_select(config, "plotting.legend.center_x", compute_header_center_x(all_axes, config)))
        legend = fig.legend(
            loc=str(cfg_select(config, "plotting.legend.loc", "upper center")),
            bbox_to_anchor=tuple(legend_anchor),
            **legend_kwargs,
        )
        legend_x = None
    else:
        legend_reference = str(cfg_select(config, "plotting.legend.reference", "plot")).lower()
        if legend_reference not in {"plot", "figure"}:
            raise ValueError("plotting.legend.reference must be 'plot' or 'figure'.")

        right_pad = float(cfg_select(config, "plotting.legend.right_pad", 0.0))
        legend_x = (1.0 - right_pad) if legend_reference == "figure" else (plot_right - right_pad)
        legend_top = float(cfg_select(config, "plotting.legend.top_y", cfg_select(config, "plotting.layout.title_y", 0.98)))
        legend = fig.legend(
            loc="upper right",
            bbox_to_anchor=(legend_x, legend_top),
            **legend_kwargs,
        )

    overlap_pad_x = float(cfg_select(config, "plotting.legend.overlap_pad_x", 0.006))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = artist_bbox_in_figure_coords(fig, legend, renderer)

    if title_artist is not None:
        title_gap = float(cfg_select(config, "plotting.legend.title_gap", 0.01))
        title_bbox = artist_bbox_in_figure_coords(fig, title_artist, renderer)
        if bboxes_overlap_horizontally(legend_bbox, title_bbox, pad=overlap_pad_x):
            desired_top = title_bbox.y0 - title_gap
            anchor_x = legend_x if legend_x is not None else float(
                cfg_select(config, "plotting.legend.center_x", compute_header_center_x(all_axes, config))
            )
            legend.set_bbox_to_anchor((anchor_x, desired_top), transform=fig.transFigure)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend_bbox = artist_bbox_in_figure_coords(fig, legend, renderer)

    if header_artists:
        header_gap = float(cfg_select(config, "plotting.legend.headers_gap", 0.01))
        overlapping_header_bboxes = [
            artist_bbox_in_figure_coords(fig, artist, renderer)
            for artist in header_artists
            if bboxes_overlap_horizontally(
                legend_bbox,
                artist_bbox_in_figure_coords(fig, artist, renderer),
                pad=overlap_pad_x,
            )
        ]
        if overlapping_header_bboxes:
            header_top = max(bbox.y1 for bbox in overlapping_header_bboxes)
        else:
            header_top = None

        if header_top is not None and legend_bbox.y0 < header_top + header_gap:
            legend_shift = header_top + header_gap - legend_bbox.y0
            legend_top = legend_bbox.y1 + legend_shift
            if title_artist is not None:
                title_bbox = artist_bbox_in_figure_coords(fig, title_artist, renderer)
                if bboxes_overlap_horizontally(legend_bbox, title_bbox, pad=overlap_pad_x):
                    legend_top = min(legend_top, title_bbox.y0 - float(cfg_select(config, "plotting.legend.title_gap", 0.01)))
            anchor_x = legend_x if legend_x is not None else float(
                cfg_select(config, "plotting.legend.center_x", compute_header_center_x(all_axes, config))
            )
            legend.set_bbox_to_anchor((anchor_x, legend_top), transform=fig.transFigure)
            fig.canvas.draw()

    return legend


def add_global_axis_labels(fig, all_axes, xlabel: str, ylabel: str, config: DictConfig):
    if not all_axes:
        return

    left = min(ax.get_position().x0 for ax in all_axes)
    right = max(ax.get_position().x1 for ax in all_axes)
    bottom = min(ax.get_position().y0 for ax in all_axes)
    top = max(ax.get_position().y1 for ax in all_axes)

    if bool(cfg_select(config, "plotting.labels.show_global_xlabel", False)):
        fig.text(
            (left + right) * 0.5,
            float(cfg_select(config, "plotting.labels.global_xlabel_y", bottom - 0.035)),
            str(cfg_select(config, "plotting.labels.global_xlabel", xlabel)),
            ha="center",
            va="top",
            fontsize=float(cfg_select(config, "plotting.labels.global_xlabel_fontsize", 11.0)),
        )

    if bool(cfg_select(config, "plotting.labels.show_global_ylabel", False)):
        fig.text(
            float(cfg_select(config, "plotting.labels.global_ylabel_x", left - 0.035)),
            (bottom + top) * 0.5,
            str(cfg_select(config, "plotting.labels.global_ylabel", ylabel)),
            ha="right",
            va="center",
            rotation=90,
            fontsize=float(cfg_select(config, "plotting.labels.global_ylabel_fontsize", 11.0)),
        )


def add_row_axis_label(fig, row_label_artists, main_axes, config: DictConfig):
    if not main_axes:
        return None

    default_show = cfg_select(config, "comparison.rows", None) is not None
    if not bool(cfg_select(config, "plotting.labels.show_row_axis_label", default_show)):
        return None

    label_text = str(cfg_select(config, "plotting.labels.row_axis_label", "Model Checkpoint"))
    if not label_text:
        return None

    y0 = min(ax.get_position().y0 for row_axes in main_axes for ax in row_axes)
    y1 = max(ax.get_position().y1 for row_axes in main_axes for ax in row_axes)
    x = min(ax.get_position().x0 for row_axes in main_axes for ax in row_axes) - float(
        cfg_select(config, "plotting.labels.row_axis_label_pad", 0.055)
    )

    if row_label_artists:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        label_left = min(
            artist_bbox_in_figure_coords(fig, artist, renderer).x0 for artist in row_label_artists
        )
        x = min(
            x,
            label_left - float(cfg_select(config, "plotting.labels.row_axis_label_gap", 0.02)),
        )

    x = max(float(cfg_select(config, "plotting.labels.row_axis_label_min_x", 0.01)), x)

    return fig.text(
        x,
        (y0 + y1) * 0.5,
        label_text,
        ha="center",
        va="center",
        rotation=90,
        fontsize=float(cfg_select(config, "plotting.labels.row_axis_label_fontsize", 12.0)),
    )


def add_vertical_summary_separator(fig, main_axes, summary_axes, config: DictConfig):
    if not main_axes or not summary_axes or not bool(cfg_select(config, "plotting.separators.show_vertical_summary_separator", True)):
        return

    last_main_right = max(row_axes[-1].get_position().x1 for row_axes in main_axes)
    summary_left = min(ax.get_position().x0 for ax in summary_axes)
    y0 = min(ax.get_position().y0 for row_axes in main_axes for ax in row_axes)
    y1 = max(ax.get_position().y1 for ax in summary_axes)
    x = (last_main_right + summary_left) * 0.5

    fig.add_artist(
        Line2D(
            [x, x],
            [y0, y1],
            transform=fig.transFigure,
            color=str(cfg_select(config, "plotting.separators.color", "#9ca3af")),
            linewidth=float(cfg_select(config, "plotting.separators.linewidth", 0.8)),
            alpha=float(cfg_select(config, "plotting.separators.alpha", 0.8)),
        )
    )


def add_horizontal_row_separators(fig, row_first_axes: list[plt.Axes], all_axes, config: DictConfig):
    if len(row_first_axes) < 2 or not bool(cfg_select(config, "plotting.separators.show_horizontal_row_separators", False)):
        return

    x0 = min(ax.get_position().x0 for ax in all_axes)
    x1 = max(ax.get_position().x1 for ax in all_axes)
    for upper_ax, lower_ax in zip(row_first_axes[:-1], row_first_axes[1:]):
        y = (upper_ax.get_position().y0 + lower_ax.get_position().y1) * 0.5
        fig.add_artist(
            Line2D(
                [x0, x1],
                [y, y],
                transform=fig.transFigure,
                color=str(cfg_select(config, "plotting.separators.color", "#9ca3af")),
                linewidth=float(cfg_select(config, "plotting.separators.linewidth", 0.8)),
                alpha=float(cfg_select(config, "plotting.separators.alpha", 0.8)),
            )
        )


def compute_header_center_x(all_axes, config: DictConfig) -> float:
    explicit_center_x = cfg_select(config, "plotting.layout.header_center_x", None)
    if explicit_center_x is not None:
        return float(explicit_center_x)
    if not all_axes:
        return 0.5

    left = min(ax.get_position().x0 for ax in all_axes)
    right = max(ax.get_position().x1 for ax in all_axes)
    return (left + right) * 0.5


def build_row_specs(config: DictConfig) -> list[dict[str, object]]:
    rows_cfg = cfg_select(config, "comparison.rows", None)
    if rows_cfg is not None:
        embedding_model = str(cfg_select(config, "comparison.embedding_model", "")).lower()
        if not embedding_model:
            raise ValueError(
                "comparison.embedding_model must be set when comparison.rows is used."
            )
        default_projection_root_value = cfg_select(config, "data.projection_root", None)

        row_specs: list[dict[str, object]] = []
        for row_cfg in rows_cfg:
            step = cfg_select(row_cfg, "step")
            if step is None:
                raise ValueError("Each comparison.rows entry must define step.")

            strength_value = cfg_select(row_cfg, "noise_strength")
            if strength_value is None:
                raise ValueError("Each comparison.rows entry must define noise_strength.")

            label = cfg_select(row_cfg, "label", None)
            if label is None:
                label = f"step {step}, strength {float(strength_value):g}"

            projection_root_value = cfg_select(row_cfg, "projection_root", default_projection_root_value)
            if projection_root_value is None:
                raise ValueError(
                    "Each comparison.rows entry must define projection_root, "
                    "or data.projection_root must be set as a shared default."
                )

            row_specs.append(
                {
                    "embed_model": embedding_model,
                    "step": step,
                    "strength": float(strength_value),
                    "label": str(label),
                    "projection_root": Path(str(projection_root_value)),
                }
            )

        if not row_specs:
            raise ValueError("comparison.rows must contain at least one entry.")
        return row_specs

    models = [str(v).lower() for v in list(cfg_select(config, "comparison.models", []))]
    step = cfg_select(config, "comparison.step")
    strength_value = cfg_select(config, "comparison.noise_strength")
    if not models:
        raise ValueError(
            "comparison.models must contain at least one embedding model when comparison.rows is not used."
        )
    if step is None or strength_value is None:
        raise ValueError(
            "comparison.step and comparison.noise_strength must be set when using comparison.models."
        )

    return [
        {
            "embed_model": model,
            "step": step,
            "strength": float(strength_value),
            "projection_root": Path(str(config.data.projection_root)),
            "label": str(
                cfg_select(config, f"plotting.row_titles.{model}", title_names["model"].get(model, model))
            ),
        }
        for model in models
    ]


def build_output_path(
    out_dir: Path,
    method: str,
    row_specs: list[dict[str, object]],
    cfg_weights: list[float],
    config: DictConfig,
) -> Path:
    output_name = cfg_select(config, "comparison.output_name", None)
    if output_name is not None:
        return out_dir / str(output_name)

    embed_models = sorted({str(spec["embed_model"]) for spec in row_specs})
    step_names = "-".join(str(spec["step"]) for spec in row_specs)
    strength_names = "-".join(f"{float(spec['strength']):g}" for spec in row_specs)

    if len(embed_models) == 1:
        embed_part = embed_models[0]
    else:
        embed_part = "mixed-embedders"

    return out_dir / (
        f"{method}_cfg_grid_{embed_part}_steps-{step_names}_strengths-{strength_names}"
        f"_cfg-{cfg_weights[0]:g}-to-{cfg_weights[-1]:g}.pdf"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    out_dir = Path(config.data.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir / "config.yaml")

    reference_cache_root = Path(config.data.reference_cache_dir)
    method = str(config.projection.method).lower()
    row_specs = build_row_specs(config)
    cfg_weights = [float(v) for v in list(config.comparison.cfg_weights)]
    if not cfg_weights:
        raise ValueError("comparison.cfg_weights must contain at least one value.")

    n_rows = len(row_specs)
    n_cfg = len(cfg_weights)
    fig_w = float(cfg_select(config, "plotting.layout.fig_width", 18.0))
    fig_h = float(cfg_select(config, "plotting.layout.fig_height", 7.0))
    summary_width_ratio = float(cfg_select(config, "plotting.layout.summary_width_ratio", 1.0))
    summary_gap_ratio = float(cfg_select(config, "plotting.layout.summary_gap_ratio", 0.0))
    width_ratios = [1.0] * n_cfg
    if summary_gap_ratio > 0.0:
        # Reserve an empty column so only the summary panel gets extra left-side spacing.
        width_ratios.append(summary_gap_ratio)
    width_ratios.append(summary_width_ratio)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows,
        len(width_ratios),
        width_ratios=width_ratios,
        wspace=float(cfg_select(config, "plotting.layout.wspace", 0.18)),
        hspace=float(cfg_select(config, "plotting.layout.hspace", 0.18)),
        left=float(cfg_select(config, "plotting.layout.left", 0.08)),
        right=float(cfg_select(config, "plotting.layout.right", 0.98)),
        top=float(cfg_select(config, "plotting.layout.top", 0.92)),
        bottom=float(cfg_select(config, "plotting.layout.bottom", 0.08)),
    )

    xlabel, ylabel = axis_labels(method)
    row_first_axes = []
    main_axes = []
    summary_axes = []

    for row_idx, row_spec in enumerate(row_specs):
        embed_model = str(row_spec["embed_model"])
        step = row_spec["step"]
        strength = float(row_spec["strength"])
        projection_root = Path(row_spec["projection_root"])

        reference_cache_dir = reference_cache_root / embed_model
        sim_emb = np.load(reference_cache_dir / "sim_embed.npy")
        real_emb = np.load(reference_cache_dir / "real_embed.npy")
        projector, sim_coords, real_coords = load_or_fit_reference_projection(
            method=method,
            projection_config=config.projection,
            cache_dir=reference_cache_dir,
            sim_emb=sim_emb,
            real_emb=real_emb,
        )
        xlim, ylim = compute_reference_axis_limits(
            sim_coords=sim_coords,
            real_coords=real_coords,
            pad_frac=float(cfg_select(config, "plotting.axis_pad_frac", 0.05)),
        )

        centroids = []
        row_axes = []
        for col_idx, cfg_weight in enumerate(cfg_weights):
            embed_path = translated_embed_path(
                projection_root=projection_root,
                model=embed_model,
                step=step,
                strength=strength,
                cfg_weight=cfg_weight,
            )
            if not embed_path.exists():
                raise FileNotFoundError(f"Missing translated embedding cache: {embed_path}")
            translated_emb = np.load(embed_path)
            translated_coords = projector.transform(translated_emb)
            centroids.append(translated_coords.mean(axis=0))

            ax = fig.add_subplot(gs[row_idx, col_idx])
            panel_title_template = str(cfg_select(config, "plotting.titles.cfg_title_template", "CFG {cfg:g}"))
            draw_projection_panel(
                ax=ax,
                sim_coords=sim_coords,
                real_coords=real_coords,
                translated_coords=translated_coords,
                xlim=xlim,
                ylim=ylim,
                config=config,
                seed=int(config.run.seed) + row_idx * 100 + col_idx * 10,
                title=panel_title_template.format(cfg=cfg_weight),
                xlabel=xlabel,
                ylabel=ylabel,
                show_xlabel=row_idx == n_rows - 1 and bool(cfg_select(config, "plotting.labels.show_bottom_xlabels", False)),
                show_ylabel=col_idx == 0 and bool(cfg_select(config, "plotting.labels.show_left_ylabels", True)),
            )
            if col_idx == 0:
                row_first_axes.append(ax)
            row_axes.append(ax)

        main_axes.append(row_axes)

        centroids = np.stack(centroids, axis=0)
        traj_ax = fig.add_subplot(gs[row_idx, -1])
        draw_trajectory_panel(
            ax=traj_ax,
            sim_coords=sim_coords,
            real_coords=real_coords,
            centroids=centroids,
            cfg_weights=cfg_weights,
            xlim=xlim,
            ylim=ylim,
            config=config,
            seed=int(config.run.seed) + row_idx * 1000,
            xlabel=xlabel,
            ylabel=ylabel,
            show_xlabel=row_idx == n_rows - 1 and bool(cfg_select(config, "plotting.labels.show_bottom_xlabels", False)),
            show_ylabel=bool(cfg_select(config, "plotting.labels.show_summary_ylabels", False)),
        )
        summary_axes.append(traj_ax)

    # Resolve square box-aspect adjustments before positioning shared titles and separators.
    fig.canvas.draw()

    row_label_artists = []
    for ax, row_spec in zip(row_first_axes, row_specs):
        row_title = str(row_spec["label"])
        row_label_artists.append(
            ax.text(
            float(cfg_select(config, "plotting.layout.row_label_x", -0.30)),
            0.5,
            row_title,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=float(cfg_select(config, "plotting.layout.row_label_fontsize", 12.0)),
        )
        )

    all_axes = [ax for row_axes in main_axes for ax in row_axes] + summary_axes
    add_row_axis_label(fig, row_label_artists, main_axes, config)
    shared_title_artists = add_shared_column_titles(fig, main_axes, cfg_weights, config)
    summary_title_artists = add_summary_column_title(fig, summary_axes, config)
    add_vertical_summary_separator(fig, main_axes, summary_axes, config)
    add_horizontal_row_separators(fig, row_first_axes, all_axes, config)
    add_global_axis_labels(fig, all_axes, xlabel, ylabel, config)
    add_cfg_xlabel(fig, main_axes, config)
    header_center_x = compute_header_center_x(all_axes, config)

    overall_title = cfg_select(
        config,
        "plotting.title",
        f"{title_names['method'][method]} Projections Across CFG",
    )
    title_artist = None
    if overall_title:
        title_artist = fig.suptitle(
            str(overall_title),
            x=header_center_x,
            y=float(cfg_select(config, "plotting.layout.title_y", 0.98)),
            fontsize=float(cfg_select(config, "plotting.layout.title_fontsize", 12.0)),
        )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=str(config.plotting.colors.sim), label=str(config.plotting.labels.sim)),
        Line2D([0], [0], marker="o", linestyle="None", color=str(config.plotting.colors.real), label=str(config.plotting.labels.real)),
        Line2D([0], [0], marker="o", linestyle="None", color=str(config.plotting.colors.translated), label=str(config.plotting.labels.translated)),
        Line2D(
            [0], [0],
            color=str(cfg_select(config, "plotting.trajectory.color", str(config.plotting.colors.translated))),
            linewidth=float(cfg_select(config, "plotting.trajectory.line_width", 2.0)),
            marker="o",
            markersize=6,
            markerfacecolor=str(cfg_select(config, "plotting.trajectory.end_facecolor", str(config.plotting.colors.translated))),
            markeredgecolor=str(
                cfg_select(
                    config,
                    "plotting.trajectory.end_edgecolor",
                    cfg_select(config, "plotting.trajectory.edgecolor", "white"),
                )
            ),
            markeredgewidth=float(
                cfg_select(
                    config,
                    "plotting.trajectory.end_edgewidth",
                    cfg_select(config, "plotting.trajectory.edgewidth", 0.8),
                )
            ),
            label=str(config.plotting.labels.trajectory),
        )
    ]
    add_figure_legend(
        fig=fig,
        legend_handles=legend_handles,
        all_axes=all_axes,
        config=config,
        title_artist=title_artist,
        header_artists=summary_title_artists,
    )

    out_path = build_output_path(
        out_dir=out_dir,
        method=method,
        row_specs=row_specs,
        cfg_weights=cfg_weights,
        config=config,
    )
    fig.savefig(out_path)
    plt.close(fig)

    print(f"Wrote CFG comparison grid to {out_path}")


if __name__ == "__main__":
    main()
