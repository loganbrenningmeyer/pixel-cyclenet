import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from cyclenet.eval.lpips import lpips_sweep
from cyclenet.eval.fid import fid_sweep
from cyclenet.eval.deeplab_fid import deeplab_fid_sweep
from cyclenet.eval.boundary_edge_align import boundary_edge_sweep
from cyclenet.eval.analyze_checkpoint_metrics import pareto_sweep
from cyclenet.eval.plotting.pareto import plot_pareto_sweep


def load_config(config_path: str | Path) -> DictConfig:
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str | Path) -> None:
    OmegaConf.save(config, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # -- Get sim/real/translated data paths and steps
    image_parent_dir = config.data.image_parent_dir

    sim_dir = Path(config.data.sim_dir) / image_parent_dir
    real_dir = Path(config.data.real_dir) / image_parent_dir
    cyclenet_sim_dir = Path(config.data.cyclenet_sim_dir)
    steps = config.data.steps


    # -------------------------
    # LPIPS
    # -------------------------

    if config.run.eval.lpips:
        print(
            f"{'=' * 50}\n"
            f"{' Running [LPIPS] '.center(50, ' ')}\n"
            f"{'=' * 50}\n"
        )

        lpips_sweep(
            sim_dir=sim_dir, 
            cyclenet_sim_dir=cyclenet_sim_dir, 
            steps=steps,
        )

    # -------------------------
    # FID
    # -------------------------
    if config.run.eval.fid:
        print(
            f"{'=' * 50}\n"
            f"{' Running [FID] '.center(50, ' ')}\n"
            f"{'=' * 50}\n"
        )

        fid_sweep(
            reference_dir=real_dir,
            cyclenet_sim_dir=cyclenet_sim_dir,
            steps=steps,
        )

    # -------------------------
    # DeepLabv3-FID
    # -------------------------
    if config.run.eval.deeplab_fid:
        print(
            f"{'=' * 50}\n"
            f"{' Running [DeepLabv3-FID] '.center(50, ' ')}\n"
            f"{'=' * 50}\n"
        )

        deeplab_ckpt_path = config.deeplab.deeplab_ckpt_path
        feature_layer = config.deeplab.feature_layer
        reference_cache_dir = config.deeplab.reference_cache_dir

        deeplab_fid_sweep(
            reference_dir=real_dir,
            cyclenet_sim_dir=cyclenet_sim_dir,
            steps=steps,
            deeplab_ckpt_path=deeplab_ckpt_path,
            feature_layer=feature_layer,
            reference_cache_dir=reference_cache_dir,
        )

    # -------------------------
    # Boundary Edge Alignment
    # -------------------------
    if config.run.eval.boundary_edge:
        print(
            f"{'=' * 50}\n"
            f"{' Running [Boundary Edge Alignment] '.center(50, ' ')}\n"
            f"{'=' * 50}\n"
        )

        boundary_edge_sweep(
            sim_dir=sim_dir,
            cyclenet_sim_dir=cyclenet_sim_dir,
            steps=steps,
        )

    # -------------------------
    # Pareto Model Selection
    # -------------------------
    pareto_out_dir = Path(config.pareto.out_dir) / config.run.name
    selection_modes = config.pareto.selection_modes
    pct_thresholds = config.pareto.pct_thresholds
    select_by_checkpoint = config.pareto.select_by_checkpoint
    max_lpips = config.pareto.max_lpips

    if config.run.eval.run_pareto:
        print(
            f"{'=' * 50}\n"
            f"{' Running [Pareto Model Selection] '.center(50, ' ')}\n"
            f"{'=' * 50}\n"
        )

        pareto_sweep(
            cyclenet_sim_dir=cyclenet_sim_dir,
            steps=steps,
            out_dir=pareto_out_dir,
            selection_modes=selection_modes,
            pct_thresholds=pct_thresholds,
            select_by_checkpoint=select_by_checkpoint,
            max_lpips=max_lpips,
        )

    # -------------------------
    # Pareto Plots
    # -------------------------
    if config.run.eval.plot_pareto:
        print(
            f"{'=' * 50}\n"
            f"{' Plotting [Pareto Model Selection] '.center(50, ' ')}\n"
            f"{'=' * 50}\n"
        )

        plot_pareto_sweep(
            out_dir=pareto_out_dir,
            selection_modes=selection_modes,
            pct_thresholds=pct_thresholds,
        )


if __name__ == "__main__":
    main()