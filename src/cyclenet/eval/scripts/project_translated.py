import argparse
import json
import pickle
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig


from cyclenet.eval.embed import CLIPEmbedder, InceptionEmbedder, DeepLabEmbedder
from cyclenet.eval.project import UmapProjector, PcaProjector
from cyclenet.eval.plotting.project import (
    plot_proj_scatter, 
    plot_proj_density, 
    plot_proj_density_marginal,
)

IMAGE_SUFFIXES = {".jpg", ".png", ".tif", ".tiff"}

title_names = {
    "model": {
        "clip": "CLIP",
        "inception": "InceptionV3",
        "deeplab": "DeepLabV3",
    },
    "method": {
        "umap": "UMAP",
        "pca": "PCA",
    }
}


def load_config(config_path: str) -> DictConfig:
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)


def collect_images(root_dir: str | Path) -> list[Path]:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Image root does not exist: {root_dir}")

    img_paths: list[Path] = []
    for path in sorted(root_dir.rglob("*")):
        # -- Avoid masks / non-images
        if (
            not path.is_file()
            or path.parent.name == "gt_ss_mask"
            or path.suffix.lower() not in IMAGE_SUFFIXES
        ):
            continue
        img_paths.append(path)

    if not img_paths:
        raise ValueError(f"No image files found under {root_dir}")

    return img_paths


def sample_paths(paths: list[Path], max_count: int, seed: int) -> list[Path]:
    if max_count is None:
        return paths
    if max_count <= 0:
        raise ValueError(f"num_samples must be positive, got {max_count}")
    if len(paths) <= max_count:
        return paths

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(paths), size=max_count, replace=False))
    return [paths[i] for i in idx]


def load_or_compute_embeddings(
    embedder,
    img_paths: list[Path],
    batch_size: int,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)

    return embedder.embed(
        [str(path) for path in img_paths],
        batch_size=batch_size,
        save_path=cache_path,
    )


def load_or_create_reference_paths(
    root_dir: str | Path,
    cache_path: Path,
    num_samples: int | None,
    seed: int,
) -> list[Path]:
    if cache_path.exists():
        paths = [Path(line.strip()) for line in cache_path.read_text().splitlines() if line.strip()]
        missing = [path for path in paths if not path.exists()]
        if missing:
            missing_text = "\n".join(str(path) for path in missing[:10])
            raise FileNotFoundError(
                f"Cached reference manifest contains missing files under {cache_path}:\n{missing_text}"
            )
        return paths

    paths = sample_paths(collect_images(root_dir), max_count=num_samples, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("".join(f"{path}\n" for path in paths))
    return paths


def build_reference_cache_metadata(config: DictConfig, model: str | None = None) -> dict[str, str | int | None]:
    model = str(model if model is not None else config.embedding.model).lower()
    metadata: dict[str, str | int | None] = {
        "sim_dir": str(Path(config.data.sim_dir).resolve()),
        "real_dir": str(Path(config.data.real_dir).resolve()),
        "model": model,
        "num_samples": OmegaConf.select(config, "embedding.num_samples"),
        "seed": int(config.run.seed),
    }

    if model == "clip":
        metadata["clip_path"] = str(Path(config.embedding.clip_path).resolve())
    elif model == "deeplab":
        metadata["deeplab_path"] = str(Path(config.embedding.deeplab_path).resolve())
        metadata["feature_layer"] = str(config.embedding.feature_layer)

    return metadata


def validate_or_write_metadata(cache_dir: Path, metadata: dict[str, str | int]) -> None:
    metadata_path = cache_dir / "reference_metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return

    existing = json.loads(metadata_path.read_text())
    if existing == metadata:
        return

    mismatches = []
    for key in sorted(set(existing) | set(metadata)):
        old_value = existing.get(key)
        new_value = metadata.get(key)
        if old_value != new_value:
            mismatches.append(f"{key}: cached={old_value!r}, current={new_value!r}")

    mismatch_text = "\n".join(mismatches)
    raise ValueError(
        "Reference cache metadata does not match the current config. "
        "Point `data.reference_cache_dir` at a different cache directory or remove the stale cache.\n"
        f"{mismatch_text}"
    )


def build_projector(method: str, projection_config: DictConfig):
    if method == "umap":
        return UmapProjector(
            n_components=projection_config.n_components,
            random_state=projection_config.random_state,
        )
    if method == "pca":
        return PcaProjector(
            n_components=projection_config.n_components,
            random_state=projection_config.random_state,
        )

    raise ValueError(f"Unsupported projection method: {method}")


def load_or_fit_reference_projection(
    method: str,
    projection_config: DictConfig,
    cache_dir: Path,
    sim_emb: np.ndarray,
    real_emb: np.ndarray,
) -> tuple[object, np.ndarray, np.ndarray]:
    projector_path = cache_dir / f"{method}_projector.pkl"
    sim_coords_path = cache_dir / f"{method}_sim_coords.npy"
    real_coords_path = cache_dir / f"{method}_real_coords.npy"
    metadata_path = cache_dir / f"{method}_projector_metadata.json"

    projector_metadata = {
        "method": method,
        "n_components": int(projection_config.n_components),
        "random_state": int(projection_config.random_state),
    }

    if (
        projector_path.exists()
        and sim_coords_path.exists()
        and real_coords_path.exists()
        and metadata_path.exists()
    ):
        existing = json.loads(metadata_path.read_text())
        if existing == projector_metadata:
            with projector_path.open("rb") as f:
                projector = pickle.load(f)
            return projector, np.load(sim_coords_path), np.load(real_coords_path)

    projector = build_projector(method, projection_config)
    ref_emb = np.concatenate([sim_emb, real_emb], axis=0)
    coords = projector.fit(ref_emb)
    sim_coords = coords[: len(sim_emb)]
    real_coords = coords[len(sim_emb) :]

    cache_dir.mkdir(parents=True, exist_ok=True)
    with projector_path.open("wb") as f:
        pickle.dump(projector, f)
    np.save(sim_coords_path, sim_coords)
    np.save(real_coords_path, real_coords)
    metadata_path.write_text(json.dumps(projector_metadata, indent=2, sort_keys=True) + "\n")

    return projector, sim_coords, real_coords


def compute_reference_axis_limits(
    sim_coords: np.ndarray,
    real_coords: np.ndarray,
    pad_frac: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    ref_coords = np.concatenate([sim_coords, real_coords], axis=0)

    x_min = float(ref_coords[:, 0].min())
    x_max = float(ref_coords[:, 0].max())
    y_min = float(ref_coords[:, 1].min())
    y_max = float(ref_coords[:, 1].max())

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_pad = x_span * pad_frac
    y_pad = y_span * pad_frac

    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def build_embedder(model: str, config: DictConfig, device: str):
    if model == "clip":
        return CLIPEmbedder(device=device, clip_path=config.embedding.clip_path)
    if model == "inception":
        return InceptionEmbedder(device)
    if model == "deeplab":
        return DeepLabEmbedder(
            device=device,
            ckpt_path=config.embedding.deeplab_path,
            feature_layer=config.embedding.feature_layer,
        )

    raise ValueError(f"Unsupported embedding model: {model}")


def combo_name(step: int | str, strength: float, cfg_weight: float) -> str:
    return f"step-{step}_strength-{strength}_cfg-{cfg_weight}"


def resolve_template_path(template: str, **kwargs) -> Path:
    return Path(str(template).format(**kwargs))


def process_projection_run(
    config: DictConfig,
    translated_dir: Path,
    out_dir: Path,
    device: str,
    model: str,
    method: str,
    embedder=None,
) -> object | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    run_config.data.translated_dir = str(translated_dir)
    run_config.data.out_dir = str(out_dir)
    run_config.embedding.model = model
    run_config.projection.method = method
    save_config(run_config, out_dir / "config.yaml")

    # -------------------------
    # Sample images / get embeddings
    # -------------------------
    seed = int(config.run.seed)
    num_samples = OmegaConf.select(config, "embedding.num_samples")
    batch_size = int(config.embedding.batch_size)

    reference_cache_root = Path(
        OmegaConf.select(config, "data.reference_cache_dir", default=str(out_dir / "reference_cache"))
    )
    reference_cache_dir = reference_cache_root / model
    reference_cache_dir.mkdir(parents=True, exist_ok=True)

    validate_or_write_metadata(reference_cache_dir, build_reference_cache_metadata(config, model=model))

    translated_cache_path = out_dir / f"{model}_translated_embed.npy"
    sim_cache_path = reference_cache_dir / "sim_embed.npy"
    real_cache_path = reference_cache_dir / "real_embed.npy"

    if sim_cache_path.exists() and real_cache_path.exists() and translated_cache_path.exists():
        sim_emb = np.load(sim_cache_path)
        real_emb = np.load(real_cache_path)
        translated_emb = np.load(translated_cache_path)
    else:
        if embedder is None:
            embedder = build_embedder(model=model, config=run_config, device=device)

        sim_images = load_or_create_reference_paths(
            run_config.data.sim_dir,
            reference_cache_dir / "sim_paths.txt",
            num_samples=num_samples,
            seed=seed,
        )
        real_images = load_or_create_reference_paths(
            run_config.data.real_dir,
            reference_cache_dir / "real_paths.txt",
            num_samples=num_samples,
            seed=seed,
        )
        translated_images = sample_paths(
            collect_images(translated_dir),
            max_count=num_samples,
            seed=seed,
        )

        sim_emb = load_or_compute_embeddings(embedder, sim_images, batch_size, sim_cache_path)
        real_emb = load_or_compute_embeddings(embedder, real_images, batch_size, real_cache_path)
        translated_emb = load_or_compute_embeddings(
            embedder,
            translated_images,
            batch_size,
            translated_cache_path,
        )

    # -------------------------
    # Project embeddings to 2D coordinates
    # -------------------------
    if method == "umap":
        xlabel = "UMAP 1"
        ylabel = "UMAP 2"
    elif method == "pca":
        xlabel = "PCA 1"
        ylabel = "PCA 2"
    else:
        raise ValueError(f"Unsupported projection method: {method}")

    # -- Fit once to cached real + sim embeddings, then reuse across translated runs
    projector, sim_coords, real_coords = load_or_fit_reference_projection(
        method=method,
        projection_config=run_config.projection,
        cache_dir=reference_cache_dir,
        sim_emb=sim_emb,
        real_emb=real_emb,
    )
    translated_coords = projector.transform(translated_emb)
    xlim, ylim = compute_reference_axis_limits(
        sim_coords=sim_coords,
        real_coords=real_coords,
        pad_frac=float(OmegaConf.select(config, "plotting.axis_pad_frac", default=0.05)),
    )

    # -------------------------
    # Plot projected embeddings
    # -------------------------
    show_points = config.plotting.points.show
    show_density = config.plotting.density.show
    show_marginal = config.plotting.marginal.show

    density_alpha = 0 if not show_density else config.plotting.density.alpha

    coords = [sim_coords, real_coords, translated_coords]
    labels = [config.plotting.labels.sim, config.plotting.labels.real, config.plotting.labels.translated]
    colors = [config.plotting.colors.sim, config.plotting.colors.real, config.plotting.colors.translated]

    # -- Plot title by model/method
    method_name = title_names["method"][method]
    model_name = title_names["model"][model]

    title = f"{method_name} Projection of {model_name} Embeddings"

    if show_marginal and (show_density or show_points):
        plot_proj_density_marginal(
            coords=coords,
            labels=labels,
            colors=colors,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=out_dir / f"{model}_{method}_projection.pdf",
            max_points_per_group=config.plotting.points.max_points_per_group,
            seed=config.run.seed,
            fill=config.plotting.density.fill,
            alpha=density_alpha,
            point_alpha=config.plotting.points.alpha,
            point_size=config.plotting.points.size,
            show_points=show_points,
            xlim=xlim,
            ylim=ylim,
        )
    elif show_density:
        plot_proj_density(
            coords=coords,
            labels=labels,
            colors=colors,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=out_dir / f"{model}_{method}_projection.pdf",
            max_points_per_group=config.plotting.points.max_points_per_group,
            seed=config.run.seed,
            fill=config.plotting.density.fill,
            alpha=density_alpha,
            point_alpha=config.plotting.points.alpha,
            point_size=config.plotting.points.size,
            show_points=show_points,
            xlim=xlim,
            ylim=ylim,
        )
    elif show_points:
        plot_proj_scatter(
            coords=coords,
            labels=labels,
            colors=colors,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            save_path=out_dir / f"{model}_{method}_projection.pdf",
            max_points_per_group=config.plotting.points.max_points_per_group,
            seed=config.run.seed,
            alpha=config.plotting.points.alpha,
            point_size=config.plotting.points.size,
            xlim=xlim,
            ylim=ylim,
        )
    else:
        raise ValueError("show_points or show_density must be True.")

    print(f"Wrote projection outputs to {out_dir}")
    return embedder


def main():
    # -------------------------
    # Parse args / load + save config 
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_model = str(config.embedding.model).lower()
    default_method = str(config.projection.method).lower()
    embedders_by_model: dict[str, object] = {}

    sweep_steps = OmegaConf.select(config, "sweep.steps")
    sweep_cfgs = OmegaConf.select(config, "sweep.cfg_weights")
    sweep_strengths = OmegaConf.select(config, "sweep.noise_strengths")
    sweep_models = OmegaConf.select(config, "sweep.embedding_models")
    sweep_methods = OmegaConf.select(config, "sweep.projection_methods")

    if (
        sweep_steps is not None
        or sweep_cfgs is not None
        or sweep_strengths is not None
        or sweep_models is not None
        or sweep_methods is not None
    ):
        if sweep_steps is None or sweep_cfgs is None or sweep_strengths is None:
            raise ValueError("Sweep mode requires sweep.steps, sweep.cfg_weights, and sweep.noise_strengths.")

        translated_dir_template = OmegaConf.select(config, "data.translated_dir_template")
        if translated_dir_template is None:
            raise ValueError("Sweep mode requires data.translated_dir_template.")

        out_dir_template = OmegaConf.select(config, "data.out_dir_template")
        base_out_dir = None
        if out_dir_template is None:
            base_out_dir_value = OmegaConf.select(config, "data.out_dir")
            if base_out_dir_value is None:
                raise ValueError(
                    "Sweep mode requires either data.out_dir_template or data.out_dir."
                )
            base_out_dir = Path(base_out_dir_value)
            base_out_dir.mkdir(parents=True, exist_ok=True)
            save_config(config, base_out_dir / "config.yaml")

        steps = [int(v) for v in sweep_steps]
        cfg_weights = [float(v) for v in sweep_cfgs]
        noise_strengths = [float(v) for v in sweep_strengths]
        embedding_models = [str(v).lower() for v in (sweep_models if sweep_models is not None else [default_model])]
        projection_methods = [str(v).lower() for v in (sweep_methods if sweep_methods is not None else [default_method])]

        for model in embedding_models:
            for method in projection_methods:
                for step in steps:
                    for strength in noise_strengths:
                        for cfg_weight in cfg_weights:
                            name = combo_name(step=step, strength=strength, cfg_weight=cfg_weight)
                            template_kwargs = {
                                "step": step,
                                "strength": strength,
                                "cfg": cfg_weight,
                                "model": model,
                                "method": method,
                            }
                            translated_dir = resolve_template_path(
                                template=str(translated_dir_template),
                                **template_kwargs,
                            )
                            out_dir = (
                                resolve_template_path(
                                    template=str(out_dir_template),
                                    **template_kwargs,
                                )
                                if out_dir_template is not None
                                else base_out_dir / model / method / name
                            )
                            print(f"[{model}/{method}/{name}] translated_dir={translated_dir}")
                            embedder = process_projection_run(
                                config=config,
                                translated_dir=translated_dir,
                                out_dir=out_dir,
                                device=device,
                                model=model,
                                method=method,
                                embedder=embedders_by_model.get(model),
                            )
                            embedders_by_model[model] = embedder
    else:
        out_dir = Path(config.data.out_dir)
        embedder = process_projection_run(
            config=config,
            translated_dir=Path(config.data.translated_dir),
            out_dir=out_dir,
            device=device,
            model=default_model,
            method=default_method,
            embedder=embedders_by_model.get(default_model),
        )
        embedders_by_model[default_model] = embedder

    print("All Done!")


if __name__ == "__main__":
    main()
