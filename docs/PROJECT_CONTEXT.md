# Project overview

`pixel-cyclenet` is a PyTorch implementation of pixel-space CycleNet for
unpaired image-to-image translation between two domains, with the current
project focus on remote-sensing simulated-to-real translation.

The repo supports two related workflows:

- Stage 1: pretrain a domain-conditioned diffusion UNet on both domains.
- Stage 2: train a CycleNet model that reuses the pretrained UNet backbone,
  adds a ControlNet-style side branch, and optimizes CycleNet reconstruction,
  cycle, consistency, and invariance losses.

The codebase now has two CycleNet entrypoint families:

- RGB-conditioned CycleNet.
- A segmentation-aware CycleNet path that can now run multiple conditioning
  modes (`rgb`, `seg`, `rgb_seg`) with optional SPADE modulation. This
  segmentation-aware path is the current active remote-sensing workflow.

# Current goals

- Improve sim-to-real remote-sensing translation quality while preserving
  land-cover geometry needed for downstream segmentation.
- Use segmentation-aware conditioning as the main mechanism for stronger
  semantic consistency during CycleNet translation.
- As of 2026-04-27, test whether SPADE segmentation modulation can preserve
  semantic boundaries better than seg-only conditioning alone while still
  allowing freer, more OEM-like translations.
- Evaluate checkpoints and sampling settings with translate-sweep runs instead
  of selecting models only by ad hoc visual inspection.
- Rank sweep candidates by realism metrics against a fixed real reference set,
  while keeping LPIPS and overlay visualizations as preservation guardrails.
- Determine whether the translation model can be trained for the practical
  deployment setting where only simulated segmentation masks are available at
  inference time, even if most real training images do not have masks.
- Treat "no usable real masks at training time" as a realistic target-domain
  constraint for related low-altitude aerial imagery work, where real data can
  be too low-quality for reliable pseudo-mask generation.

As of 2026-04-17, the most important active path is the segmentation-aware
pipeline built around:

- `src/cyclenet/train_cyclenet_seg.py`
- `src/cyclenet/translate_cyclenet_seg.py`
- `src/cyclenet/training/cyclenet_trainer_seg.py`
- `src/cyclenet/eval/translate_sweep_seg.py`
- `src/cyclenet/eval/analyze_translate_sweep.py`
- `src/cyclenet/eval/analyze_translate_sweep_consensus.py`

# Architecture and important components

## Package layout

- `src/cyclenet/models`: UNet backbone, ControlNet copy, CycleNet wrapper, and
  domain/time conditioning modules.
- `src/cyclenet/diffusion`: schedules, sampling loops, and training losses.
- `src/cyclenet/data`: datasets, augmentations, and the balanced domain batch
  sampler.
- `src/cyclenet/training`: trainer classes for UNet and CycleNet.
- `src/cyclenet/eval`: translate sweep generation, metric computation, and
  ranking/projection analysis tools.
- `configs/`: runnable configs for UNet, CycleNet, translation, resume, and
  sweep evaluation.
- `scripts/`: helper scripts for dataset inspection, subset generation, sample
  plotting, and sweep overlays.
- `info/`: architecture notes and research/implementation rationale.

## Main model design

- `UNet` is the pretrained diffusion backbone used in both pretraining and
  CycleNet.
- As of 2026-04-27, `UNet` also stores architectural metadata needed to
  rebuild SPADE-capable ControlNets from a checkpointed backbone, including
  `base_ch`, `t_dim`, `d_dim`, `ch_mults`, `num_res_blocks`, `enc_heads`, and
  `mid_heads`.
- `DomainEmbedding` is a learned 2-entry embedding table. The whole project is
  currently hard-coded around exactly two domains: sim=`0`, real=`1`.
- `ControlNet` is created by deep-copying the pretrained UNet stem, encoder,
  bottleneck, and time MLP, then adding zero-initialized 1x1 control
  convolutions.
- As of 2026-04-27, there are now two ControlNet builders:
  - `ControlNet`: the previous copied-encoder control branch without SPADE.
  - `SPADEControlNet`: a copied-encoder control branch whose residual blocks
    use optional AdaGN + SPADE modulation and receive the raw segmentation map
    separately for spatial modulation.
- `CycleNet` combines:
  - source-domain embedding for the ControlNet branch
  - target-domain embedding for the backbone branch
  - ControlNet skips summed into the backbone bottleneck and decoder skips
  - optional `seg` forwarding into the control branch so SPADE-enabled runs can
    modulate each residual block with the source segmentation map

## Two-stage training pipeline

1. UNet pretraining:
   - Entrypoint: `src/cyclenet/train_unet.py`
   - Trains a domain-conditioned diffusion UNet plus `DomainEmbedding`.
   - Uses mixed-domain batches balanced 50/50 by `DomainSampler`.
   - Saves both live and EMA weights.

2. CycleNet training:
   - Entrypoints:
     - RGB: `src/cyclenet/train_cyclenet.py`
     - segmentation-aware: `src/cyclenet/train_cyclenet_seg.py`
   - Loads the pretrained UNet EMA checkpoint as the CycleNet backbone.
   - Freezes the backbone stem, time MLP, encoder, bottleneck, and
     `DomainEmbedding`.
   - Trains ControlNet plus the UNet decoder/final layer.
   - Maintains a rank-0 EMA copy for evaluation/sampling.
   - As of 2026-04-27, the segmentation-aware training path is config-driven
     through:
     - `model.cond_mode`: `rgb`, `seg`, or `rgb_seg`
     - `model.use_spade`: whether to build `SPADEControlNet`
     - `model.s_dim`: SPADE hidden width for the segmentation modulation towers

## Current default remote-sensing configs

- UNet pretraining reference:
  `configs/unet/remote_sensing/train_unet.yaml`
- Segmentation CycleNet training reference:
  `configs/cyclenet/train_cyclenet_seg.yaml`
- Segmentation translate-sweep reference:
  `configs/cyclenet/eval_translate_sweep_seg.yaml`

The checked-in configs are tied to Logan’s environment and include absolute
paths under `/cgi/...` for datasets and `/develop/code/...` for outputs.
Future runs on another machine should update paths rather than assuming configs
are portable as-is.

## Data pipeline and directory assumptions

### RGB-only workflow

- `DomainDataset`, `CycleDomainDataset`, `SourceDataset`, and `TranslateDataset`
  load RGB files recursively.
- Optional `rgb_parent_dirs` filters image discovery by immediate parent folder
  name.

### Segmentation workflow

- `CycleDomainSegDataset`, `SourceSegDataset`, and `TranslateSegDataset`
  require RGB files and masks to share the same filename.
- Expected layout is a sibling directory convention such as:
  - `.../<tile_or_scene>/opt/<image>.tif`
  - `.../<tile_or_scene>/gt_ss_mask/<image>.tif`
- The immediate RGB parent must match one of `rgb_parent_dirs` such as
  `opt` or `pre_opt`.
- The default mask parent folder name is `gt_ss_mask`.

### Label semantics

- Remote-sensing segmentation is modeled as 8 semantic classes.
- `to_one_hot_with_ignore()` assumes:
  - `0` means ignore/background and becomes all-zero channels
  - valid semantic class ids are `1..8`
  - class ids are shifted to channel indices `0..7`
- Invalid nonzero class ids raise an error.

### Augmentations

- Albumentations handles both image and mask transforms.
- Image tensors are normalized to `[-1, 1]` via mean/std `0.5`.
- Segmentation masks always use nearest-neighbor interpolation.
- Remote-sensing transform preset `transform_id=3` uses flips and 90-degree
  rotations, reflecting rotational symmetry assumptions in overhead imagery.

## Sampling and translation

- Translation is implemented in `src/cyclenet/translate_cyclenet.py` and
  `src/cyclenet/translate_cyclenet_seg.py`.
- As of 2026-04-25, there is also a segmentation-conditioned checkpoint/CFG/
  strength sweep entrypoint at `src/cyclenet/translate_cyclenet_sweep_seg.py`,
  mirroring the RGB sweep script but requiring `data.rgb_parent_dirs` and
  `data.label_parent_dir` so source RGBs can be paired with segmentation
  masks.
- Both DDPM and DDIM are supported.
- Translation starts by noising the source image according to
  `noise_strength`, then denoises toward the target domain.
- Classifier-free guidance is implemented by predicting:
  - conditioned branch: `src -> tgt`
  - unconditioned branch: `src -> src`
  - combined as `eps_uncond + w * (eps_cond - eps_uncond)`

As of 2026-04-27, segmentation-aware runs no longer hard-code one control
image format. They now use `src/cyclenet/models/conditioning/spatial.py`:

- `build_condition_input(img, seg, cond_mode)`
- `build_seg_modulation_input(seg, use_spade)`
- `control_in_channels(cond_mode, num_seg_classes)`

Supported conditioning modes are:

- `cond_mode: rgb`
  - `c_img` is normalized RGB only
  - useful for the original image-conditioned CycleNet behavior, and for
    `rgb + SPADE(seg)` hybrids where RGB enters through ControlNet while the
    segmentation map separately modulates the residual blocks
- `cond_mode: seg`
  - `c_img` is one-hot segmentation only
  - useful for semantics-first translation and `seg-only + SPADE`
- `cond_mode: rgb_seg`
  - `c_img` is `concat(normalized_rgb, one_hot_seg)`
  - useful for the earlier mixed conditioning setup and for hybrid
    `rgb_seg + SPADE`

SPADE functionality:

- `use_spade: false`
  - builds the standard `ControlNet`
- `use_spade: true`
  - builds `SPADEControlNet`
  - forwards the original segmentation map separately as `seg`
  - applies spatially adaptive modulation inside SPADE residual blocks using:
    - global AdaGN domain modulation from `d_emb`
    - spatial segmentation modulation from resized one-hot masks

Default SPADE parameter:

- `s_dim` is the hidden channel width of the shared segmentation towers inside
  each SPADE site.
- `s_dim=128` is the current default and matches the repo’s typical
  `base_ch=128` and `d_dim=128`.

Important interpretation:

- `use_spade: true` with `cond_mode: seg` is a hybrid `seg feature injection +
  seg modulation` design, not a pure modulation-only SPADE generator.
- `use_spade: true` with `cond_mode: rgb` is the current cleanest way to test
  `RGB conditioning + SPADE(seg)`, where RGB provides detail hints and the
  segmentation map remains the spatial semantic controller.
- As of 2026-04-27, the active SPADE ablations are based on the `oem_only`
  pretrained UNet backbone and currently focus on:
  - `cond_mode: seg`, `use_spade: true`
  - `cond_mode: rgb`, `use_spade: true`
  The working hypothesis is that SPADE may let the model add more realistic,
  class-specific texture while reducing the boundary drift seen in earlier
  seg-only ControlNet runs without SPADE.

## Evaluation workflow

- `translate_sweep.py` and `translate_sweep_seg.py` generate deterministic
  translated candidate sets across checkpoint, CFG weight, and noise-strength
  grids.
- As of 2026-04-20, `src/cyclenet/eval/scripts/analysis/translate_sweep.py`
  supports `data.num_shards` / `data.shard_index` by partitioning sweep
  candidate combinations rather than source images, so each shard can write
  complete candidate outputs and metrics safely into the same eval directory.
- Sweeps compare each translated candidate against a fixed saved real reference
  subset and also compute a source-vs-real baseline.
- Reference manifests can be reused across reruns to keep comparisons stable.
- As of 2026-04-20, repeated `translate_sweep.py` runs in the same `eval.out_dir`
  preserve existing `metrics.csv` / `metrics.json` rows and append only missing
  candidate configs. Metric writes are merged under a file lock so concurrent
  shard runs do not clobber each other.
- `analyze_translate_sweep.py` converts metrics into normalized realism and
  preservation scores and ranks candidates.
- `analyze_translate_sweep.py` also supports sweep-level embedding projection
  plots using the DeepLab/CLIP/Inception settings from
  `configs/cyclenet/eval/project_translated.yaml`. The intended workflow is to
  fit PCA/UMAP once on a fixed sampled sim+real reference set, then transform
  translated candidate embeddings into that same space so movement across
  checkpoints/CFG/strength can be compared directly.
- As of 2026-04-22, `src/cyclenet/eval/scripts/project_cfg_grid.py` supports an
  explicit `plotting.points.rasterize` config flag so dense scatter clouds can
  be rasterized in PDF output while titles, lines, arrows, and other artists
  remain vectorized. This is useful for thesis figures where large point clouds
  can slow LaTeX/PDF rendering.
- As of 2026-04-22, `src/cyclenet/eval/scripts/project_cfg_grid.py` also uses
  `plotting.trajectory.edgecolor` for the start centroid marker. Previously the
  start marker edge was hard-coded to the trajectory color, so YAML-only
  changes could not fully disable trajectory marker outlines.
- As of 2026-04-22, `src/cyclenet/eval/scripts/project_cfg_grid.py` supports
  `plotting.trajectory.start_edgecolor` / `start_edgewidth` and
  `end_edgecolor` / `end_edgewidth` overrides. The defaults still fall back to
  the shared `edgecolor` / `edgewidth`, but the end centroid can now be styled
  independently from the other trajectory markers.
- As of 2026-04-22, `src/cyclenet/eval/scripts/project_cfg_grid.py` also
  supports `plotting.layout.summary_gap_ratio`, which inserts an empty spacer
  column before the trajectory summary so that gap can be widened without
  changing inter-CFG spacing. The vertical summary separator remains centered
  because it is computed from the rendered main-grid and summary-axis bounds.
- `project_translated.py` now also fixes plot axis limits from the cached
  sim+real reference coordinates only, with optional padding via
  `plotting.axis_pad_frac`, so translated runs do not visually rescale the
  reference clouds across plots.
- `project_translated.py` now skips embedder initialization entirely when the
  cached sim, real, and translated embedding `.npy` files already exist for
  the current run, making pure replotting much faster.
- As of 2026-04-25, `scripts/analyze_class_feature_distances.py` supports
  cross-class centroid analysis in addition to same-class reference
  comparisons. It now writes long-form and matrix CSVs plus heatmaps for
  comparison-class vs reference-class centroid L2/cosine distances, and when a
  baseline dataset is configured it also writes delta-vs-baseline heatmaps to
  highlight class drift such as movement toward buildings.
- As of 2026-04-25, that same script also writes per-class alignment summary
  CSVs and plots derived from the cross-class matrices, including nearest real
  class, own-class rank, and own-vs-buildings / own-vs-water margins, plus
  delta-vs-baseline versions to make class-attractor behavior easier to spot.
- `analyze_translate_sweep_consensus.py` summarizes best settings across
  multiple sweep directories.
- `project_translate_sweep_projections.py` adds PCA/t-SNE diagnostics.
- `scripts/plot_translate_sweep_seg_overlays.py` is the main visual sanity
  check for segmentation preservation.
- `scripts/plot_random_image_grid.py` is a minimal utility that recursively
  samples images from a directory and saves a simple grid using configurable
  `num_samples`, `n_rows`/`n_cols`, and horizontal/vertical panel spacing. It
  is intended for quick thesis/sample-figure assembly without rerunning any
  translation or projection workflow.
- As of 2026-04-22, `src/cyclenet/eval/plotting/model_comparison.py` builds
  thesis-style side-by-side comparison figures directly from saved
  `source_samples/` and `translated_samples/` grid directories. Each model
  column is a repeated `Input` / `Output` pair and can use its own fixed
  `(noise_strength, cfg_weight)` setting, so cross-model visual comparisons do
  not require every checkpoint to share the same selected operating point.
- `scripts/plot_lpips_vs_fid_scatter.py` merges translation-sweep LPIPS and
  FID CSVs on `(step, noise_strength, cfg_weight)`, saves the merged table, and
  plots LPIPS-vs-FID tradeoff figures. As of 2026-04-21, the default figure
  style is a cleaner thesis-oriented layout: one panel per checkpoint step,
  a single marker shape, color-coded noise strengths with thin lines following
  increasing CFG weight, an optional lower-left Pareto frontier, and optional
  highlighting of the best joint LPIPS/FID tradeoff point. It is intended for
  realism-vs-preservation tradeoff figures in the thesis workflow.
- As of 2026-04-22, `src/cyclenet/eval/plotting/pareto.py` saves the 3D Pareto
  scatter with manual subplot margins and disables the global tight save bbox
  for that figure, because Matplotlib 3D export can clip the z-axis label when
  combined with `tight_layout()` and `savefig.bbox="tight"`.
- As of 2026-04-22, `src/cyclenet/eval/plotting/pareto.py` also writes a
  staged selection figure for the `fid_deeplab_pareto_then_lpips` rule: a 3D
  overview, a `FID` vs `DeepLab FD` Pareto-survivor panel, and a front-only
  LPIPS strip that highlights the final lowest-LPIPS choice among survivors.
  The plotting helper recomputes `is_pareto_fid_deeplab` and the staged
  selected point from the merged metrics if those flags are absent in the CSV.
  The story figure's 3D overview now labels the task-aware metric as
  `DeepLabv3-FID`, uses `FID` as the vertical axis, and repositions that
  vertical axis to the opposite edge via Matplotlib 3D axis `_axinfo`
  juggling so the `FID` labels sit beside `DeepLabv3-FID` without changing the
  overall viewing orientation; the story figure also inverts the LPIPS axis so
  lower LPIPS visually decreases toward the center/good region. Because the
  z-axis is manually repositioned, the story figure also forces the `FID`
  z-label rotation to a fixed readable orientation. As of 2026-04-22,
  `src/cyclenet/eval/plotting/pareto.py` exposes pairwise/3D/story title size,
  legend size, title padding, and a manual story 3D title `y` offset directly
  in `main()` so thesis-figure typography can be tuned without editing plotting
  internals. Checkpoint legends/panel ordering are now also sorted by numeric
  `step` rather than lexicographic `checkpoint_name`, so labels appear in the
  expected order such as `2.5k, 10k, 20k, 30k`. The story figure now keeps
  selected-point stars in the 3D and Pareto-front panels but only draws text
  parameter annotations in the final LPIPS panel to reduce clutter.
- As of 2026-04-22, `src/cyclenet/eval/plotting/pareto.py` also provides a
  simple single-panel `plot_fid_lpips_tradeoff()` helper for clean
  realism-vs-preservation plots. It keeps checkpoint colors and Pareto front
  markers/lines but omits selected-point stars and text annotations.
- `scripts/analyze_checkpoint_metrics.py` performs cross-checkpoint selection
  analysis given per-checkpoint LPIPS, FID, and DeepLab-FD CSVs. It merges the
  metrics on `(step, noise_strength, cfg_weight)`, computes Pareto-optimal
  flags for `(DeepLab FD, LPIPS)`, `(FID, DeepLab FD)`, and for the full
  `(FID, LPIPS, DeepLab FD)` triplet, selects one operating point per
  checkpoint using a configurable rule, and writes merged CSVs plus a
  checkpoint summary table (including a LaTeX export) for thesis reporting.
- `src/cyclenet/eval/plotting/heatmap.py` now supports LPIPS, FID, and
  CLIP-FID sweep CSVs from the translation-evaluation helpers. As of
  2026-04-22, it also supports DeepLab feature-space Fréchet distance CSVs.
  It exposes
  metric-specific `save_lpips_heatmaps_from_csv()`,
  `save_fid_heatmaps_from_csv()`, and `save_clip_fid_heatmaps_from_csv()`
  entry points while sharing the same core grid/annotation plotting logic, and
  now also exposes `save_deeplab_fd_heatmaps_from_csv()`.
- `src/cyclenet/eval/plotting/pareto.py` plots cross-checkpoint tradeoff
  figures from the merged CSV produced by `scripts/analyze_checkpoint_metrics.py`.
  It generates a 3-panel pairwise tradeoff grid
  (`LPIPS/DeepLab FD`, `LPIPS/FID`, `DeepLab FD/FID`) plus a 3D scatter, with
  checkpoint colors, Pareto-optimal points highlighted, and selected operating
  points marked separately.
- `src/cyclenet/eval/fid.py` now runs at the granularity of a single
  `step-*` directory rather than scanning a whole translated-root tree. Point
  `step_dir` at one checkpoint folder, and it will evaluate only that step's
  `strength-* / cfg-*` subdirectories and save `fid_stats.csv` inside the
  same `step-*` directory. As of 2026-04-21, `main()` also exposes a
  `direct_pair` mode for one-off baseline comparisons such as `sim vs real`,
  writing a small CSV with the baseline FID value.
- `src/cyclenet/eval/lpips.py` now follows the same single-step workflow as
  `fid.py`: point `step_dir` at one `step-*` directory, and it will evaluate
  only that step's `strength-* / cfg-*` translated outputs against the source
  sim directory and save `lpips_stats.csv` inside the same `step-*` directory.
- `src/cyclenet/eval/clip_fid.py` mirrors that single-step sweep workflow for
  CLIP-space Fréchet distance. It reuses cached CLIP embeddings from
  `project_translated.py`, reading `reference_cache/clip/real_embed.npy` plus
  each candidate directory's `clip_translated_embed.npy`, and writes
  `clip_fid_stats.csv` inside the chosen `step-*` directory.
- `src/cyclenet/eval/deeplab_fd.py` now mirrors the same single-step sweep
  workflow for task-aware Fréchet distance in DeepLab feature space. Point it
  at one `step-*` directory, a DeepLab checkpoint, and a DeepLab reference
  cache directory; it will compute or reuse `real_embed.npy`, cache
  `deeplab_translated_embed.npy` inside each `strength-* / cfg-*` directory,
  and write `deeplab_fd_stats.csv` inside the chosen `step-*` directory.
- `src/cyclenet/eval/frechet_dist.py` is now the shared helper for Fréchet
  distance over arbitrary cached embedding arrays and is used by the CLIP and
  DeepLab distribution-distance scripts.
- `scripts/analyze_mask_dataset_by_class.py` provides mask-only sim-vs-real
  dataset mismatch analysis, including per-class area fractions, connected
  component counts, component size statistics, and boundary-density summaries.
- `DeepLabEmbedder.embed_by_class()` in `src/cyclenet/eval/embed.py` now
  supports class-masked feature pooling: it extracts spatial DeepLab features,
  resizes label masks to feature resolution with nearest-neighbor sampling, and
  returns per-image pooled feature vectors grouped by raw class id `1..8`.
- `scripts/cache_class_feature_vectors.py` caches reusable DeepLab
  class-conditional feature vectors for sim, real, and optional translated
  datasets. It is configured by
  `configs/cyclenet/eval/cache_class_feature_vectors.yaml` and writes
  per-dataset pair manifests plus `.npz` feature bundles so later distance
  computations do not need to rerun embedding extraction.
- `scripts/analyze_class_feature_distances.py` consumes those cached class
  feature bundles and computes per-class distance tables and plots against a
  reference dataset, including Fréchet distance, centroid cosine/L2, and
  baseline-improvement summaries. It can also optionally generate per-class
  UMAP plots using the shared projector/plotting helpers with UMAP and plotting
  settings defined directly in
  `configs/cyclenet/eval/analyze_class_feature_distances.yaml`.
- `scripts/cache_and_analyze_class_feature_distances.py` provides a one-shot
  workflow that first computes or reuses cached DeepLab class feature bundles,
  then immediately runs the per-class distance analysis and optional UMAP
  plots. It is configured by
  `configs/cyclenet/eval/cache_and_analyze_class_feature_distances.yaml`.
- `src/cyclenet/translate_cyclenet_sweep.py` provides a lightweight
  translation-only sweep for the RGB workflow across checkpoint step,
  CFG weight, and noise strength, writing each candidate to its own output
  directory without computing metrics.
- `src/cyclenet/eval/scripts/plot_cfg_str_grid.py` now supports a rerun-only
  plotting mode via `plotting.rerun_from_saved`, allowing grids to be rebuilt
  directly from an existing `translated_samples` directory without rerunning
  translation. Optional overrides can point to saved translated and source
  sample folders explicitly. As of 2026-04-21, it also supports
  `plotting.source_mode: grid_column`, which repeats the source image as the
  leftmost grid column for a more standard publication-style layout instead of
  using the older oversized standalone source panel. The shared
  `plot_image_grid()` helper also now supports placing the source label above
  the source column while keeping only the translated CFG labels on the bottom,
  plus an explicit left-side `ylabel_pad` knob for tightening the y-axis
  label. It also supports `title_span: full` so scripts such as
  `plot_cfg_str_grid.py` can center the figure title across the full
  source-plus-translation layout without changing x-label centering.

# Key decisions

- 2026-04-17: Treat the segmentation-conditioned CycleNet path as the primary
  active remote-sensing workflow. The RGB-only path still exists, but current
  repo momentum is around semantic preservation with RGB+seg conditioning.
- 2026-04-17: Keep the project scoped to exactly two domains with fixed index
  conventions `0=sim`, `1=real`. Many datasets, configs, losses, and sampling
  paths assume `tgt_idx = 1 - src_idx`.
- 2026-04-17: Continue using a two-stage recipe: pretrained domain-conditioned
  UNet first, then CycleNet fine-tuning with a mostly frozen backbone.
- 2026-04-17: Preserve segmentation masks across translation rather than trying
  to translate masks. In segmentation-conditioned losses, `seg_0` stays fixed
  while only the RGB image changes between `x_0` and pseudo-target `y_0`.
- 2026-04-17: Use translate sweeps plus quantitative ranking as the main model
  selection mechanism. Do not rely on single-checkpoint visual inspection.
- 2026-04-17: Balance real and sim batches explicitly with `DomainSampler`,
  accepting that each epoch is capped by the smaller domain.
- 2026-04-17: Treat `invar_unet_grad` as an explicit experiment knob rather
  than a fixed behavior. Both RGB and segmentation CycleNet training paths now
  support enabling or disabling UNet decoder/final-layer gradients for the
  invariance `x->y` pass from config.
- 2026-04-17: Add optional invariance-weight ramping for fresh runs via
  `model.invar_weight_start` and `model.invar_weight_ramp_steps`. Current
  recommended starting point for `invar_unet_grad=true` is ramping from `0.0`
  to `0.1` over about `15k` steps instead of turning full invariance pressure
  on immediately.
- 2026-04-17: Allow resume-time model overrides through
  `configs/cyclenet/resume_cyclenet.yaml` and `src/cyclenet/resume_cyclenet.py`
  for loss weights and `invar_unet_grad`, so 20k->30k fine-tunes can change
  optimization behavior without editing the original run config.
- 2026-04-18: Allow `resume_cyclenet.py` to override `train.lr` at resume time.
  If only the learning rate changes, the resume path still restores optimizer
  state and then rewrites each param group's LR to the requested override.
- 2026-04-18: Allow `resume_cyclenet.py` to override logging intervals at
  resume time (`loss_interval`, `ckpt_interval`, `sample_interval`). Logging
  overrides can be applied in-place on an existing run, or saved into a new
  branch config if `run.out_run_dir` is provided.
- 2026-04-19: `src/cyclenet/eval/scripts/project_translated.py` now supports a
  shared reference cache for projection analysis. Keep
  `data.reference_cache_dir` fixed across translated-dataset runs so sim/real
  embeddings and fitted PCA/UMAP projectors are reused; only translated
  embeddings should vary per output directory.
- 2026-04-27: Treat "boundary faithfulness under freer translation" as a key
  success criterion for segmentation-aware ControlNet experiments, not just
  raw realism. Recent observations suggest seg-only conditioning without SPADE
  can produce more OEM-like outputs than stricter RGB-conditioned runs, but it
  also drifts further from the source layout as training progresses.

## Non-obvious implementation details worth remembering

- `CycleNet.forward()` always runs the pretrained backbone encode path inside
  `torch.no_grad()`. The trainable backbone portion is decoder + final layer.
- The `no_unet_grad=True` path temporarily disables gradients on the decoder and
  final layer during forward passes used by some CycleNet losses.
- Training sample visualization now uses random source-sample batches again
  (`shuffle=True`) in the training and resume paths. The old checkpoint
  `sample_batch_idx` bookkeeping was removed, so sample preview streams no
  longer resume from the same exact source batch order across restarts.
- With `invar_unet_grad=false`, the invariance loss behaves more like a
  frozen-teacher objective for the shared decoder/final layer. With
  `invar_unet_grad=true`, the invariance branch directly updates that trainable
  target-side path, which materially changes optimization behavior.
- As of 2026-04-27, one important observed failure mode in downstream
  segmentation experiments is apparent class collapse toward dominant-looking
  real classes such as buildings and water. The current hypothesis is that
  translated images sometimes place class-specific real textures onto the
  wrong semantic regions, which can improve realism superficially while
  harming class correctness.
- `project_translated.py` intentionally reuses cached sim/real reference
  manifests, embeddings, and fitted projectors. Translated embeddings are
  cached per embedding model under `data.out_dir` as
  `{model}_translated_embed.npy`, so repeated UMAP/PCA runs with the same
  embedder reuse translated features while different embedders keep separate
  caches.
- `project_translated.py` also supports a template-driven sweep mode across
  `embedding model`, `projection method`, `step`, `cfg`, and
  `noise_strength` combinations. In sweep mode, keep
  `data.reference_cache_dir` fixed, set `data.translated_dir_template`, and
  optionally set `data.out_dir_template`; otherwise outputs fall back to
  `data.out_dir / <model> / <method> / step-..._strength-..._cfg-...`.
- `project_cfg_trajectory.py` is a lightweight follow-on analysis that reads
  cached translated embedding arrays from an existing projection root, reuses
  the cached reference projector/coords, and plots CFG centroid trajectories
  at fixed step and noise strength without KDE or marginal panels.
- `project_cfg_grid.py` builds a comparison figure over fixed CFG weights with
  one projection panel per CFG value plus a trajectory summary panel per row.
  It reads only cached translated embedding arrays and shared reference caches,
  and exposes optional KDE/marginal overlays through config. As of 2026-04-20,
  the cleaner default presentation is shared column headers instead of repeated
  per-panel CFG titles, boxed subplot frames, optional global axis labels, and
  an optional vertical separator before the trajectory column. Its
  translated-embedding lookup now assumes
  `projection_root/step-.../strength-.../cfg-.../<model>_translated_embed.npy`
  rather than an extra `<model>/` subdirectory, and the trajectory panel now
  supports unlabeled centroid ordering via configurable arrow rendering. As of
  2026-04-21, CFG-grid panels also default to `plotting.panels.force_square:
  true` with `summary_width_ratio: 1.0`, so changing row count or figure
  height does not leave the projection panels narrower than the trajectory
  summary panels. The script now also supports a row-per-translation-setting
  mode via `comparison.rows` plus a fixed `comparison.embedding_model`, which
  is intended for thesis figures that compare checkpoints such as `2.5k` vs
  `30k` across the same CFG sweep in a single embedding space. As of
  2026-04-21, each row can also override `projection_root`, so one figure can
  compare translated candidates coming from entirely different parent model
  directories instead of assuming all rows share one common projection root.
  As of 2026-04-21, `plotting.legend.position: top_right` places the legend in
  the header band on the right and automatically keeps it between the suptitle
  and the shared column headers in the default layout to avoid overlap. The
  legend placement now checks horizontal overlap before moving, so with
  `plotting.legend.reference: figure` it can stay tucked into the top-right
  corner instead of being pushed below the centered title when the two do not
  actually intersect. As of 2026-04-21, the default thesis-style CFG-grid
  labeling convention places CFG values as bottom numeric labels under the
  non-trajectory projection columns and centers `CFG weight ($w$)` under those
  columns only, instead of using top `w = ...` headers for the CFG panels.
- In CycleNet losses, the repo follows the official implementation behavior,
  which differs from the paper in some conditioning details. The notes in
  `info/cyclenet_info.md` and `info/loss_grads.md` are the source of truth for
  how this repo actually computes those losses.
- EMA models are only maintained on rank 0 / main process in the current
  training code.
- Evaluation scripts expect local model assets for CLIP and Inception in the
  configured cache paths; they are not written as online-download-first tools.
- `project_translated.py` writes cache metadata and reference path manifests in
  the shared reference cache directory. If sim/real roots, embedding model, or
  sampling settings change, point it at a new cache directory or remove the
  stale cache instead of silently reusing mismatched artifacts.
- In `project_translated.py`, setting `embedding.num_samples` to `null` means
  "use all collected images" for sim, real, and translated sets. That `null`
  value is part of the reference-cache metadata, so it is intentionally
  distinguished from a finite sample count.
- `plot_cfg_str_grid.py` now saves individual translated outputs alongside the
  summary grid PDFs. The per-sample layout is
  `translated_samples/<sample_name>/strength-x.xx/cfg-x.x/img.png`, matching
  the training sample naming convention closely enough to compare settings
  without resampling.
  treated as distinct from any fixed sample count.
- In DDIM translation, larger `noise_strength` both starts from a noisier point
  and uses more reverse denoising steps in this repo's implementation because
  strength selects the truncated starting index from the uniformly spaced DDIM
  schedule. Higher CFG still pushes harder toward the real-domain branch, but
  artifact severity is not strictly monotonic in `cfg * strength`.

# Constraints and conventions

- Python target is 3.10 via `environment.yml`.
- The project uses editable install mode (`pip -e .`).
- Primary runtime stack: PyTorch, torchvision, albumentations, torchmetrics,
  transformers, lpips, umap-learn, OmegaConf.
- The codebase assumes GPU training/inference and supports `torchrun`-style DDP.
- `DomainSampler` requires even batch sizes because each batch is split 50/50
  across domains.
- TensorBoard is the main training log sink.
- Checkpoints are stored as `step-<n>.ckpt` and usually sampled/evaluated via
  `ema_model`.
- Many helper tools are oriented around remote-sensing imagery and may embed
  that assumption in defaults, class labels, or directory names.

# Open issues / risks

- The repo is not general multi-domain CycleNet yet; many paths are explicitly
  two-domain only.
- Absolute dataset/output/model paths in configs make reproduction
  environment-specific unless configs are edited first.
- The segmentation path assumes mask values `0..8` with ignore=`0`; any new
  dataset with a different label encoding will need adapter work.
- Some practical target domains may have no reliable real segmentation masks at
  all, including pseudo-labels, so methods that depend on real-mask
  conditioning should be treated as optional experiments rather than assumed
  training requirements.
- Because `DomainSampler` truncates to the smaller domain each epoch, very
  imbalanced datasets underuse the larger domain unless sampling strategy is
  revisited.
- FID and CLIP metrics in sweeps are ranking heuristics, not publication-grade
  scores, especially at the smaller default sample counts.
- Preservation can still regress even when realism metrics improve, so overlay
  checks and LPIPS should remain part of candidate selection.
- Seg-only ControlNet conditioning without SPADE has recently looked more
  realistic and more OEM-like than expected, but it may also reduce source
  faithfulness and disturb semantic boundaries later in training.
- If translated appearance drifts across class boundaries, downstream
  segmentation models can collapse toward visually dominant classes such as
  buildings or water even when the source mask geometry is correct.
- Invariance loss has repeatedly shown a tendency to fall early and then rise
  later in training, including runs resumed from `20k` with larger invariance
  weight. This suggests the issue is not solved by weight increases alone and
  may reflect a deeper optimization conflict with reconstruction-dominated
  training.
- Early fresh runs with `invar_unet_grad=true` can develop black-dot / blob
  artifacts surprisingly early. In recent observations, these became noticeable
  by roughly `7.5k` steps at moderate settings and were clearly present by
  `10k` even at low `noise_strength` / modest CFG, so full-strength invariance
  gradients from step 1 are risky.
- There is ongoing local work in eval/translation files in the git worktree.
  Future edits should inspect current uncommitted changes before touching those
  areas.

# Next recommended steps

- Keep `docs/PROJECT_CONTEXT.md` aligned with the segmentation-conditioned
  remote-sensing workflow as the code evolves.
- When working on evaluation, prefer updating the sweep/ranking path
  (`translate_sweep*`, `analyze_translate_sweep*`, overlay/projection scripts)
  instead of introducing one-off notebook logic.
- If future work broadens beyond two domains, plan a deliberate refactor of:
  `DomainEmbedding`, dataset index conventions, loss code, and CFG sampling.
- If new datasets are introduced, document their RGB/mask folder conventions and
  label encoding here immediately because the data contract is easy to break.
- Before changing CycleNet losses or gradient flow, re-read
  `info/cyclenet_info.md` and `info/loss_grads.md`; those notes capture the
  current intended implementation behavior better than the original paper.
- For fresh `invar_unet_grad=true` experiments, prefer a conservative starting
  recipe before broader sweeps: `lr=5e-6`, `recon=1.0`, `cycle=0.01`,
  `consis=0.1`, `invar=0.1`, `invar_weight_start=0.0`,
  `invar_weight_ramp_steps=15000`.
- When comparing grad/no-grad invariance experiments, monitor fixed sample
  settings early in training. Moderate settings such as around
  `strength=0.3, cfg=2..3` and low-noise settings such as `strength=0.1, cfg=2`
  have been useful early-warning probes for dot/blob failure modes.
