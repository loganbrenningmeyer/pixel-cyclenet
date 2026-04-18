# Project overview

`pixel-cyclenet` is a PyTorch implementation of pixel-space CycleNet for
unpaired image-to-image translation between two domains, with the current
project focus on remote-sensing simulated-to-real translation.

The repo supports two related workflows:

- Stage 1: pretrain a domain-conditioned diffusion UNet on both domains.
- Stage 2: train a CycleNet model that reuses the pretrained UNet backbone,
  adds a ControlNet-style side branch, and optimizes CycleNet reconstruction,
  cycle, consistency, and invariance losses.

The codebase now has two CycleNet variants:

- RGB-conditioned CycleNet.
- Segmentation-conditioned CycleNet, where ControlNet sees source RGB plus an
  8-channel land-cover mask. This segmentation-conditioned path is the current
  active remote-sensing workflow.

# Current goals

- Improve sim-to-real remote-sensing translation quality while preserving
  land-cover geometry needed for downstream segmentation.
- Use segmentation-aware conditioning as the main mechanism for stronger
  semantic consistency during CycleNet translation.
- Evaluate checkpoints and sampling settings with translate-sweep runs instead
  of selecting models only by ad hoc visual inspection.
- Rank sweep candidates by realism metrics against a fixed real reference set,
  while keeping LPIPS and overlay visualizations as preservation guardrails.

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
- `DomainEmbedding` is a learned 2-entry embedding table. The whole project is
  currently hard-coded around exactly two domains: sim=`0`, real=`1`.
- `ControlNet` is created by deep-copying the pretrained UNet stem, encoder,
  bottleneck, and time MLP, then adding zero-initialized 1x1 control
  convolutions.
- `CycleNet` combines:
  - source-domain embedding for the ControlNet branch
  - target-domain embedding for the backbone branch
  - ControlNet skips summed into the backbone bottleneck and decoder skips

## Two-stage training pipeline

1. UNet pretraining:
   - Entrypoint: `src/cyclenet/train_unet.py`
   - Trains a domain-conditioned diffusion UNet plus `DomainEmbedding`.
   - Uses mixed-domain batches balanced 50/50 by `DomainSampler`.
   - Saves both live and EMA weights.

2. CycleNet training:
   - Entrypoints:
     - RGB: `src/cyclenet/train_cyclenet.py`
     - RGB+seg: `src/cyclenet/train_cyclenet_seg.py`
   - Loads the pretrained UNet EMA checkpoint as the CycleNet backbone.
   - Freezes the backbone stem, time MLP, encoder, bottleneck, and
     `DomainEmbedding`.
   - Trains ControlNet plus the UNet decoder/final layer.
   - Maintains a rank-0 EMA copy for evaluation/sampling.

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
- Both DDPM and DDIM are supported.
- Translation starts by noising the source image according to
  `noise_strength`, then denoises toward the target domain.
- Classifier-free guidance is implemented by predicting:
  - conditioned branch: `src -> tgt`
  - unconditioned branch: `src -> src`
  - combined as `eps_uncond + w * (eps_cond - eps_uncond)`

For segmentation runs, the control image is:

- `build_seg_condition(img, seg) = concat(normalized_rgb, one_hot_seg)`

This means the ControlNet sees `3 + num_seg_classes` channels, currently `11`
for the default remote-sensing setup.

## Evaluation workflow

- `translate_sweep.py` and `translate_sweep_seg.py` generate deterministic
  translated candidate sets across checkpoint, CFG weight, and noise-strength
  grids.
- Sweeps compare each translated candidate against a fixed saved real reference
  subset and also compute a source-vs-real baseline.
- Reference manifests can be reused across reruns to keep comparisons stable.
- `analyze_translate_sweep.py` converts metrics into normalized realism and
  preservation scores and ranks candidates.
- `analyze_translate_sweep.py` also supports sweep-level embedding projection
  plots using the DeepLab/CLIP/Inception settings from
  `configs/cyclenet/eval/project_translated.yaml`. The intended workflow is to
  fit PCA/UMAP once on a fixed sampled sim+real reference set, then transform
  translated candidate embeddings into that same space so movement across
  checkpoints/CFG/strength can be compared directly.
- `analyze_translate_sweep_consensus.py` summarizes best settings across
  multiple sweep directories.
- `project_translate_sweep_projections.py` adds PCA/t-SNE diagnostics.
- `scripts/plot_translate_sweep_seg_overlays.py` is the main visual sanity
  check for segmentation preservation.

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
- In CycleNet losses, the repo follows the official implementation behavior,
  which differs from the paper in some conditioning details. The notes in
  `info/cyclenet_info.md` and `info/loss_grads.md` are the source of truth for
  how this repo actually computes those losses.
- EMA models are only maintained on rank 0 / main process in the current
  training code.
- Evaluation scripts expect local model assets for CLIP and Inception in the
  configured cache paths; they are not written as online-download-first tools.
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
- Because `DomainSampler` truncates to the smaller domain each epoch, very
  imbalanced datasets underuse the larger domain unless sampling strategy is
  revisited.
- FID and CLIP metrics in sweeps are ranking heuristics, not publication-grade
  scores, especially at the smaller default sample counts.
- Preservation can still regress even when realism metrics improve, so overlay
  checks and LPIPS should remain part of candidate selection.
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
