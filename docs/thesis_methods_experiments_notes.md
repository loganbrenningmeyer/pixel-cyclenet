# Thesis Notes: Methods and Experiments for Pixel-Space CycleNet

This document maps your planned thesis sections to the current `pixel-cyclenet` implementation. It is written as implementation-grounded notes rather than final thesis prose, so you can turn each subsection into your own writing while staying faithful to what the code actually does.

The notes below assume the primary system you are describing is the current segmentation-conditioned remote-sensing pipeline:

- `src/cyclenet/train_cyclenet_seg.py`
- `src/cyclenet/training/cyclenet_trainer_seg.py`
- `src/cyclenet/translate_cyclenet_seg.py`
- `src/cyclenet/eval/translate_sweep_seg.py`
- `configs/cyclenet/train_cyclenet_seg.yaml`
- `configs/cyclenet/translate_cyclenet_seg.yaml`
- `configs/cyclenet/eval_translate_sweep_seg.yaml`

If your thesis also compares against the RGB-only CycleNet variant, keep that as a separate ablation or baseline and explicitly state which results come from `train_cyclenet.yaml` versus `train_cyclenet_seg.yaml`.

## Methods

### Training Setup

#### Datasets

What you should explain:

- The project uses a two-stage training pipeline.
- Stage 1 trains a domain-conditioned diffusion UNet on both simulated and real imagery.
- Stage 2 initializes CycleNet from the pretrained UNet and fine-tunes it for unpaired image-to-image translation.
- The active remote-sensing workflow is simulated-to-real translation with segmentation-aware conditioning.

Implementation-grounded details to include:

- The code is hard-coded around exactly two domains:
  - simulated domain index = `0`
  - real domain index = `1`
- Domain labels are not inferred dynamically. Many parts of the implementation assume `tgt_idx = 1 - src_idx`.
- In the current segmentation-conditioned config, the source domain is:
  - simulated remote-sensing imagery under `data.src_dir`
- The target domain is:
  - real remote-sensing imagery under `data.tgt_dir`
- In the checked-in segmentation config, the active run is specifically:
  - simulated `synrs3d` tiles to real `OEM` tiles
- The dataloaders recursively scan directories and keep only image files with extensions:
  - `.jpg`, `.png`, `.tif`, `.tiff`
- Image discovery can be filtered by immediate parent directory names using `rgb_parent_dirs`.
- In the active segmentation config, only images inside `opt/` are used.
- The RGB-only config is broader and allows both `opt/` and `pre_opt/`.

Segmentation-specific details to include:

- The segmentation-conditioned pipeline requires an RGB image and a corresponding mask with the same filename.
- The expected directory convention is:
  - `.../<scene>/opt/<image>.tif`
  - `.../<scene>/gt_ss_mask/<image>.tif`
- The mask directory name is controlled by `label_parent_dir`, which defaults to `gt_ss_mask`.
- If a matching mask is missing, the dataset loader raises an error rather than silently skipping the image.
- Masks are converted to one-hot semantic tensors after augmentation.

Mask semantics you should state explicitly:

- The implementation assumes 8 semantic classes.
- Raw mask value `0` is treated as ignore / unlabeled background.
- Valid semantic labels are `1..8`.
- After conversion:
  - class `1` maps to channel `0`
  - class `8` maps to channel `7`
- Ignore pixels become all-zero vectors across the 8 channels.

The class names are not used inside the training code, but the repo’s visualization scripts assume the following label semantics:

- `1`: Bareland
- `2`: Rangeland
- `3`: Developed Space
- `4`: Road
- `5`: Trees
- `6`: Water
- `7`: Agriculture land
- `8`: Buildings

Preprocessing and augmentation details to describe:

- All images are resized to `256 x 256` in the active configs.
- RGB images are normalized with mean `0.5` and standard deviation `0.5` per channel, which maps them to `[-1, 1]`.
- Segmentation masks are always resized with nearest-neighbor interpolation.
- The active remote-sensing transform preset is `transform_id = 3`.
- For UNet pretraining, `transform_id = 3` applies:
  - random horizontal flips
  - random vertical flips
  - random 90-degree rotations
  - mild brightness/contrast jitter
  - mild gamma jitter
- For segmentation-conditioned CycleNet training, `transform_id = 3` applies:
  - random horizontal flips
  - random vertical flips
  - random 90-degree rotations
- Photometric jitter is omitted in the segmentation-conditioned training transform.

How to explain why this makes sense:

- Overhead imagery is approximately rotation-equivariant, so 90-degree rotations are a natural augmentation.
- Nearest-neighbor mask interpolation is necessary to avoid corrupting discrete class labels.
- The segmentation-conditioned path keeps the semantic map fixed while translating only appearance.

Important implementation caveat to mention:

- The checked-in configs use absolute machine-specific paths. In the thesis, describe the dataset organization and the specific subsets used, not the literal filesystem paths.

Recommended thesis wording angle:

- Describe this section as "dataset usage within the training pipeline" rather than as the full dataset chapter.
- If your thesis already has a dedicated data section, keep this subsection focused on:
  - which splits fed which training stage
  - how masks were paired
  - how samples were normalized and augmented

What is still not fully decided and what I recommend:

- You should explicitly state whether the real reference images used for distribution-level evaluation are disjoint from the real images used in diffusion training.
- The current evaluation config defaults to reusing the CycleNet training target directory if `eval.real_dir` is not overridden.
- Recommendation:
  - For the thesis, use a held-out real evaluation subset if possible.
  - If you cannot, state clearly that realism metrics measure closeness to the training-domain distribution rather than strict out-of-sample generalization.

#### Sampling Strategy

What you should explain:

- Sampling strategy here means both dataset sampling and diffusion-time sampling during training.

Dataset sampling details:

- Both UNet pretraining and CycleNet training use balanced mixed-domain batches.
- The code concatenates datasets in the order:
  - real dataset first
  - simulated dataset second
- `DomainSampler` then creates each batch with exactly half real samples and half simulated samples.
- This requires an even batch size.
- The active training batch size is `16`, so each global batch contains:
  - `8` real samples
  - `8` simulated samples
- Under distributed training, the sampler shards complete batches across ranks rather than letting each rank rebalance independently.

Important effect you should mention:

- Each epoch is capped by the smaller domain.
- The number of usable samples per epoch is:
  - `min(n_real, n_sim)` after balancing
- This means the larger domain is partially under-sampled each epoch, but domain balance is preserved.

Diffusion-time sampling during training:

- For every training batch, the trainer samples one diffusion timestep `t` per image uniformly from `[0, T-1]`.
- It then samples Gaussian noise and forms the noised image `x_t` using the standard forward diffusion process.
- This happens independently for each training step.

CycleNet-specific sampling logic:

- Each sample carries:
  - `src_idx`
  - `tgt_idx = 1 - src_idx`
- Because both real and simulated images are present in every batch, the model learns both directions:
  - simulated to real
  - real to simulated
- Even though your practical use case is likely simulated to real translation, the training objective is symmetric at the batch-construction level.

Segmentation-conditioned control image:

- The conditioning tensor passed to ControlNet is:
  - `concat(normalized_rgb, one_hot_segmentation)`
- With 3 RGB channels and 8 segmentation channels, the ControlNet condition has 11 channels.

How to explain the semantic preservation design:

- In cycle and invariance terms, the RGB part of the condition changes from source-style to pseudo-target-style, but the segmentation part is kept fixed.
- This reflects the intended behavior of the translator:
  - change appearance
  - preserve land-cover geometry and semantics

Important implementation fact to state explicitly:

- There is no classifier-free dropout during training in the current code.
- The UNet pretraining stage does not use any conditional dropout.
- The CycleNet training code also does not randomly drop or replace conditioning during optimization.
- Classifier-free guidance is only constructed at inference time by combining two forward passes.

That matters because:

- You should not claim that the model was trained with standard CFG dropout.
- The guidance mechanism in this repo is an inference-time domain-interpolation trick built from the same trained model.

#### Hyperparameters

The cleanest way to write this subsection is to separate:

- stage-1 UNet pretraining hyperparameters
- stage-2 CycleNet fine-tuning hyperparameters
- inference-monitoring hyperparameters used during training

##### Stage 1: domain-conditioned UNet pretraining

Model hyperparameters:

- Input channels: `3`
- Base channels: `128`
- Time embedding dimension: `512`
- Domain embedding dimension: `128`
- Channel multipliers: `[1, 2, 4, 4]`
- Residual blocks per stage: `2`
- Encoder attention heads: `[0, 0, 0, 4]`
- Bottleneck attention heads: `4`
- Residual dropout: `0`
- Attention dropout: `0`
- Feed-forward dropout: `0`

Optimization hyperparameters:

- Steps: `600000`
- Batch size: `64`
- Optimizer: `AdamW`
- Learning rate: `1e-4`
- Weight decay: `1e-4`
- EMA decay: `0.9999`
- Gradient clipping: max norm `1.0`
- Mixed precision: enabled through `torch.amp.autocast` and `GradScaler`

Parameter-group details worth mentioning:

- Bias parameters and normalization parameters are excluded from weight decay.
- Domain embedding parameters are also placed in a no-weight-decay parameter group.

Diffusion hyperparameters:

- Beta schedule: linear
- Number of diffusion steps `T`: `1000`
- `beta_start = 1e-4`
- `beta_end = 0.02`
- `s = 0.008` is still provided, although it is only relevant for cosine schedules
- Prediction target: epsilon / noise prediction

Training objective:

- Standard diffusion noise-prediction MSE:
  - the UNet predicts the noise added to `x_0`

##### Stage 2: CycleNet fine-tuning

Optimization hyperparameters:

- Steps: `50000`
- Batch size: `16`
- Optimizer: `AdamW`
- Learning rate: `1e-5`
- Weight decay: `1e-4`
- EMA decay: `0.9999`
- Gradient clipping: max norm `1.0`
- Mixed precision: enabled

Loss weights in the active segmentation config:

- reconstruction weight: `1.0`
- cycle weight: `0.01`
- consistency weight: `0.1`
- invariance weight: `0.1`
- invariance weight start: `0.0`
- invariance ramp steps: `15000`
- `invar_unet_grad = true`

What the invariance ramp means:

- The invariance term is not applied at full strength from the start.
- Its coefficient increases linearly from `0` to `0.1` over the first `15000` optimization steps.
- This helps avoid early over-regularization before the translation branch has stabilized.

Frozen versus trainable parameters:

- The CycleNet model is initialized from the pretrained UNet EMA checkpoint.
- The following components are frozen:
  - backbone stem
  - time MLP
  - backbone encoder
  - bottleneck / mid block
  - domain embedding
- The main trainable components are:
  - ControlNet
  - UNet decoder
  - UNet final layer

Important implementation nuance:

- In some loss terms, the code temporarily disables gradients for the decoder and final layer during the `x -> y` forward pass.
- This makes the translation branch more ControlNet-heavy and more conservative than a fully trainable backbone update would be.

Why this matters for your thesis:

- It helps explain why the model tends to preserve structure strongly and why stronger inference steering may be needed to produce visibly stronger target-domain appearance changes.

Loss implementation details you should report carefully:

- The repo follows an implementation that is close to, but not identical with, the equations as written in the original CycleNet paper.
- In particular, the code uses:
  - source embedding for the ControlNet branch
  - target embedding for the backbone branch
- The actual losses are:
  - reconstruction loss
  - cycle loss
  - consistency loss
  - invariance-style regularization
- The code separates cycle consistency into two terms:
  - `cycle`
  - `consis`
- The implementation also uses `detach()` and selective gradient blocking in several places.

Recommendation for how to explain this:

- Do not over-claim exact equation-level equivalence to the paper.
- State that your implementation follows the paper’s overall CycleNet objective structure but uses the repository’s concrete loss formulation.
- If needed, include an appendix or footnote noting that the conditioning and gradient-flow details follow the actual implementation rather than the paper’s idealized notation.

Training-monitoring sampling settings:

- During CycleNet training, sample visualizations are generated every `2500` steps.
- The training-time sampling config uses:
  - sampler: DDIM
  - DDIM steps: `50`
  - `eta = 0`
  - CFG weights for sample grids: `[1, 2, 3, 4, 5]`
  - noise strengths: `[0.1, 0.2, 0.3, 0.4, 0.5]`

You should describe these as:

- qualitative monitoring settings
- not the final evaluation sweep by themselves

### Inference Procedure

What you should explain:

- Inference translates a source image by partially noising it and then denoising it toward the target domain using the trained CycleNet.
- The translation process reuses the EMA checkpoint rather than the raw training weights.

Implementation-grounded steps:

1. Load the saved CycleNet training config.
2. Recover the original UNet training config from the pretrained checkpoint path.
3. Rebuild:
   - UNet backbone
   - domain embedding
   - ControlNet
4. Set the ControlNet input channels to `3 + num_seg_classes = 11`.
5. Load the CycleNet EMA checkpoint.
6. Build the diffusion schedule from the UNet config.
7. Load source images and matching source segmentation masks.
8. Form the control image as concatenated source RGB and one-hot segmentation.
9. Set:
   - `src_idx` from the translation config
   - `tgt_idx = 1 - src_idx`
10. Run DDIM or DDPM translation.
11. Save translated outputs while preserving each source image’s relative path.

Important implementation detail:

- The output image is clamped to `[-1, 1]`, converted to `[0, 1]`, and written to disk as an RGB image.

How to explain the role of the segmentation map at inference:

- The segmentation-conditioned model requires a source segmentation tensor at inference, not only during training.
- In the current implementation, inference uses the source mask directly.
- The translator therefore assumes segmentation information is available at test time.

What you need to decide and explain clearly in the thesis:

- Are you assuming ground-truth source masks are available during translation?
- Or are you assuming a separate segmentation model produces them?

Recommendation:

- If your actual pipeline uses ground-truth sim masks when translating simulated training data, say that explicitly.
- That is a valid setup for generating translated synthetic training data.
- If you later want deployment-time translation of unlabeled inputs, that is a different problem and should be presented as future work unless you also train or provide a segmentation predictor.

#### Classifier-Free Guidance

What you should explain mathematically:

- The implementation uses two forward passes at every denoising step:
  - a conditional translation pass
  - an unconditioned identity-domain pass
- The final noise prediction is:

`eps_cfg = eps_uncond + w * (eps_cond - eps_uncond)`

where `w` is the CFG weight.

What the two branches are in this repo:

- Conditional branch:
  - ControlNet uses the source-domain embedding
  - UNet backbone uses the target-domain embedding
  - This corresponds to `src -> tgt`
- Unconditional branch:
  - ControlNet uses the source-domain embedding
  - UNet backbone also uses the source-domain embedding
  - This corresponds to `src -> src`

How to explain the interpretation:

- The unconditional branch serves as an identity-preserving baseline.
- The conditional branch pushes the denoising trajectory toward the target domain.
- CFG scales the difference between those two predictions.

What happens as `w` increases:

- Lower values keep the output closer to the source image.
- Higher values produce stronger target-domain stylization.
- Excessively high values can overshoot and introduce artifacts or distort spatial structure.

Important implementation caveat:

- The model is not trained with explicit CFG dropout.
- So the thesis should present CFG here as an inference-time guidance construction, not a training-time conditioning-dropout scheme.

Recommended thesis interpretation:

- Frame CFG as a controllable tradeoff between:
  - appearance translation strength
  - structural preservation
- This fits your downstream segmentation motivation well.

What results you should show:

- A grid or sweep over `w` values, ideally matched with LPIPS and realism metrics.
- In the current evaluation setup, the tested values are:
  - `1.0, 2.0, 3.0, 4.0, 5.0`

Recommended thesis claim:

- Do not claim that larger CFG is always better.
- Argue instead that there is an optimal intermediate guidance range where realism improves without unacceptable damage to label-relevant structure.

#### DDIM Sampling

What you should explain conceptually:

- Translation in this repo is not pure unconditional generation from Gaussian noise.
- It starts from a source image, noises that image to a chosen diffusion level, and then denoises it toward the target domain.

How the implementation chooses the starting noise level:

- A `noise_strength` parameter in `[0, 1]` selects how far into the diffusion process the source image is pushed.
- For DDIM, the full diffusion timeline is first subsampled into `num_steps` uniformly spaced timesteps.
- The maximum starting DDIM index is:
  - `int(strength * (num_steps - 1))`
- The source image is then noised to the corresponding timestep using the forward diffusion process.

What this means physically:

- If `strength` is small:
  - more of the source signal is retained
  - translation is gentler
- If `strength` is large:
  - more source information is destroyed before reverse denoising
  - translation can be stronger but may better match the target distribution at the cost of preservation

The forward noising equation you can describe:

`x_t = sqrt(alpha_bar_t) * x_src + sqrt(1 - alpha_bar_t) * eps`

DDIM-specific details:

- The active evaluation config uses:
  - sampler: DDIM
  - DDIM steps: `100`
  - `eta = 0`
- `eta = 0` makes sampling deterministic for a fixed noise seed and checkpoint.
- This is important because it makes sweep comparisons more controlled and reproducible.

Why DDIM is a good fit for your experiments:

- It is faster than full DDPM sampling.
- It provides deterministic behavior at `eta = 0`.
- It makes it practical to sweep checkpoints, CFG values, and noise strengths.

How to explain the role of `noise_strength` in your results:

- `noise_strength` is the main knob that decides how much of the source image is preserved before guided denoising begins.
- It therefore interacts strongly with CFG:
  - high strength + high CFG is the most aggressive regime
  - low strength + moderate CFG is usually more preservation-friendly

Recommended results framing:

- Treat `noise_strength` and CFG jointly, not independently.
- Heatmaps over both variables are more informative than reporting only one-dimensional sweeps.

## Experiments

### Experimental Setup

What you should explain:

- Your evaluation is built around deterministic translation sweeps rather than ad hoc checkpoint selection by visual inspection.
- Each sweep translates a fixed source subset across a grid of checkpoints and sampling settings, then compares each translated candidate against a fixed real reference subset.

#### Checkpoints for Evaluation

What the current implementation does:

- The segmentation sweep config explicitly evaluates:
  - `step-10000.ckpt`
  - `step-20000.ckpt`
  - `step-30000.ckpt`
  - `step-40000.ckpt`
- The evaluation loads the `ema_model` weights rather than the raw model weights.

Why this is defensible:

- CycleNet training saves checkpoints every `2500` steps.
- Evaluating every saved checkpoint would be expensive and noisy.
- A coarser spacing of every `10000` steps gives coverage across early, middle, and later training while keeping the sweep tractable.

What you should probably add:

- If the final training run reaches `50000` steps, you should strongly consider also evaluating:
  - `step-50000.ckpt`
- Otherwise reviewers may ask why the final checkpoint was omitted.

Recommended thesis wording:

- Say that checkpoints were sampled at regular intervals across the training trajectory to capture changes in translation quality over optimization, while using EMA weights for stability at inference.

If results are not done yet, I recommend:

- Evaluate `50000` as well.
- If one interval clearly wins, also evaluate one or two neighboring checkpoints at finer resolution.
- That gives you a stronger argument that the chosen checkpoint is not a one-off artifact of coarse sampling.

#### Inference Settings

What the sweep currently evaluates:

- CFG weights:
  - `1.0, 2.0, 3.0, 4.0, 5.0`
- Noise strengths:
  - `0.1, 0.2, 0.3, 0.4, 0.5`
- Sampler:
  - DDIM
- DDIM steps:
  - `100`
- `eta = 0`
- Translation seed:
  - `42`

Why these settings matter:

- The seed is reset before every candidate translation run.
- That means all candidates are evaluated using the same underlying noise realization.
- This makes comparisons cleaner because differences are due to the model and sampling parameters, not different random draws.

How to describe this in the thesis:

- The evaluation uses deterministic source selection and deterministic DDIM translation settings so that checkpoint and hyperparameter comparisons are directly comparable.

Recommended analysis structure:

- Report the full grid as a heatmap or table.
- Then identify the best region rather than only one winning point.
- If several nearby settings behave similarly, state that the method is robust in that region.

If you have not finalized the inference grid, I recommend:

- Keep the current grid for the main sweep.
- For downstream segmentation pilots, focus on a narrower range around the best-performing region.
- Based on the current implementation and intended use, moderate values are likely the most useful:
  - CFG around `2` to `3`
  - noise strength around `0.2` to `0.3`

#### Evaluation Data

What the current sweep actually does:

- It chooses a deterministic subset of source simulated images.
- It chooses a deterministic subset of real reference images.
- It saves both subsets to disk under a `reference/` directory.
- It stores manifests so the same subset can be reused across reruns.

Current default sizes:

- source sample size: `256`
- real sample size: `512`
- seed: `42`
- reuse references: `true`

What you should explain:

- The translated candidates are always compared against the same saved real reference subset within a sweep.
- This is essential for fair comparison across checkpoints and inference settings.

Critical caveat you should address explicitly:

- By default, `eval.real_dir = null`, which means the sweep falls back to the CycleNet training target directory.
- Unless you override it, your real evaluation set may therefore come from the same overall pool used during training.

Recommendation for the thesis:

- Best practice:
  - use a held-out real reference split for evaluation
- If that is not possible:
  - state that the realism metrics quantify closeness to the target-domain distribution represented in the training corpus, not generalization to unseen real scenes

For downstream segmentation evaluation, what still needs to be decided:

- The repo does not currently implement segmentation-model training or segmentation metrics.
- You need to define:
  - segmentation training split
  - validation split
  - test split
  - whether translated images are generated only from the segmentation training split

My recommendation:

- Keep diffusion training data and segmentation evaluation data conceptually separate.
- For segmentation experiments:
  - translate only the synthetic images used as segmentation training data
  - evaluate segmentation on real validation/test sets that are never used for diffusion metric fitting
- If real labeled data is scarce, use it only for validation/test or for a clearly defined low-data fine-tuning condition.

### Evaluation Metrics

You should separate this section into:

- metrics already implemented and used in the sweep
- optional metrics that are plausible but not yet part of the main evaluation pipeline

#### FID / FID-CLIP

What is implemented:

- `real_fid`
- CLIP-based distribution metrics:
  - `real_clip_centroid_cosine`
  - `real_clip_centroid_l2`
  - `real_clip_nearest_cosine_mean`
  - `real_clip_frechet`
  - `real_clip_mmd_rbf`

FID details to explain:

- FID is computed using TorchMetrics `FrechetInceptionDistance(feature=2048, normalize=True)`.
- Images are resized to `299 x 299` before computing Inception features.
- Lower FID means the translated image distribution is closer to the real reference distribution in Inception feature space.

Important limitations you should acknowledge:

- Your sample sizes are much smaller than canonical FID practice.
- Inception features are trained on natural-image distributions, not remote sensing.
- FID measures realism / distribution alignment, not content preservation.

Recommended thesis stance:

- Use FID as a heuristic realism metric, not as the only decision criterion.

CLIP metric details to explain:

- CLIP embeddings are extracted from a local CLIP model path configured in the sweep.
- The current config uses:
  - `clip_feature_layer = pooler`
- Embeddings are L2-normalized before metric computation.

How to explain the CLIP metrics:

- `real_clip_centroid_cosine`:
  - cosine similarity between translated and real embedding centroids
  - higher is better
- `real_clip_centroid_l2`:
  - Euclidean distance between translated and real centroids
  - lower is better
- `real_clip_nearest_cosine_mean`:
  - average nearest-neighbor cosine similarity from translated images to real images
  - higher is better
  - mention that this can reward partial mode collapse
- `real_clip_frechet`:
  - FID-style Fréchet distance in CLIP embedding space
  - lower is better
- `real_clip_mmd_rbf`:
  - RBF-kernel MMD between translated and real CLIP distributions
  - lower is better

Recommended thesis prioritization:

- For remote-sensing translation, `real_clip_frechet` and `real_clip_mmd_rbf` are probably your strongest distribution-level metrics.
- Keep FID as a secondary but still useful realism indicator.

#### UMAP Embeddings

What is currently implemented:

- The sweep can write a CLIP-based UMAP for every candidate.
- Each UMAP jointly embeds:
  - source simulated references
  - translated simulated outputs
  - real references

What to explain:

- UMAP is used as a qualitative diagnostic for domain movement in embedding space.
- A desirable translation moves the translated set away from the simulated cluster and toward the real cluster without collapsing diversity.

Important limitation to state:

- The UMAP is fit separately for each candidate directory.
- Therefore, axes are not directly comparable across different candidate plots.

How to write this well:

- Present UMAP as a within-candidate diagnostic, not a cross-candidate metric.

About PCA and t-SNE:

- The main sweep uses UMAP.
- A separate projection script can generate PCA and t-SNE style diagnostics.
- Those projections are currently auxiliary, not part of the core evaluation loop.

My recommendation:

- Use UMAP as the primary thesis figure because it is already integrated.
- If you want stronger support, include PCA or t-SNE in supplementary material.
- If you need strict cross-candidate comparability, modify the analysis to fit one shared reducer over all compared samples instead of separate UMAPs.

#### LPIPS / CLIP Similarity

What is currently implemented:

- LPIPS is computed between each source simulated image and its translated counterpart.
- The code reports:
  - `source_lpips_mean`
  - `source_lpips_std`

How to explain LPIPS in this context:

- LPIPS is not a realism metric here.
- It is a preservation / amount-of-change metric.
- Lower LPIPS means the translation remains perceptually closer to the source.

The key thesis interpretation:

- Very low LPIPS can mean under-translation.
- Very high LPIPS can mean destructive translation that may damage segmentation-relevant structure.

Recommendation:

- Use LPIPS as a guardrail rather than a single optimization target.
- In your plots, pair LPIPS with a realism metric such as `real_clip_frechet` or FID.

About CLIP similarity:

- You already have several CLIP-based realism metrics.
- I would not add another standalone CLIP similarity score unless it answers a new question.

Best recommendation:

- Use:
  - `real_clip_frechet`
  - `real_clip_mmd_rbf`
  - LPIPS
- That gives you one or two realism metrics plus one preservation metric without metric overload.

#### Power Spectrum / Frequency Analysis

What exists in the repo:

- `src/cyclenet/diffusion/spectral.py` contains utilities for:
  - luminance conversion
  - random patch extraction
  - radial power spectrum estimation
  - log-spectrum comparison
- `scripts/residual_signal.py` helps interpret how much source signal remains at different `noise_strength` values.

What is not yet implemented:

- This is not currently part of the main evaluation sweep.
- There is no checked-in script that automatically computes and reports spectral statistics across candidate directories in the same way as FID/CLIP/LPIPS.

How I recommend you handle this in the thesis:

- Only include power-spectrum analysis if you actually run it and it answers a visible artifact question.
- Good use cases:
  - showing that translated images recover higher-frequency texture closer to real imagery
  - showing that some settings inject unrealistic high-frequency noise

If you include it, explain it as:

- a supplemental low-level texture analysis
- not a primary selection metric

Recommended methodology if you decide to run it:

- Convert RGB to luminance.
- Sample fixed-size patches, for example `64 x 64`.
- Compute radial power spectra with Hann windowing.
- Compare average translated spectra against source and real spectra.
- Report qualitative trends:
  - closer high-frequency behavior to real
  - or excessive high-frequency amplification for aggressive settings

### Distribution-Level Evaluation

This section should present the translation model as a distribution-matching system before you discuss downstream segmentation.

#### Quantitative Realism Metrics

What to report:

- Source baseline versus real:
  - untranslated simulated references against real references
- Translated candidates versus real:
  - per checkpoint
  - per CFG value
  - per noise strength

Best presentation format:

- heatmaps over:
  - CFG
  - noise strength
- one heatmap per checkpoint, or one heatmap for the best checkpoint with others summarized in a table

Recommended primary metrics:

- `real_clip_frechet`
- `real_clip_mmd_rbf`
- `real_fid`

Recommended preservation metric reported beside them:

- `source_lpips_mean`

Important recommendation for thesis presentation:

- Present raw metrics in the main paper/chapter.
- You can mention that you used a composite ranking script internally to shortlist candidates.
- But the thesis should not rely only on a custom normalized `selection_score`, because raw metrics are easier to interpret and defend.

#### Embedding Projections

What to show:

- CLIP UMAP plots for a small set of representative candidates:
  - one weak translation
  - one best tradeoff
  - one overly aggressive translation

What you should explain:

- The translated cluster should move away from the source simulated cluster and toward the real cluster.
- Healthy translation should not collapse into a very small translated cluster.

Good narrative to use:

- This figure shows whether the model reduces the domain gap in feature space while maintaining diversity.

Important caveat:

- Because UMAP is fit per candidate, compare relative geometry within each plot rather than absolute coordinates across plots.

#### Preservation vs Translation

This is a strong section for your thesis even though it is not automatically plotted by the repo.

What I recommend plotting:

- x-axis:
  - `source_lpips_mean`
- y-axis:
  - `real_clip_frechet` or `real_fid`
- one point per candidate

How to explain it:

- The plot exposes the tradeoff between making the images look more real and preserving source content.
- Good candidates lie near the Pareto frontier:
  - lower realism distance
  - without excessively large LPIPS

Why this is especially important for your project:

- Your end goal is not only visual realism.
- It is realism that still preserves label-relevant spatial structure for segmentation.

#### Low-Level Feature Analysis

What you can say if you run spectral analysis:

- This analysis probes whether translation improves low-level texture statistics beyond what embedding-based metrics capture.
- In remote sensing, realism is often visible in texture granularity, noise characteristics, and local frequency content.

If you do not run it:

- Keep this subsection short and frame it as optional or future work.
- Do not promise a full quantitative spectral result unless you actually compute it.

My recommendation:

- Make this a supplemental subsection unless your visual results clearly show frequency artifacts that need explanation.

### Downstream-Task Evaluation

This is the section where the thesis becomes strongest, but it is also the part least implemented in the current repo.

The important point:

- The repo currently implements translation training and translation evaluation.
- It does not currently implement the downstream segmentation training/evaluation pipeline.

That means this subsection should clearly distinguish:

- what the translation system already provides
- what experimental protocol you add for downstream validation

#### Segmentation Setup

What you need to define outside this repo:

- the segmentation architecture
- the optimizer and training schedule
- the train/val/test splits
- the training data regimes being compared

The most useful training regimes to compare are:

- simulated only
- translated simulated only
- simulated + real labeled
- translated simulated + real labeled
- real labeled only

If real labels are scarce, also consider:

- small real labeled set + simulated
- small real labeled set + translated simulated

Critical fairness recommendations:

- Use the same segmentation architecture and training hyperparameters across all regimes.
- Keep the real validation/test set fixed.
- If you translate simulated training images, only translate the training split, not validation/test.
- Do not compare regimes that differ in both data volume and data type unless you explicitly normalize for sample count.

Best thesis explanation:

- The segmentation study tests whether reduced visual domain gap translates into improved target-domain semantic segmentation, not just prettier images.

#### Quantitative Results

Metrics I recommend:

- mean IoU
- pixel accuracy
- per-class IoU

If you can afford multiple runs:

- report mean and standard deviation across several random seeds

Why per-class IoU matters:

- Some classes may benefit more from realism transfer than others.
- Classes with fine structure or texture-sensitive appearance may respond differently from large homogeneous classes.

Best table structure:

- rows:
  - training regimes
- columns:
  - mIoU
  - pixel accuracy
  - important per-class IoUs

If space is limited:

- put all per-class IoUs in an appendix
- keep mIoU and pixel accuracy in the main chapter

#### Segmentation Outputs

What I recommend showing:

- real input image
- ground-truth mask
- predicted mask from sim-only model
- predicted mask from translated-sim model
- optionally predicted mask from real-labeled baseline

How to choose examples:

- include one success case where translation clearly improves boundary or class recognition
- include one neutral case
- include one failure case where translation harms or does not help

Good narrative:

- These qualitative examples illustrate whether the translated data improves semantic alignment in the target domain, not just image realism.

#### Failure Cases

This subsection is important and you should plan for it now.

What failure modes are plausible given the implementation:

- high CFG values can produce visually stronger but structurally less faithful translations
- high noise strengths can erase source geometry before denoising recovers target style
- some classes may not benefit if their class cues are already well represented in synthetic imagery
- translation may improve global realism while slightly shifting boundaries or local texture cues in a way that harms segmentation

Implementation-linked explanation of why these failures can happen:

- The training loss strongly anchors content preservation through reconstruction and consistency terms.
- The inference procedure can still override that bias when CFG and noise strength are pushed too far.
- Therefore, the most visually realistic samples are not guaranteed to be the most segmentation-useful samples.

Recommended failure-case framing:

- Report whether the best realism candidate is different from the best downstream segmentation candidate.
- If so, that is not a problem; it is an important result.
- It supports the argument that downstream utility requires a realism-preservation tradeoff rather than realism alone.

## Recommended Thesis Decisions

These are the most important unresolved decisions I recommend you make before writing the final methods/experiments chapters.

### 1. Make the segmentation-conditioned model the main system

Reason:

- It is the current active implementation.
- It aligns with your downstream segmentation motivation.
- It gives you a more distinctive thesis story than a plain RGB translation model.

How to write it:

- Present the RGB-only CycleNet as a baseline or earlier variant.
- Present the RGB+seg model as the final method.

### 2. Use raw metrics for reporting and the composite score only for internal selection

Reason:

- Reviewers can directly interpret FID, CLIP-Frechet, MMD, and LPIPS.
- A custom score is useful for model selection but weaker as a primary scientific result.

How to write it:

- "We used a composite ranking internally to shortlist candidates, then reported raw realism and preservation metrics for the selected settings."

### 3. Hold out a real evaluation subset if possible

Reason:

- The current defaults otherwise compare against the same real-domain corpus used in diffusion training.
- That is acceptable for model selection but weaker for thesis claims about generalization.

How to write it:

- If you use a holdout:
  - say distribution metrics were computed against unseen real reference images
- If you do not:
  - say the evaluation measures alignment to the target-domain training distribution

### 4. Add downstream segmentation as the final model-selection criterion

Reason:

- Your thesis value is not only image translation quality.
- It is whether translation improves downstream performance on real imagery.

How to write it:

- "Distribution-level metrics were used to narrow the candidate set, while downstream segmentation performance served as the final criterion for selecting practical translation settings."

### 5. Prefer moderate guidance and moderate noise as your likely operating region

Reason:

- The current implementation is conservative during training and can become unstable when inference steering is too strong.
- Moderate settings are most consistent with preserving semantic geometry.

How to write it once you have results:

- "The best tradeoff was obtained at intermediate CFG and noise-strength values, which provided measurable realism gains without excessive structural drift."

## Suggested Chapter Structure

If you want a clean final narrative, I would structure it like this:

1. Methods
2. Training setup
3. Segmentation-conditioned CycleNet objective
4. Inference procedure with CFG and DDIM
5. Experiments
6. Distribution-level evaluation
7. Downstream segmentation evaluation
8. Failure analysis and ablations

That ordering fits the implementation and gives you a logical progression from:

- how the model is trained
- how translation is controlled at inference
- how realism is measured
- how practical downstream value is validated
