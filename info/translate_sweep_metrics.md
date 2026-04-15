# Translate Sweep Metrics

This note describes the metrics produced by `src/cyclenet/eval/translate_sweep.py`.
The sweep translates a fixed sample of simulated images, compares each translated
candidate against a fixed sample of real images, and also computes a baseline
comparison between the untranslated simulated images and the real images.

## Comparisons

Each row in `metrics.csv` and `metrics.json` is one comparison.

`kind=source_baseline` / `comparison=source_sim_vs_real` compares the saved
source simulated reference images against the saved real reference images. This
is the baseline domain gap before CycleNet translation.

`kind=translated_candidate` / `comparison=translated_sim_vs_real` compares one
translated simulated candidate set against the same saved real reference images.
The candidate is identified by `checkpoint`, `cfg_weight`, and `noise_strength`.

The real reference set is fixed across all candidates in a run. When
`eval.reuse_references=true` and the manifests already exist, the same saved
source and real reference images are reused across reruns as well.

## FID

CSV column: `real_fid`

Enabled by: `fid` in `eval.metrics`

FID measures how close the candidate image distribution is to the real image
distribution in InceptionV3 feature space. The implementation resizes images to
`299 x 299`, converts them to tensors, feeds them through TorchMetrics'
`FrechetInceptionDistance(feature=2048, normalize=True)`, and computes the
Fréchet distance between the real and fake feature Gaussians:

```text
||mu_fake - mu_real||^2
+ Tr(Sigma_fake + Sigma_real - 2 (Sigma_real^1/2 Sigma_fake Sigma_real^1/2)^1/2)
```

Interpretation: lower is better. A lower `real_fid` means the translated sim
images look more like the real reference image distribution according to
Inception features.

Important caveats:

- FID is noisy with small samples. Values from 512 translated images and 1024
  real images are useful for ranking nearby sweep settings, but not as a final
  publication-quality FID.
- FID is not paired. It does not care whether each translated image preserves
  the content of its source image.
- Inception features are trained on natural images, not remote sensing imagery,
  so FID should be treated as a heuristic realism/domain-gap metric.

## CLIP Embeddings

CSV columns:

- `real_clip_centroid_cosine`
- `real_clip_centroid_l2`
- `real_clip_nearest_cosine_mean`
- `real_clip_frechet`
- `real_clip_mmd_rbf`
- `clip_umap_csv`
- `clip_umap_svg`
- `clip_umap_error`, when UMAP cannot be written

Enabled by: `clip` in `eval.metrics`

The script loads CLIP from `eval.clip_model_path`. With
`eval.clip_feature_layer: pooler`, it uses `model.vision_model(...).pooler_output`
directly, which gives the 768-dimensional vision embedding from your local CLIP
setup. Each embedding vector is L2-normalized before metric computation.

### `real_clip_centroid_cosine`

This computes the cosine similarity between the mean normalized CLIP embedding
of the candidate set and the mean normalized CLIP embedding of the real set.

Interpretation: higher is better. Values closer to `1.0` mean the global CLIP
centroid of the translated images is closer to the real-image centroid.

This is a coarse summary. It can look good even when the candidate distribution
has the wrong spread or covers only part of the real distribution.

### `real_clip_centroid_l2`

This computes the Euclidean distance between the normalized candidate centroid
and normalized real centroid.

Interpretation: lower is better. It is mostly the inverse view of
`real_clip_centroid_cosine`: small values mean the global CLIP centroids are
close.

### `real_clip_nearest_cosine_mean`

For each candidate image embedding, the script computes cosine similarity to
every real image embedding, takes the nearest real neighbor, then averages those
nearest-neighbor similarities across candidate images.

Interpretation: higher is better. It asks whether each translated image lands
near at least one real image in CLIP space.

Caveat: this metric can reward mode collapse. A candidate set can score well if
many translated images cluster near a few real images, even if it does not cover
the full real distribution.

### `real_clip_frechet`

This is an FID-style Fréchet distance computed in normalized CLIP embedding
space instead of Inception feature space. The script computes the mean and
covariance of the candidate CLIP embeddings and the real CLIP embeddings, then
uses the same Gaussian Fréchet distance form as FID.

Interpretation: lower is better. It measures both centroid shift and covariance
shape mismatch in CLIP space.

This is usually one of the more useful CLIP metrics for sweep ranking because it
penalizes candidates that match the average real embedding but have the wrong
spread.

### `real_clip_mmd_rbf`

This computes a biased maximum mean discrepancy in normalized CLIP space using
an RBF kernel. The RBF bandwidth is chosen by the median pairwise squared
distance of the pooled candidate and real embeddings, capped to the first 1024
pooled embeddings for that bandwidth estimate.

The value is:

```text
mean k(fake, fake) + mean k(real, real) - 2 mean k(fake, real)
```

Interpretation: lower is better. It is a nonparametric two-sample discrepancy:
smaller values mean the translated and real CLIP embedding distributions are
harder to distinguish with the RBF kernel.

Compared with `real_clip_frechet`, MMD makes fewer Gaussian assumptions, but it
depends on the kernel bandwidth and can be sensitive to sample size.

## CLIP UMAP

Output files per candidate:

- `clip_umap.csv`
- `clip_umap.svg`

Enabled by: `eval.plot_clip_umap: true`

For each candidate, the script concatenates CLIP embeddings from:

- `source_sim`: the fixed source simulated reference set
- `translated_sim`: the translated candidate set
- `real`: the fixed real reference set

It then fits a 2D UMAP with `eval.umap_random_state` and writes both point
coordinates and an SVG scatterplot.

Interpretation: this is a diagnostic visualization, not a scalar metric. A good
translation should move `translated_sim` away from `source_sim` and toward
`real`, while still maintaining enough diversity to overlap the real cluster
rather than collapsing to a small region.

Caveats:

- UMAP changes distances and densities. Use it to spot obvious cluster behavior,
  not to make fine-grained quantitative claims.
- The UMAP is fit separately per candidate, so visual axes are not directly
  comparable across candidate SVGs. Compare the relative positions of
  `source_sim`, `translated_sim`, and `real` within each plot.

## LPIPS Preservation

CSV columns:

- `source_lpips_mean`
- `source_lpips_std`

Enabled by: `lpips` in `eval.metrics`

LPIPS is computed between each source simulated image and its translated output.
The script uses the AlexNet LPIPS model and reports the mean and standard
deviation across translated samples.

Interpretation: lower means the translated image is perceptually closer to the
source image. In this sweep, LPIPS is a content-preservation/change metric, not a
realism metric.

For CycleNet selection, LPIPS is useful as a guardrail:

- Very low LPIPS can mean the model barely changed the sim image.
- Very high LPIPS can mean the translation is too destructive and may alter
  segmentation-relevant structure.
- A good setting should improve realism metrics relative to `source_baseline`
  without pushing LPIPS so high that land-cover geometry or class cues are
  visibly distorted.

## Error Columns

Metric failures are recorded rather than crashing the whole sweep when possible.
Common error columns are:

- `fid_error`
- `clip_error`
- `lpips_error`

If a metric fails, the corresponding scalar is usually set to `nan` or omitted,
and the error column stores the exception type and message.

## Practical Ranking Guidance

Use the source baseline as the reference point. A useful CycleNet candidate
should improve `real_fid`, `real_clip_frechet`, and `real_clip_mmd_rbf` compared
with `source_sim_vs_real`.

For selecting steps and sampling parameters, prioritize:

1. Lower `real_clip_frechet` and `real_clip_mmd_rbf`.
2. Lower `real_fid`, when the FID sample size is large enough and Inception
   behaves consistently for the imagery.
3. Reasonable `source_lpips_mean`, checked against visual samples so the
   translation does not erase segmentation-relevant structure.
4. UMAP plots where `translated_sim` moves toward `real` without collapsing into
   a tiny cluster.

Avoid picking solely by the lowest realism score. For simulated-to-real data
augmentation, the best candidate is usually the one that reduces the real-domain
gap while preserving label geometry and land-cover cues.
