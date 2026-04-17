Yes. With those three runs, I would use your eval code in a fairly specific order.

**What To Run**

First, run [translate_sweep.py](/home/logan/projects/pixel-cyclenet/src/cyclenet/eval/translate_sweep.py:1) for each model/run you want to compare:

- `all_real` resumed to `30k`
- `all_real_ft_invar` to `30k`
- `oem_only_ft_invar` to `30k`

Use the same:
- `data.src_dir`
- `data.rgb_parent_dirs`
- `eval.real_dir`
- `eval.source_sample_size`
- `eval.real_sample_size`
- `eval.seed`
- `eval.translation_seed`

across all three. That way the comparisons are actually apples-to-apples.

I would also keep the sweep narrower than your earlier exploratory one. Since high CFG is already showing obvious failure modes, I would focus the new sweep on the region that is most likely useful for segmentation data:

- `cfg_weights: [1.0, 2.0, 3.0, 4.0]`
- `noise_strengths: [0.1, 0.2, 0.3]`

Maybe include `0.4` only if you still suspect it helps. I would not spend much time on `cfg 5.0` now except as a stress test.

Second, for each resulting `metrics.csv`, run [analyze_translate_sweep.py](/home/logan/projects/pixel-cyclenet/src/cyclenet/eval/analyze_translate_sweep.py:1). That script is your main per-model shortlist generator.

Third, once you have per-model `analysis/.../ranked_candidates.csv` outputs, run [analyze_translate_sweep_consensus.py](/home/logan/projects/pixel-cyclenet/src/cyclenet/eval/analyze_translate_sweep_consensus.py:1) across the three model folders. That will tell you which settings are robust across models rather than just winning in one run.

You do not need to run [project_translate_sweep_projections.py](/home/logan/projects/pixel-cyclenet/src/cyclenet/eval/project_translate_sweep_projections.py:1) for the new sweeps, because `translate_sweep.py` now writes PCA / t-SNE / UMAP directly. That script is mainly useful if you want to backfill projections for old sweep directories without rerunning the whole sweep.

**What To Look For**

For your use case, I would not optimize for the single best realism metric. I would look for the best tradeoff among four things:

- lower `real_fid`, `real_clip_frechet`, and `real_clip_mmd_rbf`
- moderate `source_lpips_mean`, not maximal change
- projection plots where translated points move toward the real cluster without spraying into weird sub-clusters
- actual image quality with no local nonsense like green boxes, blocky canopies, or geometry drift

That third point matters more than it might seem. In the PCA / t-SNE / UMAP plots, bad candidates often show up as:
- translated samples scattering far outside both source and real clouds
- multiple strange translated islands
- translated points overshooting past the real cluster instead of moving into it

Good candidates usually look more like:
- translated points shift away from source
- translated points overlap or approach real
- the translated distribution stays coherent

For segmentation training specifically, I would interpret the metrics with this priority:

1. `source_lpips_mean` should stay moderate, not huge.
2. realism metrics should improve over the source baseline.
3. projections should show a clean movement toward real.
4. visual inspection should confirm that object boundaries and canopy shapes still make sense.

So if you have a candidate with slightly better FID but noticeably worse LPIPS and worse artifacts, I would reject it for segmentation data.

**How I’d Use Your Analysis Scripts**

For [analyze_translate_sweep.py](/home/logan/projects/pixel-cyclenet/src/cyclenet/eval/analyze_translate_sweep.py:572), I would change the scoring to be more preservation-aware than the current example. For segmentation training, I would probably not use a realism-heavy weighting like `realism_weight = 0.75`.

I’d start closer to:

- `realism_weight = 0.5`
- `preservation_weight = 0.5`

or even:

- `realism_weight = 0.4`
- `preservation_weight = 0.6`

I would also use an LPIPS target rather than blindly minimizing LPIPS to zero. You probably do want some change, just not excessive change. Based on what you’ve been describing, a first target range like:

- `lpips_target = 0.15` to `0.25`
- `max_lpips = 0.25` to `0.30`

seems reasonable.

That lets the analysis favor candidates that are:
- changed enough to move toward real
- not changed so much that label fidelity is likely compromised

Then, after per-model ranking, run [analyze_translate_sweep_consensus.py](/home/logan/projects/pixel-cyclenet/src/cyclenet/eval/analyze_translate_sweep_consensus.py:1) on the root that contains all three sweep outputs. What you want there is not “the single global winner,” but:
- settings that rank near the top across multiple models
- settings that do not collapse on one run and fail on another

If the consensus script says the robust region is something like:
- `cfg = 2-3`
- `strength = 0.2`

that is much more trustworthy than a one-off `cfg = 4, strength = 0.4` winner from one model.

**My Current Expectation**

Given what you’ve seen so far, my prior is:

- the resumed `all_real` to `30k` may reduce the ugliest undertrained artifacts but could drift more reconstructive
- `all_real_ft_invar` to `30k` has the best chance of landing in the useful middle
- `oem_only_ft_invar` may look cleaner and more coherent than `all_real`, but may not transfer the same way if your eventual segmentation target is broader than OEM-only style

So when you compare them, I would pay special attention to whether `all_real_ft_invar`:
- improves realism metrics over plain `all_real`
- keeps LPIPS moderate
- produces cleaner projection structure
- removes the high-CFG artifact tendency at the moderate CFGs you’d actually use

If it does, that is probably your best candidate family for generating translated sim for segmentation.

**Decision Rule I’d Actually Use**

After you run the sweeps and analyses, I would pick 2-3 candidate settings, not one, and then do a small downstream segmentation pilot.

Specifically:
- pick the best-looking candidate from `all_real`
- pick the best from `all_real_ft_invar`
- pick the best from `oem_only_ft_invar`

Then translate a small sim subset with each, train short segmentation runs, and compare validation performance. That downstream test should be the final tie-breaker.

So the analysis pipeline is:

1. `translate_sweep.py` on all three runs with the same source/reference setup
2. `analyze_translate_sweep.py` per run with preservation-aware scoring
3. `analyze_translate_sweep_consensus.py` across the three runs
4. visual inspection of top candidate directories and their PCA/t-SNE/UMAP plots
5. short segmentation pilot on the top 2-3 candidates

If you want, I can help you define the exact sweep grid and the exact `analyze_translate_sweep.py` settings I’d use for this segmentation-oriented comparison.