# CycleNet Conditioning Modes and Objective Interpretation

This note summarizes how to interpret the official CycleNet conditioning inputs and training objectives, and how that maps onto the `from_idx` / `to_idx` version in this repo.

## Core Interpretation

The cleanest conceptual split is:

- `from_idx` / ControlNet domain: what domain the conditioning image should be **read as**
- `to_idx` / UNet backbone domain: what domain the denoised result should be **written as**
- `c_img`: which **instance / layout / structure anchor** the model should stay aligned to

So the two conditioning channels answer different questions:

- ControlNet: "what kind of image is this hint?"
- UNet backbone: "what kind of image should come out?"

`c_img` is best understood as an **instance anchor**, not just "the image to reconstruct." It carries the scene/object layout, geometry, pose, identity, and other non-domain-specific details that should be preserved.

## Four Conditioning Modes

| Mode | `from_idx` meaning | `to_idx` meaning | `c_img` means | Conceptual operation | Used for |
|---|---|---|---|---|---|
| `x -> x` | Read hint as an `X` image | Write output in `X` | Original source exemplar `x_0` | Source fixed-point / identity denoising | Reconstruction, `consis` reverse branch |
| `x -> y` | Read hint as an `X` image | Write output in `Y` | Original source exemplar `x_0` | Forward translation: preserve source structure while editing domain semantics | Forward branch, invariance left side |
| `y -> y` | Read hint as a `Y` image | Write output in `Y` | Translated exemplar `\bar y_0` | Target fixed-point / "already in target domain" denoising | Invariance right side |
| `y -> x` | Read hint as a `Y` image | Write output in `X` | Translated exemplar `\bar y_0` | True backward translation from target-domain state back to source | `cycle` reverse branch |

Short way to remember it:

- ControlNet decides how to interpret the conditioning image
- UNet decides what domain the denoised sample should land in
- `c_img` decides which specific image instance the model should stay faithful to

## What the Conditioning Image Means

The same image can play different roles depending on `from_idx`.

- `x_0` with `from_idx = X` means:
  "This is a source-domain exemplar. Preserve its structure, but identify the source-domain semantics that may need editing."
- `\bar y_0` with `from_idx = Y` means:
  "This is already a target-domain exemplar. Preserve it as a target-domain image."

So changing both the conditioning image and the ControlNet domain changes the model's interpretation of the hint:

- `x_0 @ X` means "edit this source exemplar"
- `\bar y_0 @ Y` means "preserve this target exemplar"

That is why the invariance branch swaps both together.

## Objective Table

| Objective | Inputs being compared | What it asks | High-level role |
|---|---|---|---|
| Reconstruction | `\epsilon(x_t, x -> x, x_0)` vs `\epsilon_x` | Can the model denoise a source image back to itself when everything says "stay in source"? | Establish source-domain fixed point |
| Invariance | `\epsilon(x_t, x -> y, x_0)` vs `\epsilon(x_t, y -> y, \bar y_0)` | Does the translated result behave like a genuine target exemplar that is stable under target conditioning? | Make forward translations land on the target manifold |
| Consistency | `\epsilon(x_t, x -> y, x_0) + \epsilon(y_t, x -> x, x_0)` vs `\epsilon_x + \epsilon_y` | After editing toward `Y`, can the sample still be explained / recovered from the original source anchor? | Source-preservation / easy reversibility |
| Cycle | `\epsilon(x_t, x -> y, x_0) + \epsilon(y_t, y -> x, \bar y_0)` vs `\epsilon_x + \epsilon_y` | After editing into `Y`, can I genuinely use that target-domain state to translate back to `X`? | True transitivity through target space |

## Key Algebra Behind `consis` and `cycle`

Let

- `x_t = sqrt(a) x_0 + sqrt(1-a) * epsilon_x`
- `epsilon_xy = \epsilon(x_t, x -> y, x_0)`
- `\bar y_0 = (x_t - sqrt(1-a) * epsilon_xy) / sqrt(a)`
- `y_t = sqrt(a) * \bar y_0 + sqrt(1-a) * epsilon_y`

Substituting `\bar y_0` into `y_t` gives

- `y_t = sqrt(a) * x_0 + sqrt(1-a) * (epsilon_x + epsilon_y - epsilon_xy)`

So if a reverse branch is supposed to recover `x_0` from `y_t`, the reverse noise it should predict is

- `epsilon_rev = epsilon_x + epsilon_y - epsilon_xy`

which is equivalent to

- `epsilon_xy + epsilon_rev = epsilon_x + epsilon_y`

This is why both `consis` and `cycle` compare a sum of predicted noises against `epsilon_x + epsilon_y`.

Important point:

- In `y -> y`, the branch would predict `epsilon_y`
- In `x`-recovering branches, the branch predicts the total noise needed to recover `x_0` from `y_t`
- That total noise is `epsilon_x + epsilon_y - epsilon_xy`, not just `epsilon_y`

## What `consis` Means

`consis` is:

- noisy input: `y_t`
- conditioning mode: `x -> x`
- image hint: `x_0`

So it is **not** a true `y -> x` translation branch. It is asking:

- "After I translate toward `Y`, is the sample still source-recoverable if I keep the model fully anchored to the original source image?"

That is why it is best described as a **source-anchored proxy**:

- it keeps the original `x_0` as the anchor
- it keeps both domain conditionings in `X`
- it tests whether the forward edit preserved enough source information that reversal stays easy

It mainly enforces:

- structure preservation
- low drift from the source
- easy reversibility when the original source anchor is still available

Short summary:

- `consis` = "do not lose the source"

## What `cycle` Means

`cycle` is:

- noisy input: `y_t`
- conditioning mode: `y -> x`
- image hint: `\bar y_0`

So it asks:

- "Can I treat `\bar y_0` as a real target-domain intermediate, and use that translated state to come back to `X`?"

This is stricter than `consis` because it requires:

- `\bar y_0` to be meaningful as a `Y` image
- the reverse branch to genuinely depend on that target-domain exemplar
- the full path `x -> y -> x` to be coherent

It mainly enforces:

- true transitivity through target space
- reversibility through the translated intermediate
- actual use of the translated sample rather than only the original source anchor

Short summary:

- `cycle` = "actually pass through the target"

## What Invariance Means

The official implementation compares:

1. forward translation:
   `\epsilon(x_t, x -> y, x_0)`

2. target fixed-point:
   `\epsilon(x_t, y -> y, \bar y_0)`

The UNet output domain is `Y` in both cases. What changes is how the side network interprets the conditioning image:

- branch 1: "edit this source exemplar into `Y`"
- branch 2: "treat this translated image as already being a `Y` exemplar"

So invariance asks:

- "Did my forward translation produce something that the model itself recognizes as a stable target-domain exemplar?"

This is why the invariance branch swaps both:

- conditioning image: `x_0 -> \bar y_0`
- ControlNet domain: `X -> Y`

The swap changes the semantic status of the hint from:

- "source image that needs editing"

to:

- "target image that should already be stable"

## Practical Takeaways

- Reconstruction establishes a source-domain identity fixed point
- Invariance establishes a target-domain identity fixed point
- Consistency makes sure forward translation preserves enough source information to be easily reversible
- Cycle makes sure the model truly goes through target space and can return from that translated state

One-line summaries:

- `x -> x, x_0`: source identity anchor
- `x -> y, x_0`: edit source into target while preserving instance structure
- `y -> y, \bar y_0`: target self-consistency / target fixed point
- `y -> x, \bar y_0`: true reverse translation from translated target state

One-line objective summaries:

- Reconstruction: source fixed point
- Invariance: target fixed point
- Consistency: preserve enough source information to reverse easily
- Cycle: make the translated image a real usable intermediate for inversion
