# SPADE For Seg-Only ControlNet In CycleNet

This note explains:

- how SPADE works at the block level
- how it differs from the current `ResBlock` + AdaGN design
- how to wire it into this repository's ControlNet path
- what files need edits
- where this design matches or diverges from "traditional" SPADE
- what behavior changes to expect for translation

The target design here is:

- `seg-only` control conditioning
- SPADE used inside the `ControlNet` branch
- the pretrained/frozen main diffusion `UNet` backbone left unchanged initially
- domain conditioning retained via the current AdaGN-style domain FiLM


## 1. Current Architecture Summary

Today, the main relevant path is:

- `src/cyclenet/models/unet.py`
- `src/cyclenet/models/controlnet.py`
- `src/cyclenet/models/cyclenet.py`
- `src/cyclenet/models/blocks/resblock.py`

The current logic is:

1. The backbone `UNet` sees `x_t`, a noisy source RGB image.
2. `CycleNet.forward()` sends `x_t`, `t`, `from_idx`, `to_idx`, and `c_img` into the `ControlNet`.
3. `ControlNet` produces zero-conv control skips.
4. Those control skips are added into the backbone bottleneck and decoder skips.

The current residual block uses:

- `GroupNorm`
- optional domain FiLM using the domain embedding `d_emb`

Current modulation equation in `ResBlock`:

```python
h = norm(x)
gamma_d, beta_d = d_proj(d_emb)
h = h * (1.0 + gamma_d) + beta_d
```

This is "AdaGN-style" domain conditioning: normalize first, then apply a learned domain-specific affine modulation.


## 2. What SPADE Does

SPADE stands for Spatially-Adaptive Denormalization.

The main idea is:

- normalize a feature map
- use a segmentation map to predict spatial `gamma` and `beta`
- modulate the normalized features differently at each spatial location

Canonical SPADE equation:

```python
h_norm = GN(x)
gamma_s, beta_s = seg_encoder(seg)
h = h_norm * (1.0 + gamma_s) + beta_s
```

Shapes:

- `x`: `(B, C, H, W)`
- `h_norm`: `(B, C, H, W)`
- `seg`: `(B, Cseg, Hseg, Wseg)` or resized to `(H, W)`
- `gamma_s`, `beta_s`: `(B, C, H, W)`

The key difference from ordinary FiLM/AdaGN is:

- AdaGN uses a global `(B, C, 1, 1)` modulation from an embedding
- SPADE uses a spatial `(B, C, H, W)` modulation from the segmentation map


## 3. Why SPADE Fits This Use Case

The current problem is:

- late-timestep translation is too source-faithful
- earlier, more realistic translation can damage structure

The desired behavior is:

- preserve segmentation structure
- preserve class identity
- allow freer target-domain texture and artifacts

`seg-only + SPADE` is a direct architectural answer because it says:

- the control branch should preserve semantic layout, not source RGB appearance
- the segmentation map should influence features repeatedly throughout the control branch, not only once at the input

This is a more direct statement of the intended inductive bias than:

- scaling RGB by hand
- dropout heuristics
- tuning control weights


## 4. How SPADE And AdaGN Should Be Combined Here

SPADE and AdaGN should not be implemented as two separate normalization stages.

Do not do this:

```python
h = GN(x)
h = AdaGN(h, d_emb)
h = GN(h)
h = SPADE(h, seg)
```

Instead, use:

- one shared base normalization per sub-block
- one global domain modulation
- one spatial segmentation modulation

Combined equation:

```python
h_norm = GN(x)

gamma_d, beta_d = domain_proj(d_emb)   # (B, C, 1, 1)
gamma_s, beta_s = seg_proj(seg)        # (B, C, H, W)

h = h_norm * (1.0 + gamma_d + gamma_s) + (beta_d + beta_s)
```

This is the cleanest formulation for this repository.

Recommended normalization:

```python
nn.GroupNorm(32, channels, affine=False)
```

Reason:

- GroupNorm affine parameters become redundant once dynamic domain and segmentation affine terms are applied
- `affine=False` keeps the conditional modulation behavior cleaner and more interpretable


## 5. Recommended High-Level Design In This Repo

### Use SPADE only in the ControlNet first

Do not modify the main backbone `UNet` first.

Recommended first design:

- backbone `UNet`: unchanged
- `ControlNet`:
  - use the same copied stem/encoder/mid topology
  - replace control-branch `ResBlock`s with SPADE-aware `ResBlock`s
  - keep zero-conv outputs and additive skip injection unchanged

This is lower risk because:

- the backbone is pretrained and partly frozen
- the ControlNet is already the place where additional conditioning enters
- a ControlNet-local SPADE experiment isolates the new semantic-conditioning mechanism


## 6. Exact Architectural Changes

### 6.1 Add a SPADE-aware residual block

Create a block that mirrors `ResBlock` but uses:

- `GroupNorm(..., affine=False)`
- domain FiLM from `d_emb`
- segmentation modulation from `seg`

The current draft file is:

- `src/cyclenet/models/blocks/spade.py`

Recommended interface:

```python
class SPADEResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_dim: int,
        d_dim: int,
        seg_ch: int,
        dropout: float = 0.0,
        spade_hidden_ch: int = 128,
    ):
        ...

    def forward(
        self,
        x: torch.Tensor,
        seg: torch.Tensor,
        t_emb: torch.Tensor,
        d_emb: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

Important details:

- `seg_ch` should be the number of segmentation channels, for example `8`
- the SPADE convs must take `seg_ch` as input, not `in_ch` or `out_ch`
- `seg` must be resized to the current feature resolution before predicting `gamma_s` and `beta_s`

Recommended block internals:

```python
self.norm1 = nn.GroupNorm(32, in_ch, affine=False)
self.norm2 = nn.GroupNorm(32, out_ch, affine=False)

self.d1 = zero_module(nn.Linear(d_dim, 2 * in_ch))
self.d2 = zero_module(nn.Linear(d_dim, 2 * out_ch))

self.s1_shared = nn.Sequential(
    nn.Conv2d(seg_ch, spade_hidden_ch, kernel_size=3, padding=1),
    nn.SiLU(),
)
self.s1_gamma = zero_module(nn.Conv2d(spade_hidden_ch, in_ch, kernel_size=3, padding=1))
self.s1_beta = zero_module(nn.Conv2d(spade_hidden_ch, in_ch, kernel_size=3, padding=1))

self.s2_shared = nn.Sequential(
    nn.Conv2d(seg_ch, spade_hidden_ch, kernel_size=3, padding=1),
    nn.SiLU(),
)
self.s2_gamma = zero_module(nn.Conv2d(spade_hidden_ch, out_ch, kernel_size=3, padding=1))
self.s2_beta = zero_module(nn.Conv2d(spade_hidden_ch, out_ch, kernel_size=3, padding=1))
```

Recommended helper:

```python
import torch.nn.functional as F

def _spade_mod(self, seg: torch.Tensor, shared, gamma_head, beta_head, size):
    seg = F.interpolate(seg.float(), size=size, mode="nearest")
    h = shared(seg)
    gamma_s = gamma_head(h)
    beta_s = beta_head(h)
    return gamma_s, beta_s
```

Recommended forward logic:

```python
# block 1
h_norm = self.norm1(x)
gamma_d, beta_d = self._adagn_mod(d_emb, self.d1)
gamma_s, beta_s = self._spade_mod(
    seg, self.s1_shared, self.s1_gamma, self.s1_beta, x.shape[-2:]
)
h = h_norm * (1.0 + gamma_d + gamma_s) + (beta_d + beta_s)
h = self.act(h)
h = self.conv1(h)
h = h + self.t_proj(t_emb)[:, :, None, None]

# block 2
h_norm = self.norm2(h)
gamma_d, beta_d = self._adagn_mod(d_emb, self.d2)
gamma_s, beta_s = self._spade_mod(
    seg, self.s2_shared, self.s2_gamma, self.s2_beta, h.shape[-2:]
)
h = h_norm * (1.0 + gamma_d + gamma_s) + (beta_d + beta_s)
h = self.act(h)
h = self.drop(h)
h = self.conv2(h)
return h + skip
```


### 6.2 Add SPADE-aware encoder and bottleneck blocks for ControlNet

The current encoder/bottleneck code assumes `ResBlock.forward(x, t_emb, d_emb)`.

SPADE-aware versions need `seg` too:

```python
x = res_block(x, seg, t_emb, d_emb)
```

That means the cleanest implementation is to add new modules rather than force the old ones to handle both signatures.

Recommended new files or classes:

- `SPADEEncoderBlock`
- `SPADEBottleneck`

These should mirror:

- `src/cyclenet/models/blocks/encoder.py`
- `src/cyclenet/models/blocks/bottleneck.py`

But with `SPADEResBlock` replacing `ResBlock`.

Example encoder inner loop:

```python
for res_block, transformer_block in zip(self.res_blocks, self.transformer_blocks):
    x = res_block(x, seg, t_emb, d_emb)
    x = transformer_block(x, d_ctx)
    skips.append(x)
```

Example bottleneck:

```python
x = self.res1(x, seg, t_emb, d_emb)
x = self.transformer_block(x, d_ctx)
x = self.res2(x, seg, t_emb, d_emb)
```


### 6.3 Update `ControlNet` to use the SPADE-aware blocks

Current file:

- `src/cyclenet/models/controlnet.py`

Current behavior:

- deep-copy `stem`, `encoder`, `mid`, `t_mlp`
- run `c_stem(c_img)` once
- use zero-conv conditioning and copied encoder/mid blocks

For `seg-only + SPADE`, recommended changes are:

1. Keep copied weights as initialization sources.
2. Replace the copied encoder/mid blocks with SPADE-aware equivalents.
3. Pass raw `seg` through the control branch all the way down.
4. Keep zero-conv outputs and backbone skip injection unchanged.

The best constructor pattern is:

```python
class ControlNet(nn.Module):
    def __init__(self, backbone: UNet, seg_ch: int):
        super().__init__()
        self.t_mlp = copy.deepcopy(backbone.t_mlp)
        self.stem = copy.deepcopy(backbone.stem)

        self.encoder = build_spade_encoder_from_backbone(
            backbone.encoder,
            t_dim=backbone.t_mlp[0].out_features,
            d_dim=backbone.d_dim,
            seg_ch=seg_ch,
        )
        self.mid = build_spade_mid_from_backbone(
            backbone.mid,
            t_dim=backbone.t_mlp[0].out_features,
            d_dim=backbone.d_dim,
            seg_ch=seg_ch,
        )
        ...
```

In `forward()`, you no longer need a `c_stem` for a mixed RGB/seg condition if this design is truly `seg-only + SPADE`.

Instead:

```python
def forward(self, x, t, seg, d_emb):
    t_emb = sinusoidal_embedding(t, self.base_ch)
    t_emb = self.t_mlp(t_emb)

    d_ctx = None if d_emb is None else d_emb.unsqueeze(1)

    h = self.stem(x)

    # Optional very small seg-to-feature injection at input, but not required
    # for the first design.

    ctrl_skips = []

    for enc_block, enc_zero_conv in zip(self.encoder, self.encoder_zero_convs):
        h, skips = enc_block(h, seg, t_emb, d_emb, d_ctx)
        outs = enc_zero_conv(skips)
        ctrl_skips.extend(outs)

    h = self.mid(h, seg, t_emb, d_emb, d_ctx)
    ctrl_skips.append(self.mid_zero_conv(h))
    return ctrl_skips
```

This is a philosophical shift:

- the control branch no longer consumes source RGB as a second clean conditioning tensor
- it uses the noisy source image path `x_t` plus segmentation modulation


### 6.4 Update `CycleNet.forward()`

Current file:

- `src/cyclenet/models/cyclenet.py`

Today:

```python
ctrl_skips = self.control(x_t, t, c_img, from_emb)
```

For `seg-only + SPADE`, `c_img` should effectively be `seg`, so the call becomes:

```python
ctrl_skips = self.control(x_t, t, seg, from_emb)
```

This can still reuse the existing `c_img` parameter name if you want minimal surface change, but semantically it becomes the segmentation tensor.


### 6.5 Update loss-side conditioning assumptions

Current file:

- `src/cyclenet/diffusion/losses.py`

Current `seg-only` path is already close to what is needed:

```python
if use_rgb_condition:
    return torch.cat([img_norm, seg.float()], dim=1)
else:
    return seg.float()
```

For `seg-only + SPADE`, this is fine.

Important note:

- `x_0` and `y_0` still matter because they define `x_t`, `y_t`, and denoising targets
- they simply stop serving as clean control inputs in the ControlNet branch


### 6.6 Export SPADE blocks from `blocks/__init__.py`

File:

- `src/cyclenet/models/blocks/__init__.py`

Add:

```python
from .spade import SPADEResBlock, SPADEEncoderBlock, SPADEBottleneck
```

And include them in `__all__`.


## 7. Copy Initialization Strategy

The ControlNet philosophy in this repository is:

- start close to the backbone behavior
- add trainable control behavior gently

The same principle should be used for SPADE.

### Copy from the pretrained backbone:

- `stem`
- `t_mlp`
- `ResBlock.skip`
- `ResBlock.conv1`
- `ResBlock.conv2`
- `ResBlock.t_proj`
- transformer blocks
- downsample ops
- bottleneck transformer

### Initialize new SPADE pieces to identity:

- final `gamma` heads zero-initialized
- final `beta` heads zero-initialized
- domain modulation heads remain zero-initialized as they already are

Result:

- at initialization, the block behaves like plain normalized pretrained features
- training learns how much spatial segmentation control to apply

This is the SPADE analogue of zero-initialized ControlNet residuals.


## 8. Where This Matches Traditional SPADE

This design is faithful to the core SPADE idea in the following ways:

- segmentation controls features spatially
- normalization is modulated after normalization, not just concatenated at input
- modulation happens at multiple depths
- segmentation is resized to each feature resolution

Those are the essential SPADE behaviors.


## 9. Where This Diverges From Traditional SPADE

This design is not a textbook SPADE generator. Main differences:

### 9.1 SPADE is only applied in the ControlNet branch

Traditional SPADE often modulates the main generator throughout.

Here:

- the main diffusion backbone remains unchanged
- only the control branch is SPADE-aware

Likely effect:

- lower implementation risk
- weaker total semantic authority than full-backbone SPADE
- cleaner isolation of the conditioning change


### 9.2 Domain FiLM and SPADE are combined

Traditional SPADE usually uses only segmentation-conditioned modulation.

Here:

- a global domain-conditioned affine term is retained
- a spatial segmentation-conditioned affine term is added

Likely effect:

- preserves the repository's current domain-conditioning design
- lets ControlNet remain aware of source-domain context
- slightly departs from the pure SPADE formulation


### 9.3 The main noisy image path still exists

Traditional SPADE generators are often driven directly from semantic maps plus noise/style.

Here:

- the main backbone still sees `x_t`
- semantic conditioning only acts through ControlNet

Likely effect:

- stronger source-scene anchoring than a pure semantic generator
- better geometric faithfulness
- some residual source-faithfulness pressure remains


### 9.4 Minimal SPADE vs canonical shared MLP

A very minimal implementation could use only one `Conv2d(seg -> 2C)`.

That is acceptable as an ablation, but it is weaker than canonical SPADE.

Recommended design here is the small shared tower plus separate gamma/beta heads, because it is still lightweight while being closer to the original idea.


## 10. Expected Translation Behavior

### What should improve

- stronger preservation of semantic structure
- better class-region faithfulness
- more consistent class-specific texture placement
- less clean-RGB shortcut copying than RGB-conditioned ControlNet

### What may still remain

- some source-scene anchoring, because `x_t` still contains source image information
- possible loss of instance-specific detail not represented by the segmentation map
- class-generic textures if masks are coarse or inaccurate

### Structure vs texture

This design should improve both:

- structure: because segmentation modulates features spatially at every block
- texture/class-area realism: because each region gets class-conditioned local feature modulation

But it improves them in different ways:

- structure becomes more semantically faithful
- texture becomes more class-consistent, not more source-instance-faithful


## 11. Minimal Edit Checklist

If implementing this design, the main edit points are:

1. `src/cyclenet/models/blocks/spade.py`
   - finalize `SPADEResBlock`
   - add `SPADEEncoderBlock`
   - add `SPADEBottleneck`

2. `src/cyclenet/models/blocks/__init__.py`
   - export the new SPADE-aware blocks

3. `src/cyclenet/models/controlnet.py`
   - build SPADE-aware encoder/mid blocks instead of raw copies
   - pass segmentation through the branch
   - remove dependence on RGB control input for this design

4. `src/cyclenet/models/cyclenet.py`
   - pass segmentation to `ControlNet.forward()`

5. `src/cyclenet/diffusion/losses.py`
   - continue using `seg-only` conditioning
   - no major conceptual change beyond that

6. training / translation scripts
   - ensure `control_in_ch` for this design is the segmentation channel count only
   - ensure `use_rgb_condition=False`


## 12. Recommended First Implementation

For the first experiment, the most defensible version is:

- `seg-only`
- SPADE only in the `ControlNet`
- domain FiLM retained and combined additively with SPADE
- backbone unchanged
- canonical-ish SPADE shared tower per modulation site
- all SPADE heads zero-initialized

This is the cleanest architectural test of:

"Can strong semantic modulation in the control branch preserve structure while allowing freer, more target-realistic class-specific appearance?"

