# CycleNet Architecture

This document summarizes the architecture implemented in `src/cyclenet`, not the original reference code under `CycleNet/`.

## Configured Backbone Shape

From [`configs/unet/train_unet.yaml`](/Users/loganbrenningmeyer/Documents/GitHub/pixel-cyclenet/configs/unet/train_unet.yaml):

- `in_ch = 3`
- `base_ch = 128`
- `t_dim = 512`
- `d_dim = 128`
- `ch_mults = [1, 2, 4, 4]`
- `num_res_blocks = 2`
- `enc_heads = [0, 0, 0, 4]`
- `mid_heads = 4`

This gives the main spatial/channel path:

- `H x W x 3`
- `H x W x 128`
- `H/2 x W/2 x 128`
- `H/4 x W/4 x 256`
- `H/8 x W/8 x 512`
- `H/8 x W/8 x 512` bottleneck
- decoder mirrors back to `H x W x 128`
- final `3`-channel epsilon prediction

## 1. Model Topology

```mermaid
flowchart TD
    XT["Noisy image x_t\nB x 3 x H x W"]
    T["Timestep t\nB"]
    FROM["from_idx\nsource domain id"]
    TO["to_idx\ntarget domain id"]
    CIMG["Condition image c_img\nRGB or RGB+seg"]

    DE["DomainEmbedding\n2 x d_dim lookup"]
    TEMB["sinusoidal_embedding(t)\n-> Time MLP"]

    FROM --> DE
    TO --> DE
    T --> TEMB

    subgraph CTRL["ControlNet branch (trainable)"]
        C_STEM["Conditioning stem\n3x3 conv + GN + SiLU"]
        X_STEM_C["Copied UNet stem"]
        ENC_C["Copied encoder blocks\nsame widths as backbone"]
        MID_C["Copied bottleneck"]
        ZIN["input zero-conv 1x1"]
        ZSKIPS["per-skip zero-conv 1x1 blocks"]
        ZMID["mid zero-conv 1x1"]
        CTRLOUT["ctrl_skips list\nencoder skip controls + mid control"]
    end

    subgraph BB["Backbone UNet"]
        STEM["Stem\n3x3 conv + GN + SiLU"]
        ENC["Encoder\n4 EncoderBlocks"]
        MID["Bottleneck\nResBlock -> Transformer -> ResBlock"]
        DEC["Decoder\n4 DecoderBlocks"]
        FINAL["FinalLayer\nGN + SiLU + zero-init 3x3 conv"]
    end

    XT --> X_STEM_C
    XT --> STEM
    CIMG --> C_STEM
    TEMB --> ENC_C
    TEMB --> MID_C
    TEMB --> ENC
    TEMB --> MID
    FROM --> DE
    DE --> FROMEMB["from_emb"]
    FROMEMB --> ENC_C
    FROMEMB --> MID_C
    C_STEM --> ZIN
    X_STEM_C --> ADDIN["add conditioning to input stem"]
    ZIN --> ADDIN
    ADDIN --> ENC_C
    ENC_C --> ZSKIPS
    MID_C --> ZMID
    ZSKIPS --> CTRLOUT
    ZMID --> CTRLOUT

    TO --> DE
    DE --> TOEMB["to_emb"]
    STEM --> ENC
    TOEMB --> ENC
    TOEMB --> MID
    ENC --> MID
    MID --> ADDMID["add mid control"]
    CTRLOUT --> ADDMID
    ADDMID --> DEC
    TOEMB --> DEC
    DEC --> FINAL
    FINAL --> EPS["Predicted noise eps_hat\nB x 3 x H x W"]
```

## 2. Backbone Internals

```mermaid
flowchart TD
    X["x_t"]
    T["t_emb"]
    D["d_emb / d_ctx"]

    subgraph U["UNet Backbone"]
        S["Stem\nConv3x3 3->128\nGN + SiLU"]

        E1["EncoderBlock 1\n2 x ResBlock(128)\nno attention\nDownsample"]
        E2["EncoderBlock 2\n2 x ResBlock(256)\nno attention\nDownsample"]
        E3["EncoderBlock 3\n2 x ResBlock(512)\nno attention\nDownsample"]
        E4["EncoderBlock 4\n2 x ResBlock(512)\nTransformer blocks\nno downsample"]

        M["Bottleneck\nResBlock(512)\nTransformer(4 heads)\nResBlock(512)"]

        D4["DecoderBlock 1 @ H/8\n3 skip fusions\nTransformer blocks\nUpsample"]
        D3["DecoderBlock 2 @ H/4\n3 skip fusions\nno attention\nUpsample"]
        D2["DecoderBlock 3 @ H/2\n3 skip fusions\nno attention\nUpsample"]
        D1["DecoderBlock 4 @ H\n2 skip fusions\nno attention\nno upsample"]

        F["FinalLayer\nGN + SiLU + Conv3x3 128->3\nzero-init"]
    end

    X --> S --> E1 --> E2 --> E3 --> E4 --> M --> D4 --> D3 --> D2 --> D1 --> F
    T --> E1
    T --> E2
    T --> E3
    T --> E4
    T --> M
    T --> D4
    T --> D3
    T --> D2
    T --> D1
    D --> E1
    D --> E2
    D --> E3
    D --> E4
    D --> M
    D --> D4
    D --> D3
    D --> D2
    D --> D1
```

## 3. Per-Block Composition

```mermaid
flowchart LR
    X["x"]
    T["t_emb"]
    D["d_emb"]

    subgraph RB["ResBlock"]
        N1["GN or AdaGN"]
        A1["SiLU"]
        C1["Conv3x3"]
        TP["Linear(t_emb)\nadd as bias"]
        N2["GN or AdaGN"]
        A2["SiLU"]
        DR["Dropout"]
        C2["Conv3x3\nzero-init"]
        SK["Skip path\nIdentity or 1x1 conv"]
        SUM["Residual sum"]
    end

    X --> N1 --> A1 --> C1 --> TP --> N2 --> A2 --> DR --> C2 --> SUM
    X --> SK --> SUM
    T --> TP
    D --> N1
    D --> N2
```

Notes:

- `AdaGN` is GroupNorm plus FiLM-style scale/shift from the domain embedding.
- `conv2` in each `ResBlock` is zero-initialized.
- Attention blocks are also zero-initialized on the output projection.

## 4. ControlNet Injection Pattern

```mermaid
flowchart LR
    X["x_t"]
    C["c_img"]
    T["t_emb"]
    DF["from_emb"]

    SX["Copied UNet stem"]
    SC["Condition stem"]
    Z0["zero 1x1 conv"]
    ADD0["add"]

    ENC["Copied encoder"]
    ZS["ZeroConvBlock per skip"]
    MID["Copied bottleneck"]
    ZM["zero 1x1 conv"]
    CTRL["ctrl_skips"]

    X --> SX --> ADD0 --> ENC --> MID
    C --> SC --> Z0 --> ADD0
    T --> ENC
    T --> MID
    DF --> ENC
    DF --> MID
    ENC --> ZS --> CTRL
    MID --> ZM --> CTRL
```

The backbone decoder consumes these controls by summing them into:

- the bottleneck feature before decoding
- every backbone skip tensor before concatenation in each `DecoderBlock`

## 5. Trainable vs Frozen During CycleNet Training

```mermaid
flowchart TD
    subgraph FROZEN["Frozen after UNet pretraining"]
        F1["UNet stem"]
        F2["UNet time MLP"]
        F3["UNet encoder"]
        F4["UNet bottleneck"]
        F5["DomainEmbedding"]
    end

    subgraph TRAINABLE["Trainable in CycleNet"]
        T1["ControlNet branch"]
        T2["UNet decoder"]
        T3["UNet final layer"]
    end
```

## 6. Cycle Training Flow

```mermaid
flowchart TD
    X0["Clean source image x_0"]
    T["Sample timestep t"]
    EX["Noise eps_x"]
    QX["q_sample(x_0, t, eps_x)"]
    XT["x_t"]
    CX["c_x0\n= normalized x_0\nor [x_0, seg]"]

    R1["Recon pass\nx->x"]
    XY["Translation pass\nx->y\nno_unet_grad=True"]
    Y0["Predict y_0 from x_t and eps_xy"]
    EY["Noise eps_y"]
    QY1["q_sample(detach(y_0), t, eps_y)"]
    QY2["q_sample(y_0, t, eps_y)"]
    YT["y_t / y_t_c"]
    CY["c_y0\n= normalized detach(y_0)\nor [y_0, seg]"]
    YX["Cycle pass\ny->x"]
    XX["Consistency pass\nx-conditioned self pass on y_t"]
    YY["Invariance pass\ny->y"]

    LREC["recon loss"]
    LCYC["cycle loss"]
    LCON["consistency loss"]
    LINV["invariance loss"]

    X0 --> QX
    T --> QX
    EX --> QX
    QX --> XT
    X0 --> CX

    XT --> R1
    CX --> R1

    XT --> XY
    CX --> XY
    XY --> Y0
    Y0 --> QY1
    Y0 --> QY2
    EY --> QY1
    EY --> QY2
    T --> QY1
    T --> QY2
    QY1 --> YT
    QY2 --> YT
    Y0 --> CY

    YT --> YX
    CY --> YX

    YT --> XX
    CX --> XX

    XT --> YY
    CY --> YY

    R1 --> LREC
    XY --> LCYC
    YX --> LCYC
    XX --> LCON
    YY --> LINV
```

## 7. Practical Summary

- Your `CycleNet` is not a latent-diffusion model. It operates directly in pixel space and predicts image-space diffusion noise.
- `ControlNet` is initialized from a pretrained UNet encoder/bottleneck, then trained to inject source-domain structural and low-level conditioning into a mostly frozen backbone.
- Domain translation is done by:
  - `from_emb` driving the ControlNet branch
  - `to_emb` driving the backbone branch
  - `c_img` providing image or image+seg conditioning
- During CycleNet finetuning, most representation learning capacity sits in:
  - ControlNet zero-conv-conditioned branch
  - backbone decoder
  - backbone final output layer
