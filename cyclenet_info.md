# UNet Backbone

## Data

- Use both real and sim data in pretraining, mixing within batches
    - Balance sampling if real/sim proportions are wildly off

## Conditioning

- Pass correct `d_emb(domain_idx)` for each sample
- Compute `d_ctx = d_emb.unsqueeze(1)` in UNet `forward()`
- **No CFG dropout** in this stage

# CycleNet

## Data

- Use **either** only SIM samples or only REAL samples per batch
    - SIM $\rarr$ REAL
        - $x_0$: Image from SIM dataset
        - `cond_img`: Same image from SIM dataset
        - `src_idx` = SIM (1)
        - `tgt_idx` = REAL (0)
    - REAL $\rarr$ SIM
        - $x_0$: Image from REAL dataset
        - `cond_img`: Same image from REAL dataset
        - `src_idx` = REAL (0)
        - `tgt_idx` = SIM (1)

## Conditioning

- Pass **source embedding** to the **ControlNet**
- Pass **target embedding** to the **UNet backbone**

### CFG-style Dropout (Training)

- Replace target embedding with source embedding with probability `p`

### CFG-style Dropout (Sampling)

- **First Pass ($\epsilon_\text{cond}$)**: Run `forward()` with correct domain embeddings
    - SIM $\rarr$ ControlNet
    - REAL $\rarr$ UNet backbone
- **Second Pass ($\epsilon_\text{uncond}$)**: Replace target embedding with source embedding
    - SIM $\rarr$ ControlNet
    - SIM $\rarr$ UNet backbone
- Compute CFG weighting on the two predictions:
    - $\epsilon = \epsilon_\text{uncond} + w * (\epsilon_\text{cond} - \epsilon_\text{uncond})$

