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
        - `src_idx` = SIM (0)
        - `tgt_idx` = REAL (1)
    - REAL $\rarr$ SIM
        - $x_0$: Image from REAL dataset
        - `cond_img`: Same image from REAL dataset
        - `src_idx` = REAL (1)
        - `tgt_idx` = SIM (0)

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

<br>

# CycleNet Loss Functions Implementation

In the official repo, the loss function implementations are different than the equations in the paper.

In the paper, the authors define $c_\text{text}$ as $c_\text{text} = \{c^+, c^-\}$, consisting of a conditional and an unconditional prompt, respectively. The conditional prompt $c^+$ is given to the UNet backbone, while the unconditional prompt $c^-$ is given to the ControlNet:

```
We keep the conditional prompt in the frozen SD encoder and the unconditional prompt in the ControlNet, so that the LDM backbone focuses on the translation and the side network looks for the semantics that needs modification.
```

Therefore, $c_x$ (for translating $y \to x$) and $c_y$ (for translating $x \to y$) are defined:

$$c_x = \{c_x^+, c_y^-\}$$
$$c_y = \{c_y^+, c_x^-\}$$

However, in the actual repo implementation, they do not follow using $c_x$ and $c_y$ as described in the paper. The actual implementation is given below:


## Reconstruction Loss

### Paper

$$\mathcal{L}_{x \to x} = \mathbb{E}_{x_0, \varepsilon_x} \|\varepsilon_\theta(x_t, c_x, x_0) - \varepsilon_x\|_2^2 \quad (5)$$

- In the paper, they use $c_x = \{c_x^+, c_y^-\}$, meaning the UNet receives $c_x^+$ while the ControlNet receives $c_y^-$.

### Code

- In the code, they use the source ($x$) embedding for both the conditional and unconditional prompts, i.e., they feed it to the UNet _and_ the ControlNet.

    - $\epsilon_\theta(x_t, c_{x \to x}, x_0)=$`model.forward(`$x_t, t,$`from_x_idx`$,$`to_x_idx`$,x_0$`)`


$$\mathcal{L}_\text{rec} = \mathbb{E}_{x_0, \epsilon_x} \Vert \epsilon_\theta(x_t, c_{x \to x}, x_0) - \epsilon_x \Vert_2^2$$



## Cycle Consistency Loss

### Paper

$$\mathcal{L}_{x \to y \to x} = \mathbb{E}_{x_0, \varepsilon_x, \varepsilon_y} \|\varepsilon_\theta(y_t, c_x, x_0) + \varepsilon_\theta(x_t, c_y, x_0) - \varepsilon_x - \varepsilon_y\|_2^2 \quad (7)$$

- In the paper, for $x \to y$ they use $c_y = \{c_y^+, c_x^-\}$, and for $y \to x$ they use $c_x = \{c_x^+, c_y^-\}$. 

### Code

- In the code, they break the cycle consistency loss function into separate "cycle" and "consistency" loss functions. 

- **Cycle Loss**

    1. **Noise $x_0$ ($\epsilon_x$)** 
        - $\epsilon_x=$`torch.randn_like(`$x_0$`)`
        - $x_t=$`q_sample(`$x_0,t,\epsilon_x$`)`
    
    $$\sqrt{\bar \alpha_t}x_0 + \sqrt{1 - \bar \alpha_t}\epsilon_x \to x_t$$

    2. **Predict noise ($x \to y$) & `detach()`**
        - $\epsilon_\theta(x_t, c_{x \to y}, x_0)=$`model.forward(`$x_t,t,$`from_x_idx`$,$ `to_y_idx`$,x_0$`).detach()`
    
    3. **Predict clean $\bar y_0$**
        - $\bar y_0=$`x0_from_eps(`$x_t,t,\epsilon_\theta(x_t, c_{x \to y}, x_0)$`)`

    4. **Noise $\bar y_0$**
        - $\epsilon_y=$`torch.randn_like(`$\bar y_0$`)`
        - $y_t=$`q_sample(`$\bar y_0,t,\epsilon_y$`)`

    5. **Predict noise ($y \to x$)**
        - $\epsilon_\theta(y_t, c_{y \to x}, \bar y_0)=$`model.forward(`$y_t,t,$`from_y_idx`$,$`to_x_idx`$,\bar y_0$`)`

    6. **Compute cycle loss**
        - $L_\text{cycle} = \mathbb{E}_{x_0,\epsilon_x,\epsilon_y}\Vert \epsilon_\theta(x_t, c_{x \to y}, x_0) + \epsilon_\theta(y_t, c_{y \to x}, \bar y_0) - \epsilon_x - \epsilon_y \Vert_2^2$

- **Consistency Loss**

    1. **Reuse $\epsilon_\theta(x_t, c_{x \to y}, x_0)$ prediction from cycle & `detach()`**
        - $\epsilon_\theta(x_t, c_{x \to y}, x_0)=$`model.forward(`$x_t,t,$`from_x_idx`$,$ `to_y_idx`$,x_0$`).detach()`

    2. **Predict noise for $y_t$ ($x \to x$)**
        - $\epsilon_\theta(y_t, c_{x \to x}, x_0)=$`model.forward(`$y_t,t,$`from_x_idx`$,$`to_x_idx`$,x_0$`)`

    3. **Compute consistency loss**
        - $L_\text{consis} = \mathbb{E}_{x_0,\epsilon_x,\epsilon_y}\Vert \epsilon_\theta(x_t, c_{x \to y}, x_0) + \epsilon_\theta(y_t, c_{x \to x}, x_0) - \epsilon_x - \epsilon_y \Vert_2^2$

- **Total Cycle-Consistency Loss**

$$w_\text{cycle} \times L_\text{cycle} + w_\text{consis} \times L_\text{consis}$$


## Invariance Loss

### Paper

$$\mathcal{L}_{x \to y \to y} = \mathbb{E}_{x_0, \varepsilon_x} \|\varepsilon_\theta(x_t, c_y, x_0) - \varepsilon_\theta(x_t, c_y, \bar{y}_0)\|_2^2 \quad (8)$$

- In the paper, for $x \to y$ with both $x_0$ and $\bar y_0$ as conditioning images they use $c_y = \{c_y^+,c_x^-\}$.

### Code

- In the code, they compute $\epsilon_\theta$ using $x_0$ as the conditioning image the same as in the paper:

    - $\epsilon_\theta(x_t, c_{x \to y}, x_0)=$`model.forward(`$x_t,t,$`from_x_idx`$,$`to_y_idx`$,x_0$`)`

- When using $\bar y_0$ as the conditioning image, they use the target ($y$) embedding for both conditional and unconditional prompts and detach:

    - $\epsilon_\theta(x_t, c_{y \to y}, \bar y_0)=$`model.forward(`$x_t,t,$`from_y_idx`$,$`to_y_idx`$,\bar y_0$`).detach()`

- **Total Invariance Loss**

$$L_\text{inv} = \mathbb{E}_{x_0,\epsilon_x,\epsilon_y}\Vert \epsilon_\theta(x_t, c_{x \to y}, x_0) - \epsilon_\theta(x_t, c_{y \to y}, \bar y_0) \Vert_2^2$$


## Total CycleNet Loss

$$L_\text{CycleNet} = w_\text{rec}L_\text{rec} + w_\text{cycle}L_\text{cycle} + w_\text{consis}L_\text{consis} + w_\text{inv}L_\text{inv}$$


<br>

# CycleNet Repo Variables

- `noise_x`$=\epsilon_x \quad$ = `eps_x`
- `noise_y`$=\epsilon_y \quad$ = `eps_y`
- `x_noise`$=x_t \quad$ = `x_t = q_sample(x_0, t, eps_x, sched)`

<br>

- `y_prime`$=\bar y_0 \quad$ = `y_0 = x0_from_eps(x_t, t, eps_xt_x2y_x0, sched)`
- `y_cond`$=\bar y_0 \quad$ = `y_0_cond = y_0.detach()`
- `y_noise`$=y_t \quad$ = `y_t = q_sample(y_0.detach(), t, eps_y, sched)`
- `y_noise_c`$=y_t \quad$ = `y_t_c = q_sample(y_0, t, eps_y, sched)`

<br>

- `noise_xy_prime`$=\epsilon_\theta(x_t, c_{x \to y}, x_0) \quad$ = `eps_xt_x2y_x0 = model.forward(x_t, t, src_idx, tgt_idx, x_0)`
- `noise_xy`$=\epsilon_\theta(x_t, c_{y \to y}, \bar y_0) \quad$ = `eps_xt_y2y_y0 = model.forward(x_t, t, tgt_idx, tgt_idx, y_0_cond)`
- `noise_yx`$=\epsilon_\theta(y_t, c_{x \to x}, x_0) \quad$ = `eps_yt_x2x_x0 = model.forward(y_t, t, src_idx, src_idx, x_0)`
- `noise_yx_c`$=\epsilon_\theta(y_t, c_{y \to x}, \bar y_0) \quad$ = `eps_yt_y2x_y0 = model.forward(y_t_c, t, tgt_idx, src_idx, y_0_cond)`

<br> 

- Reconstruction Loss
    - `recon_x_output`$=\epsilon_\theta(x_t, c_{x \to x}, x_0) \quad$ = `eps_xt_x2x_x0 = model.forward(x_t, t, src_idx, src_idx, x_0)`
    - `recon_x_target`$=\epsilon_x \quad$ = `eps_x`

<br>

- Cycle Loss
    - `cycle_output`$=\epsilon_\theta(x_t, c_{x \to y}, x_0) + \epsilon_\theta(y_t, c_{y \to x}, \bar y_0) \quad$ = `eps_xt_x2y_x0.detach() + eps_yt_y2x_y0`
    - `c_target`$=\epsilon_x + \epsilon_y \quad$ = `eps_x + eps_y`

<br>

- Consistency Loss
    - `consis_output`$=\epsilon_\theta(x_t, c_{x \to y}, x_0) + \epsilon_\theta(y_t, c_{x \to x}, x_0) \quad$ = `eps_xt_x2y_x0.detach() + eps_yt_x2x_x0`
    - `c_target`$=\epsilon_x + \epsilon_y \quad$ = `eps_x + eps_y`

- Invariance Loss
    - `disc_output`$=\epsilon_\theta(x_t, c_{x \to y}, x_0) \quad$ = `eps_xt_x2y_x0`
    - `disc_target`$=\epsilon_\theta(x_t, c_{y \to y}, \bar y_0) \quad$ = `eps_xt_y2y_y0.detach()`

