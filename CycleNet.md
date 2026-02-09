

# CycleNet: Rethinking Cycle Consistency in Text-Guided Diffusion for Image Manipulation

Sihan Xu<sup>1\*</sup> Ziqiao Ma<sup>1\*</sup> Yidong Huang<sup>1</sup> Honglak Lee<sup>1,2</sup> Joyce Chai<sup>1</sup>

<sup>1</sup>University of Michigan, <sup>2</sup>LG AI Research  
{sihanxu,marstin,owenhji,honglak,chaijy}@umich.edu

## Abstract

Diffusion models (DMs) have enabled breakthroughs in image synthesis tasks but lack an intuitive interface for consistent image-to-image (I2I) translation. Various methods have been explored to address this issue, including mask-based methods, attention-based methods, and image-conditioning. However, it remains a critical challenge to enable unpaired I2I translation with pre-trained DMs while maintaining satisfying consistency. This paper introduces CycleNet, a novel but simple method that incorporates cycle consistency into DMs to regularize image manipulation. We validate CycleNet on unpaired I2I tasks of different granularities. Besides the scene and object level translation, we additionally contribute a multi-domain I2I translation dataset to study the physical state changes of objects. Our empirical studies show that CycleNet is superior in translation consistency and quality, and can generate high-quality images for out-of-domain distributions with a simple change of the textual prompt. CycleNet is a practical framework, which is robust even with very limited training data (around 2k) and requires minimal computational resources (1 GPU) to train.

![Figure 1: A high-resolution example of CycleNet for diffusion-based image-to-image translation compared to other diffusion-based methods. The figure shows a 2x5 grid of mountain landscape images. The columns are labeled: Input, CycleNet (Ours), P2P + NullText, Cycle Diffusion, and SDEdit. The rows are labeled: Summer -> Winter. Each image contains several dashed white boxes indicating areas for detailed comparison. The CycleNet column shows the most consistent and high-quality translations across the different methods.](0538daaa5583c23e17db3a12f2281a55_img.jpg)

Figure 1: A high-resolution example of CycleNet for diffusion-based image-to-image translation compared to other diffusion-based methods. The figure shows a 2x5 grid of mountain landscape images. The columns are labeled: Input, CycleNet (Ours), P2P + NullText, Cycle Diffusion, and SDEdit. The rows are labeled: Summer -> Winter. Each image contains several dashed white boxes indicating areas for detailed comparison. The CycleNet column shows the most consistent and high-quality translations across the different methods.

Figure 1: A high-resolution example of CycleNet for diffusion-based image-to-image translation compared to other diffusion-based methods. CycleNet produces high-quality translations with satisfactory consistency. The areas in the boxes are enlarged for detailed comparisons.

## 1 Introduction

Recently, pre-trained diffusion models (DMs) [38, 37, 39] have enabled an unprecedented breakthrough in image synthesis tasks. Compared to GANs [9] and VAEs [22], DMs exhibit superior stability and quality in image generation, as well as the capability to scale up to open-world multi-modal data. As such, pre-trained DMs have been applied to image-to-image (I2I) translation, which is to acquire a mapping between images from two distinct domains, e.g., different scenes, different

\*Equal contribution.

objects, and different object states. For such translations, text-guided diffusion models typically require mask layers [32, 2, 7, 1] or attention control [10, 30, 25, 35]. However, the quality of masks and attention maps can be unpredictable in complex scenes, leading to semantic and structural changes that are undesirable. Recently, researchers have explored using additional image-conditioning to perform paired I2I translations with the help of a side network [52] or an adapter [31]. Still, it remains an open challenge to adapt pre-trained DMs in *unpaired* I2I translation with a *consistency* guarantee.

We emphasize that *consistency*, a desirable property in image manipulation, is particularly important in unpaired I2I scenarios where there is no guaranteed correspondence between images in the source and target domains. Various applications of DMs, including video prediction and infilling [14], imagination-augmented language understanding [50], robotic manipulation [18, 8] and world models [46], would rely on strong consistency across the source and generated images.

To enable unpaired I2I translation using pre-trained DMs with satisfactory consistency, this paper introduces CycleNet, which allows DMs to translate a source image by conditioning on the input image and text prompts. More specifically, we adopt ControlNet [52] with pre-trained Stable Diffusion (SD) [38] as the latent DM backbone. Motivated by cycle consistency in GAN-based methods [56], CycleNet leverages consistency regularization over the image translation cycle. As illustrated in Figure 2, the image translation cycle includes a forward translation from  $x_0$  to  $\bar{y}_0$  and a backward translation to  $\bar{x}_0$ . The key idea of our method is to ensure that when conditioned on an image  $c_{\text{img}}$  that falls into the target domain specified by  $c_{\text{ext}}$ , the DM should be able to reproduce this image condition through the reverse process.

We validate CycleNet on I2I translation tasks of different granularities. Besides the scene and object level tasks introduced by Zhu et al. [56], we additionally contribute **ManiCups**, a multi-domain I2I translation dataset for manipulating physical state changes of objects. **ManiCups** contains 6k images of empty cups and cups of coffee, juice, milk, and water, collected from human-annotated bounding boxes. The empirical results demonstrate that compared to previous approaches, CycleNet is superior in translation faithfulness, cycle consistency, and image quality. Our approach is also computationally friendly, which is robust even with very limited training data (around 2k) and requires minimal computational resources (1 GPU) to train. Further analysis shows that CycleNet is a robust zero-shot I2I translator, which can generate faithful and high-quality images for out-of-domain distributions with a simple change of the textual prompt. This opens up possibilities to develop consistent diffusion-based image manipulation models with image conditioning and free-form language instructions.

## 2 Preliminaries

We start by introducing a set of notations to characterize image-to-image translation with DMs.

**Diffusion Models** Diffusion models progressively add Gaussian noise to a source image  $z_0 \sim q(z_0)$  through a forward diffusion process and subsequently reverse the process to restore the original image. Given a variance schedule  $\beta_1, \dots, \beta_T$ , the forward process is constrained to a Markov chain  $q(z_t | z_{t-1}) := \mathcal{N}(z_t; \sqrt{1 - \beta_t} z_{t-1}, \beta_t \mathbf{I})$ , in which  $z_{1:T}$  are latent variables with dimensions matching  $z_0$ . The reverse process  $p_\theta(z_{0:T})$  is as well Markovian, with learned Gaussian transitions that begin at  $z_T \sim \mathcal{N}(0, \mathbf{I})$ . Ho et al. [13] noted that the forward process allows the sampling of  $z_t$  at any time step  $t$  using a closed-form sampling function (Eq. 1).

$$z_t = S(z_0, \varepsilon, t) := \sqrt{\alpha_t} z_0 + \sqrt{1 - \alpha_t} \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \mathbf{I}) \text{ and } t \sim [1, T] \quad (1)$$

in which  $\alpha_t := 1 - \beta_t$  and  $\bar{\alpha}_t := \prod_{s=1}^t \alpha_s$ . Thus, the reverse process can be carried out with a UNet-based network  $\varepsilon_\theta$  that predicts the noise  $\varepsilon$ . By dropping time-dependent variances, the model can be trained according to the objective in Eq. 2.

$$\min_{\theta} \mathbb{E}_{z_0, \varepsilon, t} \|\varepsilon - \varepsilon_\theta(z_t, t)\|_2^2 \quad (2)$$

Eq. 2 implies that in principle, one could estimate the original source image  $z_0$  given a noised latent  $z_t$  at any time  $t$ . The reconstructed  $\bar{z}_0$  can be calculated with the generation function:

$$\bar{z}_0 = G(z_t, t) := [z_t - \sqrt{1 - \bar{\alpha}_t} \varepsilon_\theta(z_t, t)] / \sqrt{\bar{\alpha}_t} \quad (3)$$

For simplicity, we drop the temporal conditioning  $t$  in the following paragraphs.

![Figure 2: (a) The translation cycle in diffusion-based I2I translation. (b) CycleNet architecture using Stable Diffusion (SD) and ControlNet.](b230b8f21d8e82d55c0d311c8c32ef73_img.jpg)

(a) The translation cycle in diffusion-based I2I translation. The diagram shows a cycle of images and latent representations. On the left, a source image  $x_0$  (horse) is translated to a target image  $\bar{y}_0$  (horse in a different pose). On the right, a target image  $y_0$  (horse in a different pose) is translated back to a source image  $\bar{x}_0$  (horse). Latent representations  $x_t$  and  $y_t$  are shown in the middle, connected by forward and backward translation functions  $\epsilon_\theta$ . Dashed lines indicate regularization paths between  $x_0$  and  $\bar{x}_0$ , and between  $y_0$  and  $\bar{y}_0$ .

(b) CycleNet adopts Stable Diffusion (SD) as the backbone and ControlNet for conditioning. The diagram shows the architecture:  $z_0$  (sampled) and  $\epsilon$  (noise) are input to a Sampling block.  $z_0$  also goes to a Conditional prompt block and an SD Encoder.  $\epsilon$  goes to a Zero Conv block. The Conditional prompt and  $z_t$  (output of SD Encoder) are combined. The result goes to an SD Decoder. The output of SD Decoder is  $\epsilon_\theta$ .  $\epsilon_\theta$  and  $z_0$  are combined to produce  $\bar{z}_0$ .  $c_{img}$  (image condition) goes to a Zero Conv block. The output of this Zero Conv block and the output of the SD Encoder are combined. This result goes to another Zero Conv block. The output of this second Zero Conv block is  $\bar{z}_0$ .

Figure 2: (a) The translation cycle in diffusion-based I2I translation. (b) CycleNet architecture using Stable Diffusion (SD) and ControlNet.

Figure 2: The image translation cycle includes a forward translation from  $x_0$  to  $\bar{y}_0$  and a backward translation to  $\bar{x}_0$ . The key idea of our method is to ensure that when conditioned on an image  $c_{img}$  that falls into the target domain specified by  $c_{text}$ , the LDM should reproduce this image condition through the reverse process. The dashed lines indicate the regularization in the loss functions.

**Conditioning in Latent Diffusion Models** Latent diffusion models (LDMs) like Stable Diffusion [38] can model conditional distributions  $p_\theta(z_0|c)$  over condition  $c$ , e.g., by augmenting the UNet backbone with a condition-specific encoder using cross-attention mechanism [45]. Using textual prompts is the most common approach for enabling conditional image manipulation with LDMs. With a textual prompt  $c_z$  as conditioning, LDMs strive to learn a mapping from a latent noised sample  $z_t$  to an output image  $z_0$ , which falls into a domain  $\mathcal{Z}$  that is specified by the conditioning prompt. To enable more flexible and robust conditioning in diffusion-based image manipulation, especially a mixture of text and image conditioning, recent work obtained further control over the reverse process with a side network [52] or an adapter [31]. We denote such conditional denoising autoencoder as  $\epsilon_\theta(z_t, c_{text}, c_{img})$ , where  $c_{img}$  is the image condition and the text condition  $c_{text}$ . Eq. 3 can thus be rewritten as:

$$\bar{z}_0 = G(z_t, c_{text}, c_{img}) := [z_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(z_t, c_{text}, c_{img})] / \sqrt{\bar{\alpha}_t} \quad (4)$$

The text condition  $c_{text}$  contains a pair of conditional and unconditional prompts  $\{c^+, c^-\}$ . A conditional prompt  $c^+$  guides the diffusion process towards the images that are associated with it, whereas a negative prompt  $c^-$  drives the diffusion process away from those images.

**Consistency Regularization for Unpaired Image-to-Image Translation** The goal of unpaired image-to-image (I2I) translation is to learn a mapping between two domains  $\mathcal{X} \subset \mathbb{R}^d$  and  $\mathcal{Y} \subset \mathbb{R}^d$  with unpaired training samples  $\{x_i\}$  for  $i = 1, \dots, N$ , where  $x_i \in \mathcal{X}$  belongs to  $X$ , and  $\{y_j\}$  for  $j = 1, \dots, M$ , where  $y_j \in \mathcal{Y}$ . In traditional GAN-based translation frameworks, the task typically requires two mappings  $G : \mathcal{X} \to \mathcal{Y}$  and  $F : \mathcal{Y} \to \mathcal{X}$ . **Cycle consistency** enforces transitivity between forward and backward translation functions by regularizing pairs of samples, which is crucial in I2I translation, particularly in unpaired settings where no explicit correspondence between images in source and target domains is guaranteed [56, 27, 51, 44]. To ensure cycle consistency, CycleGAN [56] explicitly regularizes the translation cycle, bringing  $F(G(x))$  back to the original image  $x$ , and vice versa for  $y$ . Motivated by consistency regularization, we seek to enable consistent unpaired I2I translation with LDMs. Without introducing domain-specific generative models, we use one single denoising network  $\epsilon_\theta$  for translation by conditioning it on text and image prompts.

## 3 Method

In the following, we discuss only the translation from domain  $\mathcal{X}$  to  $\mathcal{Y}$  due to the symmetry of the backward translation. Our goal, at inference time, is to enable LDMs to translate a source image  $x_0$  by using it as the image condition  $c_{img} = x_0$ , and then denoise the noised latent  $y_t$  to  $\bar{y}_{t-1}$  with text prompts  $c_{text} = c_y$ . To learn such a translation model  $\epsilon_\theta(y_t, c_y, x_0)$ , we consider two types of training objectives. In the following sections, we describe the **cycle consistency regularization** to ensure cycle consistency so that the structures and unrelated semantics are preserved in the generated images, and the **self regularization** to match the distribution of generated images with the target

domain, As illustrated in Figure 2, the image translation cycle includes a forward translation from a source image  $x_0$  to  $\bar{y}_0$ , followed by a backward translation to the reconstructed source image  $\bar{x}_0$ .

### 3.1 Cycle Consistency Regularization

We assume a likelihood function  $P(z_0, c_{\text{text}})$  that the image  $z_0$  falls into the data distribution specified by the text condition  $c_{\text{text}}$ . We consider a generalized case of cycle consistency given the conditioning mechanism in LDMs. If  $P(c_{\text{img}}, c_{\text{text}})$  is close to 1, i.e., the image condition  $c_{\text{img}}$  falls exactly into the data distribution described by the text condition  $c_{\text{text}}$ , we should expect that  $G(z_t, c_{\text{text}}, c_{\text{img}}) = c_{\text{img}}$  for any noised latent  $z_t$ . With the translation cycle in Figure 2, the goal is to optimize (1)  $\mathcal{L}_{x \to x} = \mathbb{E}_{x_0, \varepsilon_x} \|x_0 - G(x_t, c_x, x_0)\|_2^2$ ; (2)  $\mathcal{L}_{y \to y} = \mathbb{E}_{x_0, \varepsilon_x, \varepsilon_y} \|\bar{y}_0 - G(y_t, c_y, \bar{y}_0)\|_2^2$ ; (3)  $\mathcal{L}_{x \to y \to x} = \mathbb{E}_{x_0, \varepsilon_x, \varepsilon_y} \|x_0 - G(y_t, c_x, x_0)\|_2^2$ ; and (4)  $\mathcal{L}_{x \to y \to y} = \mathbb{E}_{x_0, \varepsilon_x} \|\bar{y}_0 - G(x_t, c_y, \bar{y}_0)\|_2^2$ .

**Proposition 1** (Cycle Consistency Regularization). *With the translation cycle in Figure 2, a set of consistency losses is given by dropping time-dependent variances:*

$$\mathcal{L}_{x \to x} = \mathbb{E}_{x_0, \varepsilon_x} \|\varepsilon_\theta(x_t, c_x, x_0) - \varepsilon_x\|_2^2 \quad (5)$$

$$\mathcal{L}_{y \to y} = \mathbb{E}_{x_0, \varepsilon_x, \varepsilon_y} \|\varepsilon_\theta(y_t, c_y, \bar{y}_0) - \varepsilon_y\|_2^2 \quad (6)$$

$$\mathcal{L}_{x \to y \to x} = \mathbb{E}_{x_0, \varepsilon_x, \varepsilon_y} \|\varepsilon_\theta(y_t, c_x, x_0) + \varepsilon_\theta(x_t, c_y, x_0) - \varepsilon_x - \varepsilon_y\|_2^2 \quad (7)$$

$$\mathcal{L}_{x \to y \to y} = \mathbb{E}_{x_0, \varepsilon_x} \|\varepsilon_\theta(x_t, c_y, x_0) - \varepsilon_\theta(x_t, c_y, \bar{y}_0)\|_2^2 \quad (8)$$

We leave the proof in Section A.2. Proposition 1 states that pixel-level consistency can be acquired by regularizing the conditional denoising autoencoder  $\varepsilon_\theta$ . Specifically, the **reconstruction loss**  $\mathcal{L}_{x \to x}$  and  $\mathcal{L}_{y \to y}$  ensures that CycleNet can function as a LDM to reverse an image similar to Eq. 2. The **cycle consistency loss**  $\mathcal{L}_{x \to y \to x}$  serves as the transitivity regularization, which ensures that the forward and backward translations can reconstruct the original image  $x_0$ . The **invariance loss**  $\mathcal{L}_{x \to y \to y}$  requires that the target image domain stays invariant under forward translation, i.e., given a forward translation from  $x_t$  to  $\bar{y}_0$  conditioned on  $x_0$ , repeating the translation conditioned on  $\bar{y}_0$  would reproduce  $\bar{y}_0$ .

### 3.2 Self Regularization

In the previous section, while  $x_0$  is naturally sampled from domain  $\mathcal{X}$ , we need to ensure that the generated images fall in the target domain  $\mathcal{Y}$ , i.e., the translation leads to  $G(x_t, c_y, x_0) \in \mathcal{Y}$ . Our goal is therefore to maximize  $P(\bar{y}_0, c_y)$ , or equivalently to minimize

$$\mathcal{L}_{\text{LDM}} = -\mathbb{E}_{x_0, \varepsilon_x} P[G(S(x_0, \varepsilon), c_y, x_0), c_y] \quad (9)$$

**Assumption 1** (Domain Smoothness). *For any text condition,  $P(\cdot, c_{\text{text}})$  is  $L$ -Lipschitz.*

$$\exists L < \infty, |P(z_0^1, c_{\text{text}}) - P(z_0^2, c_{\text{text}})| \le L \|z_0^1 - z_0^2\|_2 \quad (10)$$

**Proposition 2** (Self Regularization). *Let  $\varepsilon_\theta^*$  denote the denoising autoencoder of the pre-trained text-guided LDM backbone. Let  $x_t = S(x_0, \varepsilon_x)$  be a noised latent. A self-supervised upper bound of  $\mathcal{L}_{\text{LDM}}$  is given by:*

$$\mathcal{L}_{\text{self}} = \mathbb{E}_{x_0, \varepsilon_x} \left[ L \sqrt{\frac{1 - \bar{\alpha}_t}{\bar{\alpha}_t}} \|\varepsilon_\theta(x_t, c_y, x_0) - \varepsilon_\theta^*(x_t, c_y)\|_2 \right] + \text{const} \quad (11)$$

Lipschitz assumptions have been widely adopted in diffusion methods [54, 48]. Assumption 1 hypothesizes that similar images share similar domain distributions. A self-supervised upper bound  $\mathcal{L}_{\text{self}}$  can be obtained in Proposition 2, which intuitively states that if the output of the conditional translation model does not deviate far from the pre-trained LDM backbone, the outcome image should still fall in the same domain specified by the textual prompt. We leave the proof in Section A.3.

### 3.3 CycleNet

In practice,  $\mathcal{L}_{\text{self}}$  can be minimized from the beginning of training by using a ControlNet [52] with pre-trained Stable Diffusion (SD) [38] as the LDM backbone, which is confirmed through preliminary

experiments. As shown in Figure 2, the model keeps the SD encoder frozen and makes a trainable copy in the side network. Additional zero convolution layers are introduced to encode the image condition and control the SD decoder. These zero convolution layers are 1D convolutions whose initial weights and biases vanish and can gradually acquire the optimized parameters from zero. Since the zero convolution layers keep the SD encoder features untouched,  $\mathcal{L}_{\text{self}}$  is minimal at the beginning of the training, and the training process is essentially fine-tuning a pre-trained LDM with a side network.

The text condition  $c_{\text{ext}} = \{c^+, c^-\}$  contains a pair of conditional and unconditional prompts. We keep the conditional prompt in the frozen SD encoder and the unconditional prompt in the ControlNet, so that the LDM backbone focuses on the translation and the side network looks for the semantics that needs modification. For example, to translate an image of summer to winter, we rely on a conditional prompt  $l_x = \text{"summer"}$  and unconditional prompt  $l_y = \text{"winter"}$ . Specifically, we use CLIP [36] encoder to encode the language prompts  $l_x$  and  $l_y$  such that  $c_x = \{\text{CLIP}(l_x), \text{CLIP}(l_y)\}$  and  $c_y = \{\text{CLIP}(l_y), \text{CLIP}(l_x)\}$ .

We also note that  $\mathcal{L}_{y \to y}$  can be omitted, as  $\mathcal{L}_{x \to x}$  can serve the same purpose in the symmetry of the translation cycle from  $\mathcal{Y}$  to  $\mathcal{X}$ , and early experiments confirmed that dropping this term lead to significantly faster convergence. The simplified objective is thus given by:

$$\mathcal{L}_x = \lambda_1 \mathcal{L}_{x \to x} + \lambda_2 \mathcal{L}_{x \to y \to y} + \lambda_3 \mathcal{L}_{x \to y \to x} \quad (12)$$

Consider both translation cycle from  $\mathcal{X} \leftrightarrow \mathcal{Y}$ , the complete training objective of CycleNet is:

$$\mathcal{L}_{\text{CycleNet}} = \mathcal{L}_x + \mathcal{L}_y \quad (13)$$

The pseudocode for training is given in Algo. 1.

### 3.4 FastCycleNet

Similar to previous cycle-consistent GAN-based models for unpaired I2I translation, there is a trade-off between the image translation quality and cycle consistency. Also, the cycle consistency loss  $\mathcal{L}_{x \to y \to x}$  requires deeper gradient descent, and therefore more computation expenses during training (Table 6). In order to speed up the training process in this situation, one may consider further removing  $\mathcal{L}_{x \to y \to x}$  from the training objective, and name this variation FastCycleNet. Through experiments, FastCycleNet can achieve satisfying consistency and competitive translation quality, as shown in Table 1. Different variations of models can be chosen depending on the practical needs.

## 4 Experiments

### 4.1 Benchmarks

**Scene/Object-Level Manipulation** We validate CycleNet on I2I translation tasks of different granularities. We first consider the benchmarks used in CycleGAN by Zhu et al. [56], which contains:

- (Scene Level) Yosemite `summer ↔ winter`: We use around 2k images of summer and winter Yosemite, with default prompts “summer” and “winter”;
- (Object Level) `horse ↔ zebra`: We use around 2.5k images of horses and zebras from the dataset with default prompts “horse” and “zebra”;
- (Object Level) `apple ↔ orange`: We use around 2k apple and orange images with default prompts of “apple” and “orange”.

**State Level Manipulation** Additionally, we introduce ManiCups<sup>1</sup>, a dataset of state-level image manipulation that tasks models to manipulate cups by filling or emptying liquid to/from containers, formulated as a multi-domain I2I translation dataset for object state changes:

- (State Level) ManiCups: We use around 5.7k images of empty cups and cups of coffee, juice, milk, and water for training. The default prompts are set as “empty cup” and “cup of <liquid>”. The task is to either empty a full cup or fill an empty cup with liquid as prompted.

<sup>1</sup>Our data is available at <https://huggingface.co/datasets/sled-umich/ManiCups>.

ManiCups is curated from human-annotated bounding boxes in publicly available datasets and Bing Image Search (under Share license for training and Modify for test set). We describe our three-stage data collection pipeline. In the **image collection** stage, we gather raw images of interest from MSCOCO [26], Open Images [23], as well as Bing Image Search API. In the **image extraction** stage, we extract regions of interest from the candidate images and resize them to a standardized size. Specifically, for subsets obtained from MSCOCO and Open Images, we extract the bounding boxes with labels of interest. All bounding boxes with an initial size less than  $128 \times 128$  are discarded, and the remaining boxes are extended to squares and resized to a standardized size of  $512 \times 512$  pixels. After this step, we obtained approximately 20k extracted and resized candidate images. We then control the data quality through a **filtering and labeling** stage. Our filtering process first discards replicated images using the L2 distance metric and remove images containing human faces, as well as cups with a front-facing perspective with a CLIP processor. Our labeling process starts with an automatic annotation with a CLIP classifier. To ensure the accuracy of the dataset, three human annotators thoroughly review the collected images, verifying that the images portray a top-down view of a container and assigning the appropriate labels to the respective domains. The resulting ManiCups dataset contains 5 domains, including 3 abundant domains (empty, coffee, juice) with more than 1K images in each category and 2 low-resource domains (water, milk) with less than 1K images to facilitate research and analysis in data-efficient learning.

To our knowledge, ManiCups is one of the first datasets targeted to the physical state changes of objects, other than stylistic transfers or type changes of objects. The ability to generate consistent state changes based on manipulation is fundamental for future coherent video prediction [14] as well as understanding and planning for physical agents [50, 18, 8, 46]. For additional details on data collection, processing, and statistics, please refer to Appendix B.

### 4.2 Experiment Setup

**Baselines** We compare our proposed models FastCycleNet and CycleNet to state-of-the-art methods for unpaired or zero-shot image-to-image translation.

- GAN-based methods: CycleGAN [56] and CUT [34];
- Mask-based diffusion methods: Direct inpainting with CLIPSeg [28] and Text2LIVE [2];
- Mask-free diffusion methods: ControlNet with Canny Edge [52], ILVR [5], EGSDE [54], SDEdit [29], Pix2Pix-Zero [35], MasaCtrl [3], CycleDiffusion [48], and Prompt2Prompt [10] with null-text inversion [30].

**Training** We train our model with a batch size of 4 on only one single A40 GPU.<sup>2</sup> Additional details on the implementations are available in Appendix C.

**Sampling** As shown in Figure 3, CycleNet has a good efficiency at inference time and more sampling steps lead to better translation quality. We initialize the sampling process with the latent noised input image  $z_t$ , collected using Equation 1. Following [29], a standard 50-step sampling is applied at inference time with  $t = 100$  for fair comparison.

![Figure 3: Step skipping during sampling. The figure shows a sequence of images illustrating the translation of a winter apple scene to a summer orange scene. The first image is labeled 'Input (winter, apple)'. Subsequent images are labeled '#Step = 1', '#Step = 2', '#Step = 5', '#Step = 10', '#Step = 20', and '#Step = 50'. The images show a progression from a winter scene with a red apple on a branch to a summer scene with an orange on a branch. A vertical label '↑ summer → orange' is present on the left side of the image grid.](5549f7bbf28f047575f40f7da371a217_img.jpg)

Figure 3: Step skipping during sampling. The figure shows a sequence of images illustrating the translation of a winter apple scene to a summer orange scene. The first image is labeled 'Input (winter, apple)'. Subsequent images are labeled '#Step = 1', '#Step = 2', '#Step = 5', '#Step = 10', '#Step = 20', and '#Step = 50'. The images show a progression from a winter scene with a red apple on a branch to a summer scene with an orange on a branch. A vertical label '↑ summer → orange' is present on the left side of the image grid.

Figure 3: Step skipping during sampling. The source image is from MSCOCO [26].

### 4.4 Quantitative Evaluation

We further use three types of evaluation metrics respectively to assess the quality of the generated image, the quality of translation, and the consistency of the images. For a detailed explanation of these evaluation metrics, we refer to Appendix C.4.

- **Image Quality.** To evaluate the quality of images, we employ two metrics: The naïve Fréchet Inception Distance (FID) [12] and FID<sub>clip</sub> [24] with CLIP [36];

![Figure 5: Qualitative comparison in coffee ↔ empty and empty ↔ juice tasks. The figure is a 4x7 grid of images. The first column shows the input images: (Empty) Cup of Coffee → Empty Cup, (Fill) Empty Cup → Cup of Coffee, (Empty) Cup of Juice → Empty Cup, and (Fill) Empty Cup → Cup of Juice. The next six columns show the results from different methods: CycleNet, Cycle Diffusion, P2P + NullText, SDEdit, Text2LIVE, and Inpaint + ClipSeg. The images show how well each method preserves the original image's style and content while performing the translation task.](c834b9abb4ddf70e5d10641f87d5ff5b_img.jpg)

Figure 5: Qualitative comparison in coffee ↔ empty and empty ↔ juice tasks. The figure is a 4x7 grid of images. The first column shows the input images: (Empty) Cup of Coffee → Empty Cup, (Fill) Empty Cup → Cup of Coffee, (Empty) Cup of Juice → Empty Cup, and (Fill) Empty Cup → Cup of Juice. The next six columns show the results from different methods: CycleNet, Cycle Diffusion, P2P + NullText, SDEdit, Text2LIVE, and Inpaint + ClipSeg. The images show how well each method preserves the original image's style and content while performing the translation task.

Figure 5: Qualitative comparison in coffee ↔ empty and empty ↔ juice tasks.

- **Translation Quality.** We use CLIPScore [11] to quantify the semantic similarity of the generated image and conditional prompt;
- **Translation Consistency.** We measure translation consistency using four different metrics: L2 Distance, Peak Signal-to-Noise Ratio (PSNR) [4], Structural Similarity Index Measure (SSIM) [47], and Learned Perceptual Image Patch Similarity (LPIPS) [53].

The evaluations are performed on the reserved test set. As demonstrated in the quantitative results from Table 1, we observe that some baselines display notably improved consistency in the ManiCups tasks. This could possibly be attributed to the fact that a considerable number of the test images were not successfully translated to the target domain, as can be seen in the qualitative results presented in Figure 5. Overall, CycleNet exhibits competitive and comprehensive performance in generating high-quality images in both global and local translation tasks, especially compared to the mask-based diffusion methods. Meanwhile, our methods ensure successful translations that fulfill the domain specified by text prompts while maintaining an impressive consistency from the original images.

## 5 Analysis and Discussions

### 5.1 Ablation study

Recall that FastCycleNet removes the cycle consistency loss  $\mathcal{L}_{x \to y \to x}$  from CycleNet. We further remove the invariance loss  $\mathcal{L}_{x \to y \to y}$  to understand the role of each loss term. For better control, we initialize the sampling process with the same random noise  $\epsilon$  rather than the latent noised image  $z_t$ . Table 2 shows our ablation study on `winter → summer` task with FID, CLIP score, and LPIPS. When both losses are removed, the model can be considered as a fine-tuned LDM backbone (row 4), which produces a high CLIP similarity score of 24.35. This confirms that the pre-trained LDM backbone can already make qualitative translations, while the LPIPS score of 0.61 implies a poor consistency from the original images. When we introduced the consistency constraint (row 3), the model’s LPIPS score improved significantly with a drop of the CLIP score to 19.89. This suggests a trade-off between cycle consistency and translation quality. When we introduced the invariance constraint (row 2), the model achieved the best translation quality with fair consistency. By introducing both constraints (row 1), CycleNet strikes the best balance between consistency at a slight cost of translation quality.

| Input                                                                | CycleNet                                           | Invariance Only                                           | Consistency Only                                           | None                                           | summer → winter  | FID ↓  | CLIP ↑       | LPIPS ↓     |
|----------------------------------------------------------------------|----------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------------|------------------------------------------------|------------------|--------|--------------|-------------|
| <p>Image: Input image: a winter landscape with a lake and trees.</p> | <p>Image: CycleNet result: a summer landscape.</p> | <p>Image: Invariance Only result: a summer landscape.</p> | <p>Image: Consistency Only result: a summer landscape.</p> | <p>Image: None result: a winter landscape.</p> | CycleNet         | 77.16  | 24.15        | <b>0.15</b> |
|                                                                      |                                                    |                                                           |                                                            |                                                | Invariance Only  | 76.23  | <b>25.13</b> | 0.23        |
|                                                                      |                                                    |                                                           |                                                            |                                                | Consistency Only | 84.18  | 19.89        | <b>0.14</b> |
|                                                                      |                                                    |                                                           |                                                            |                                                | None             | 211.26 | 24.35        | 0.61        |

Table 2: Ablation study over the cycle consistency loss and invariance loss.

### 5.2 Zero-shot Generalization to Out-of-Distribution Domains

CycleNet performs image manipulation with text and image conditioning, making it potentially generalizable to out-of-distribution (OOD) domains with a simple change of the textual prompt. As illustrated in Figure 6, we demonstrate that CycleNet has a remarkable capability to generate faithful and high-quality images for unseen domains. These results highlight the robustness and adaptability of CycleNet to make the most out of the pre-trained LDM backbone to handle unseen scenarios. This underscores the potential to apply CycleNet for various real-world applications and paves the way for future research in zero-shot learning and OOD generalization.

### 5.3 Translation Diversity

Diversity is an important feature of image translation models. As shown in Figure 6, we demonstrate that CycleNet can generate a variety of images that accurately satisfy the specified translation task in the text prompts, while maintaining consistency.

![Figure 6: Examples of output diversity and zero-shot generalization to out-of-domain distributions. The figure is a 2x6 grid of images. The top row shows seasonal translations: 'winter' to 'summer', 'summer' to 'fall', 'fall' to 'fire', and 'fire' to 'minecraft'. The bottom row shows animal translations: 'horse' to 'zebra', 'zebra' to 'unicorn', 'unicorn' to 'sky', and 'sky' to 'desert'. Each row has two images for each source-target pair, demonstrating diversity in the generated results.](a86610f7a0e579fec9f34dea52fa088b_img.jpg)

Figure 6: Examples of output diversity and zero-shot generalization to out-of-domain distributions. The figure is a 2x6 grid of images. The top row shows seasonal translations: 'winter' to 'summer', 'summer' to 'fall', 'fall' to 'fire', and 'fire' to 'minecraft'. The bottom row shows animal translations: 'horse' to 'zebra', 'zebra' to 'unicorn', 'unicorn' to 'sky', and 'sky' to 'desert'. Each row has two images for each source-target pair, demonstrating diversity in the generated results.

Figure 6: Examples of output diversity and zero-shot generalization to out-of-domain distributions.

### 5.4 Limitations: Trade-off between consistency and translation

There have been concerns that cycle consistency could be too restrictive for some translation task [55]. As shown in Figure 7, while CycleNet maintains a strong consistency over the input image, the quartered apple failed to be translated into its orange equivalence. In GAN-based methods, local discriminators have been proposed to address this issue [57], yet it remains challenging to keep global consistency while making faithful local edits for LDM-based approaches.

![Figure 7: The trade-off between consistency and translation. The figure shows five columns of images. The first column is the 'Input' showing a green apple and a quartered apple. The second column is 'CycleNet' showing a green apple and a quartered orange. The third column is 'P2P + NullText' showing a green apple and a whole orange. The fourth column is 'Cycle Diffusion' showing a green apple and a quartered orange. The fifth column is 'SDEdit' showing a green apple and a quartered orange. This illustrates how different methods handle the consistency vs. translation trade-off.](e2eb8b8c35f32b665245d2c24d337dca_img.jpg)

Figure 7: The trade-off between consistency and translation. The figure shows five columns of images. The first column is the 'Input' showing a green apple and a quartered apple. The second column is 'CycleNet' showing a green apple and a quartered orange. The third column is 'P2P + NullText' showing a green apple and a whole orange. The fourth column is 'Cycle Diffusion' showing a green apple and a quartered orange. The fifth column is 'SDEdit' showing a green apple and a quartered orange. This illustrates how different methods handle the consistency vs. translation trade-off.

Figure 7: The trade-off between consistency and translation.

## 6 Related Work

### 6.1 Conditional Image Manipulation with Diffusion Models

Building upon Diffusion Probabilistic Models [42, 13], pre-trained Diffusion Models (DMs) [38, 37, 39] have achieved state-of-the-art performance in image generation tasks. Text prompts are the most common protocol to enable conditional image manipulation with DMs, which can be done by fine-tuning a pre-trained DM [19, 20]. Mask-based methods have also been proposed with the help of user-prompted/automatically generated masks [32, 7, 1] or augmentation layers [2]. To refrain from employing additional masks, recent work has explored attention-based alternatives [10, 30, 25, 35].

They first invert the source images to obtain the cross-attention maps and then perform image editing with attention control. The promising performance of these methods is largely dependent on the quality of attention maps, which cannot be guaranteed in images with complicated scenes and object relationships, leading to undesirable changes. Very recently, additional image-conditioning has been explored to perform paired image-to-image translation, using a side network [52] or an adapter [31]. In this work, we follow this line of research and seek to enable unpaired image-to-image translation with pre-trained DMs while maintaining a satisfactory level of consistency.

### 6.2 Unpaired Image-to-Image Translation

Image-to-image translation (I2I) is a fundamental task in computer vision, which is concerned with learning a mapping across images of different domains. Traditional GAN-based methods [16] require instance-level paired data, which are difficult to collect in many domains. To address this limitation, the unpaired I2I setting [56, 27] was introduced to transform an image from the source domain  $\mathcal{X}$  into one that belongs to the target domain  $\mathcal{Y}$ , given only unpaired images from each domain. Several GAN-based methods [56, 51, 27, 49] were proposed to address this problem. In recent years, DPMs have demonstrated their superior ability to synthesize high-quality images, with several applications in I2I translation [40, 5]. With the availability of pre-trained DMs, SDEdit [29] changes the starting point of generation by using a noisy source image that preserves the overall structure. EGSDE [54] combines the merit of ILVR and SDEdit by introducing a pre-trained energy function on both domains to guide the denoising process. While these methods result in leading performance on multiple benchmarks, it remains an open challenge to incorporate pre-trained DMs for high-quality image generation, and at the same time, to ensure translation consistency.

### 6.3 Cycle Consistency in Image Translation

The idea of cycle consistency is to regularize pairs of samples by ensuring transitivity between the forward and backward translation functions [41, 33, 17]. In unpaired I2I translation where explicit correspondence between source and target domain images is not guaranteed, cycle consistency plays a crucial role [56, 21, 27]. Several efforts were made to ensure cycle consistency in diffusion-based I2I translation. UNIT-DDPM [40] made an initial attempt in the unpaired I2I setting, training two DPMs and two translation functions from scratch. Cycle consistency losses are introduced in the translation functions during training to regularize the reverse processes. At inference time, the image generation does not depend on the translation functions, but only on the two DPMs in an iterative manner, leading to sub-optimal performance. Su et al. [43] proposed the DDIB framework that exact cycle consistency is possible assuming zero discretization error, which does not enforce any cycle consistency constraint itself. Cycle Diffusion [48] proposes a zero-shot approach for image translation based on Su et al. [43]’s observation that a certain level of consistency could emerge from DMs, and there is no explicit treatment to encourage cycle consistency. To the best of our knowledge, CycleNet is the first to guarantee cycle consistency in unpaired image-to-image translation using pre-trained diffusion models, with a simple trainable network and competitive performance.

## 7 Conclusion

The paper introduces CycleNet that incorporates the concept of cycle consistency into text-guided latent diffusion models to regularize the image translation tasks. CycleNet is a practical framework for low-resource applications where only limited data and computational power are available. Through extensive experiments on unpaired I2I translation tasks at scene, object, and state levels, our empirical studies show that CycleNet is promising in consistency and quality, and can generate high-quality images for out-of-domain distributions with a simple change of the textual prompt.

**Future Work** This paper is primarily concerned with the unpaired I2I setting, which utilizes images from unpaired domains during training for domain-specific applications. Although CycleNet demonstrates robust out-of-domain generalization, enabling strong zero-shot I2I translation capabilities is not our focus here. We leave it to our future work to explore diffusion-based image manipulation with image conditioning and free-form language instructions, particularly in zero-shot settings.