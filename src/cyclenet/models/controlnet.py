import copy
import torch
import torch.nn as nn

from .unet import UNet
from .blocks import (
    ZeroConvBlock,
    ResBlock,
    SPADEResBlock,
    SPADEEncoderBlock,
    SPADEBottleneck,
)
from .conditioning import sinusoidal_embedding, control_in_channels
from .utils import zero_module, zero_skip_list


class ControlNet(nn.Module):
    def __init__(
        self, 
        backbone: UNet, 
        in_ch: int,
        skip_block_mask: list[bool] | None = None,
        use_mid_skip: bool = True,
    ):
        super().__init__()
        # -------------------------
        # Copy backbone UNet encoder/mid blocks
        # -------------------------
        self.t_mlp = copy.deepcopy(backbone.t_mlp)
        self.stem = copy.deepcopy(backbone.stem)
        self.encoder = copy.deepcopy(backbone.encoder)
        self.mid = copy.deepcopy(backbone.mid)

        self.base_ch = backbone.base_ch
        self.ch_mults = backbone.ch_mults

        # -------------------------
        # Conditioning stem
        # -------------------------
        self.c_stem = nn.Sequential(
            nn.Conv2d(in_ch, self.base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(32, self.base_ch),
            nn.SiLU(),
        )

        # -------------------------
        # Initialize zero-convolutions
        # -------------------------
        self.input_zero_conv = zero_module(
            nn.Conv2d(self.base_ch, self.base_ch, kernel_size=1)
        )

        self.encoder_zero_convs = nn.ModuleList()

        # -- EncoderBlocks
        assert len(self.ch_mults) == len(self.encoder)

        enc_out_ch = self.base_ch

        for ch_mult, enc_block in zip(self.ch_mults, self.encoder):
            enc_out_ch = self.base_ch * ch_mult
            # -- Initialize 1x1 zero conv for each skip
            num_skips = enc_block.num_skips
            self.encoder_zero_convs.append(ZeroConvBlock(enc_out_ch, num_skips))

        # -- Bottleneck (init to zeros)
        self.mid_zero_conv = zero_module(
            nn.Conv2d(enc_out_ch, enc_out_ch, kernel_size=1)
        )

        # -------------------------
        # EncoderBlocks / Bottleneck skip masks
        # -- True if ControlNet skips should be used (all True if None)
        # -------------------------
        if skip_block_mask is None:
            self.skip_block_mask = [True] * len(self.encoder)
        else:
            self.skip_block_mask = skip_block_mask
            assert len(self.skip_block_mask) == len(self.encoder), (
                f"skip_block_mask must have length {len(self.encoder)}, "
                f"got {len(self.skip_block_mask)}"
            )
            
        self.use_mid_skip = use_mid_skip

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_img: torch.Tensor,
        d_emb: torch.Tensor,
        seg: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Returns list of ControlNet skips to be consumed by the backbone
        Bottleneck and Decoder blocks
        """
        # -------------------------
        # Time embedding projection
        # -------------------------
        t_emb = sinusoidal_embedding(t, self.base_ch)
        t_emb = self.t_mlp(t_emb)

        # -------------------------
        # Domain embeddings -> context tokens
        # -------------------------
        d_ctx = None if d_emb is None else d_emb.unsqueeze(1)

        # -------------------------
        # Input stem / Conditioning stem
        # -------------------------
        h = self.stem(x)
        hc = self.c_stem(c_img)

        # -------------------------
        # Zero-conv conditioning / add to input
        # -------------------------
        h = h + self.input_zero_conv(hc)

        # -------------------------
        # Store zero-conv ControlNet skips
        # -------------------------
        ctrl_skips = []

        for block_idx, (enc_block, enc_zero_conv) in enumerate(
            zip(self.encoder, self.encoder_zero_convs)
        ):
            h, skips = enc_block(h, t_emb, d_emb, d_ctx)
            # -- Apply zero-convs
            block_ctrl_skips = enc_zero_conv(skips)

            # -- Optionally zero-out EncoderBlock skips
            if not self.skip_block_mask[block_idx]:
                block_ctrl_skips = zero_skip_list(block_ctrl_skips)

            ctrl_skips.extend(block_ctrl_skips)

        # -------------------------
        # Store Bottleneck skip
        # -------------------------
        h = self.mid(h, t_emb, d_emb, d_ctx)
        # -- Apply zero-conv
        mid_skip = self.mid_zero_conv(h)

        # -- Optionally zero-out Bottleneck skip
        if not self.use_mid_skip:
            mid_skip = torch.zeros_like(mid_skip)

        ctrl_skips.append(mid_skip)

        return ctrl_skips


class SPADEControlNet(nn.Module):
    """ """

    def __init__(
        self,
        backbone: UNet,
        in_ch: int,
        seg_ch: int,
        s_dim: int,
        skip_block_mask: list[bool] | None = None,
        use_mid_skip: bool = True,
    ):
        super().__init__()
        # -------------------------
        # Copy backbone's time embedding projection / stem directly
        # -------------------------
        self.t_mlp = copy.deepcopy(backbone.t_mlp)
        self.stem = copy.deepcopy(backbone.stem)

        self.base_ch = backbone.base_ch
        self.ch_mults = backbone.ch_mults

        # -------------------------
        # Conditioning stem
        # -------------------------
        self.c_stem = nn.Sequential(
            nn.Conv2d(in_ch, self.base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(32, self.base_ch),
            nn.SiLU(),
        )

        self.input_zero_conv = zero_module(
            nn.Conv2d(self.base_ch, self.base_ch, kernel_size=1)
        )

        # -------------------------
        # Initialize Encoder before copying weights
        # -------------------------
        self.encoder = nn.ModuleList()
        self.encoder_zero_convs = nn.ModuleList()

        in_feat_ch = self.base_ch
        for ch_mult, num_heads, enc_block in zip(
            backbone.ch_mults, backbone.enc_heads, backbone.encoder
        ):
            out_feat_ch = self.base_ch * ch_mult

            self.encoder.append(
                SPADEEncoderBlock(
                    in_ch=in_feat_ch,
                    out_ch=out_feat_ch,
                    seg_ch=seg_ch,
                    t_dim=backbone.t_dim,
                    d_dim=backbone.d_dim,
                    s_dim=s_dim,
                    num_res_blocks=backbone.num_res_blocks,
                    num_heads=num_heads,
                    is_down=enc_block.is_down,
                )
            )
            self.encoder_zero_convs.append(
                ZeroConvBlock(out_feat_ch, enc_block.num_skips)
            )
            in_feat_ch = out_feat_ch

        # -------------------------
        # Initialize Bottleneck before copying weights
        # -------------------------
        self.mid = SPADEBottleneck(
            in_ch=in_feat_ch,
            seg_ch=seg_ch,
            t_dim=backbone.t_dim,
            d_dim=backbone.d_dim,
            s_dim=s_dim,
            num_heads=(
                backbone.mid.transformer_block.num_heads
                if hasattr(backbone.mid.transformer_block, "num_heads")
                else 0
            ),
        )
        self.mid_zero_conv = zero_module(
            nn.Conv2d(in_feat_ch, in_feat_ch, kernel_size=1)
        )

        # -------------------------
        # EncoderBlocks / Bottleneck skip masks
        # -- True if ControlNet skips should be used (all True if None)
        # -------------------------
        if skip_block_mask is None:
            self.skip_block_mask = [True] * len(self.encoder)
        else:
            self.skip_block_mask = skip_block_mask
            assert len(self.skip_block_mask) == len(self.encoder), (
                f"skip_block_mask must have length {len(self.encoder)}, "
                f"got {len(self.skip_block_mask)}"
            )
            
        self.use_mid_skip = use_mid_skip

        # -------------------------
        # Copy backbone weights
        # -- All weights except ResBlock GroupNorms / SPADE weights
        # -------------------------
        copy_backbone_to_spade(backbone, self)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c_img: torch.Tensor,
        d_emb: torch.Tensor,
        seg: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Returns list of ControlNet skips to be consumed by the backbone
        Bottleneck and Decoder blocks

        [SPADE Modulation]
        - Incorporates segmentation mask into SPADEResBlocks using SPADE modulation
        """
        # -------------------------
        # Time embedding projection
        # -------------------------
        t_emb = sinusoidal_embedding(t, self.base_ch)
        t_emb = self.t_mlp(t_emb)

        # -------------------------
        # Domain embeddings -> context tokens
        # -------------------------
        d_ctx = None if d_emb is None else d_emb.unsqueeze(1)

        # -------------------------
        # Input stem / Conditioning stem
        # -------------------------
        h = self.stem(x)
        hc = self.c_stem(c_img)

        # -------------------------
        # Zero-conv conditioning / add to input
        # -------------------------
        h = h + self.input_zero_conv(hc)

        # -------------------------
        # Store zero-conv ControlNet skips
        # -------------------------
        ctrl_skips = []

        for block_idx, (enc_block, enc_zero_conv) in enumerate(
            zip(self.encoder, self.encoder_zero_convs)
        ):
            # -- Incorporate seg for SPADE modulation
            h, skips = enc_block(h, seg, t_emb, d_emb, d_ctx)

            # -- Apply zero-convs
            block_ctrl_skips = enc_zero_conv(skips)

            # -- Optionally zero-out EncoderBlock skips
            if not self.skip_block_mask[block_idx]:
                block_ctrl_skips = zero_skip_list(block_ctrl_skips)

            ctrl_skips.extend(block_ctrl_skips)

        # -------------------------
        # Store bottleneck skip
        # -------------------------
        # -- Incorporate seg for SPADE modulation
        h = self.mid(h, seg, t_emb, d_emb, d_ctx)
        # -- Apply zero-conv
        mid_skip = self.mid_zero_conv(h)

        # -- Optionally zero-out Bottleneck skip
        if not self.use_mid_skip:
            mid_skip = torch.zeros_like(mid_skip)

        ctrl_skips.append(mid_skip)

        return ctrl_skips


def copy_resblock_to_spade(src: ResBlock, dst: SPADEResBlock) -> None:
    """ """
    # -- Skip projection
    if isinstance(dst.skip, nn.Conv2d) and isinstance(src.skip, nn.Conv2d):
        dst.skip.load_state_dict(src.skip.state_dict(), strict=True)

    # -- Time embedding projection / Convolutions
    dst.t_proj.load_state_dict(src.t_proj.state_dict(), strict=True)
    dst.conv1.load_state_dict(src.conv1.state_dict(), strict=True)
    dst.conv2.load_state_dict(src.conv2.state_dict(), strict=True)

    # -- Domain AdaGN (copy if use_adagn == True)
    if getattr(src, "use_adagn", False) and getattr(dst, "use_adagn", False):
        dst.d1.load_state_dict(src.d1.state_dict(), strict=True)
        dst.d2.load_state_dict(src.d2.state_dict(), strict=True)


def copy_backbone_to_spade(backbone: UNet, control: SPADEControlNet) -> None:
    """ """
    # -------------------------
    # Encoder
    # -------------------------
    for src_enc, dst_enc in zip(backbone.encoder, control.encoder):
        # -- ResBlocks
        for src_res, dst_res in zip(src_enc.res_blocks, dst_enc.res_blocks):
            copy_resblock_to_spade(src_res, dst_res)

        # -- TransformerBlocks
        for src_tf, dst_tf in zip(
            src_enc.transformer_blocks, dst_enc.transformer_blocks
        ):
            # -- Ensure TransformerBlock != ContextIdentity
            if type(src_tf) is type(dst_tf):
                dst_tf.load_state_dict(src_tf.state_dict(), strict=True)

        # -- DownsampleBlock: Ensure DownsampleBlock != Identity
        if type(src_enc.down) is type(dst_enc.down):
            dst_enc.down.load_state_dict(src_enc.down.state_dict())

    # -------------------------
    # Bottleneck
    # -------------------------
    # -- ResBlocks
    copy_resblock_to_spade(backbone.mid.res1, control.mid.res1)
    copy_resblock_to_spade(backbone.mid.res2, control.mid.res2)

    # -- TransformerBlock
    if type(backbone.mid.transformer_block) is type(control.mid.transformer_block):
        control.mid.transformer_block.load_state_dict(
            backbone.mid.transformer_block.state_dict(),
            strict=True,
        )


def build_controlnet(
    backbone: UNet,
    cond_mode: str,
    num_seg_classes: int,
    use_spade: bool,
    s_dim: int | None,
    skip_block_mask: list[bool] | None = None,
    use_mid_skip: bool = True,
) -> ControlNet | SPADEControlNet:
    """

    """
    # -------------------------
    # Determine ControlNet input channels based on conditioning mode
    # -- RGB: 3
    # -- Segmentation: num_seg_classes
    # -- RGB + Segmentation: 3 + num_seg_classes
    # -------------------------
    in_ch = control_in_channels(cond_mode, num_seg_classes)

    # -------------------------
    # SPADEControlNet
    # -- Incorporates segmentation mask SPADE modulation
    # -------------------------
    if use_spade:
        return SPADEControlNet(
            backbone=backbone,
            in_ch=in_ch,
            seg_ch=num_seg_classes,
            s_dim=s_dim,
            skip_block_mask=skip_block_mask,
            use_mid_skip=use_mid_skip,
        )
    
    # -------------------------
    # ControlNet
    # -- Default ControlNet with no SPADE modulation
    # -------------------------
    else:
        return ControlNet(
            backbone=backbone,
            in_ch=in_ch,
            skip_block_mask=skip_block_mask,
            use_mid_skip=use_mid_skip,
        )



