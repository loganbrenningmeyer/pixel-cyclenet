import torch
import torch.nn as nn
import torch.nn.functional as F

from cyclenet.utils import SelfAttentionBlock, sinusoidal_encoding


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        # -------------------------
        # Skip Projection
        # -------------------------
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

        # -------------------------
        # Time Embedding Projection
        # -------------------------
        self.t_proj = nn.Linear(t_dim, out_ch)

        # -------------------------
        # Activation / GroupNorms
        # -------------------------
        self.act   = nn.SiLU()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)

        # -------------------------
        # Convolutions
        # -------------------------
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # -- Initialize conv2 to zeros
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x += self.t_proj(t_emb)[:, :, None, None]
        
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += skip
        x *= (2 ** -0.5)
        return x


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    

class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    
    
    Args:
        in_ch (int): 
        out_ch (int): 
        skip_ch (int): 
        t_dim (int): 
        num_res_blocks (int): 
        num_heads (int): 
        is_down (bool): 
    
    Returns:
        x (Tensor): 
    """
    def __init__(
            self, 
            in_ch: int, 
            out_ch: int, 
            t_dim: int, 
            num_res_blocks: int,
            num_heads: int, 
            is_down: bool
    ):
        super().__init__()

        self.is_down = is_down
        self.num_skips = num_res_blocks + 1 if is_down else num_res_blocks

        # -------------------------
        # Residual Blocks
        # -------------------------
        self.res_blocks = nn.ModuleList(
            [ResBlock(in_ch,  out_ch, t_dim)] +
            [ResBlock(out_ch, out_ch, t_dim) for _ in range(num_res_blocks - 1)]
        )

        # -------------------------
        # Self-Attention
        # -------------------------
        if num_heads != 0:
            self.attn_blocks = nn.ModuleList([SelfAttentionBlock(out_ch, num_heads) for _ in range(num_res_blocks)])
        else:
            self.attn_blocks = nn.ModuleList([nn.Identity() for _ in range(num_res_blocks)])

        # -------------------------
        # Downsample
        # -------------------------
        self.down = Downsample(out_ch, out_ch) if is_down else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        skips = []
        # -------------------------
        # Residual Blocks / Self-Attention
        # -------------------------
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x, t_emb)
            x = attn_block(x)
            skips.append(x)

        # -------------------------
        # Downsample
        # -------------------------
        x = self.down(x)
        if self.is_down:
            skips.append(x)

        return x, skips


class DecoderBlock(nn.Module):
    """
    
    
    Args:
        in_ch (int): 
        out_ch (int): 
        skip_ch (int): 
        t_dim (int): 
        num_heads (int): 
        is_up (bool): 
    
    Returns:
        x (torch.Tensor): 
    """
    def __init__(
            self, 
            in_ch: int, 
            out_ch: int, 
            skip_chs: list[int], 
            t_dim: int, 
            num_heads: int, 
            is_up: bool
    ):
        super().__init__()
        # -- Track number of skip connections
        self.n_skips = len(skip_chs)
        # -------------------------
        # Residual Blocks
        # -------------------------
        self.res_blocks = nn.ModuleList(
            [ResBlock(in_ch  + skip_chs[0], out_ch, t_dim)] +
            [ResBlock(out_ch + skip_chs[i], out_ch, t_dim) for i in range(1, self.n_skips)]
        )

        # -------------------------
        # Self-Attention
        # -------------------------
        if num_heads != 0:
            self.attn_blocks = nn.ModuleList([SelfAttentionBlock(out_ch, num_heads) for _ in range(self.n_skips)])
        else:
            self.attn_blocks = nn.ModuleList([nn.Identity() for _ in range(self.n_skips)])

        # -------------------------
        # Upsampling
        # -------------------------
        self.up = Upsample(out_ch, out_ch) if is_up else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        # -------------------------
        # Residual Blocks / Self-Attention
        # -------------------------
        for res_block, attn_block, skip in zip(self.res_blocks, self.attn_blocks, skips):
            x = torch.cat([x, skip], dim=1)
            x = res_block(x, t_emb)
            x = attn_block(x)

        # -------------------------
        # Upsample
        # -------------------------
        x = self.up(x)

        return x
    

class Bottleneck(nn.Module):
    def __init__(self, in_ch: int, t_dim: int, num_heads: int):
        super().__init__()
        self.res1 = ResBlock(in_ch, in_ch, t_dim)
        self.attn = SelfAttentionBlock(in_ch, num_heads) if num_heads != 0 else nn.Identity()
        self.res2 = ResBlock(in_ch, in_ch, t_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x
    

class FinalLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_ch)
        self.act  = nn.SiLU()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # -- Initialize conv to zeros
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(
            self, 
            in_ch: int=3, 
            base_ch: int=128,
            num_res_blocks: int=2, 
            ch_mults: list[int]=[1,2,2,2],
            enc_heads: list[int]=[0,1,0,0],
            mid_heads: int=1
    ):
        """
        Diffusion UNet model based on the original DDPM tensorflow
        implementation. 
        
        Args:
            in_ch (int): Number of channels of input samples
            base_ch (int): Base channel width of the model
            num_res_blocks (int): Number of ResBlocks in each EncoderBlock
            ch_mults (list[int]): Base channel multipliers for each encoder layer
            enc_heads (list[int]): Number of attention heads in each encoder layer
            mid_heads (int): Number of attention heads in the Bottleneck
        """
        super().__init__()

        self.base_ch = base_ch
        self.num_res_blocks = num_res_blocks
        self.ch_mults = ch_mults

        # -------------------------
        # Time Embedding MLP
        # -------------------------
        t_dim = base_ch * 4
        self.t_mlp = nn.Sequential(
            nn.Linear(base_ch, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )

        # -------------------------
        # Stem
        # -------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, base_ch),
            nn.SiLU()
        )

        # -- Update post-stem in_ch
        in_ch = base_ch

        # -------------------------
        # Encoder
        # -------------------------
        skip_chs = []

        self.encoder = nn.ModuleList()
        for i, (ch_mult, num_heads) in enumerate(zip(ch_mults, enc_heads)):
            # -- No downsampling at final block
            down = (i != len(ch_mults) - 1)
            # -- Compute out_ch based on block's ch_mult
            out_ch = base_ch * ch_mult
            # -- Store EncoderBlock skip channels
            enc_skip_chs = [out_ch] * (num_res_blocks + 1) if down else [out_ch] * num_res_blocks
            skip_chs.extend(enc_skip_chs)
            # -- Initialize EncoderBlock
            self.encoder.append(EncoderBlock(in_ch, out_ch, t_dim, num_res_blocks, num_heads, down))
            # -- Update in_ch for next EncoderBlock
            in_ch = out_ch

        # -------------------------
        # Bottleneck
        # -------------------------
        self.mid = Bottleneck(in_ch, t_dim, mid_heads)

        # -------------------------
        # Decoder
        # -------------------------
        self.decoder = nn.ModuleList()
        for i, (ch_mult, num_heads) in enumerate(zip(ch_mults[::-1], enc_heads[::-1])):
            # -- No upsampling at final highest-resolution block
            up = (i != len(ch_mults) - 1)
            # -- Define out_ch
            out_ch = base_ch * ch_mult
            # -- Pop DecoderBlock skip channels (n_res at deepest level, n_res + 1 otherwise)
            num_skips = num_res_blocks if i == 0 else num_res_blocks + 1      # encoder n_res + 1 on downscale, n_res if not (deepest block)
            dec_skip_chs = [skip_chs.pop() for _ in range(num_skips)]
            # -- Initialize DecoderBlock
            self.decoder.append(DecoderBlock(in_ch, out_ch, dec_skip_chs, t_dim, num_heads, up))
            # -- Update in_ch for next DecoderBlock
            in_ch = out_ch

        # -------------------------
        # Final Layer
        # -------------------------
        self.final = FinalLayer(in_ch, 3)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # -------------------------
        # Time Embedding
        # -------------------------
        t_emb = sinusoidal_encoding(t, self.base_ch)
        t_emb = self.t_mlp(t_emb)

        # -------------------------
        # Stem
        # -------------------------
        x = self.stem(x)

        # -------------------------
        # Encoder
        # -------------------------
        skips = []

        for enc_block in self.encoder:
            x, skips_i = enc_block(x, t_emb)
            skips.extend(skips_i)

        # -------------------------
        # Bottleneck
        # -------------------------
        x = self.mid(x, t_emb)

        # -------------------------
        # Decoder
        # -------------------------
        for dec_block in self.decoder:
            skips_i = [skips.pop() for _ in range(dec_block.n_skips)]
            x = dec_block(x, t_emb, skips_i)

        # -------------------------
        # Final Layer
        # -------------------------
        x = self.final(x)

        return x