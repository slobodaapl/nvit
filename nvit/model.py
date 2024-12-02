import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from nvit.kohonen import KohonenMap


@dataclass
class ViTConfig:
    image_size: int = 224
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0 ** 0.5)    # 1 / sqrt(n_embd)
    use_nViT: int = 0
    sz_init_value: float = 1.00
    sz_init_scaling: float = 1.0
    dropout: float = 0.0
    bias: bool = False
    channels: int = 3
    num_classes: int = 1000
    local_patch_size: int = 8  # Size of local patches
    global_patch_size: int = 16  # Size of global patches
    kohonen_nodes: int = 512  # Total number of Kohonen nodes
    kohonen_alpha: float = 0.01  # Learning rate for Kohonen maps
    use_kohonen: bool = True
    reconstruction_weight: float = 0.1
    map_balance_weight: float = 0.5  # Learnable weight between local/global maps


class Block(nn.Module):

    def __init__(self, config: ViTConfig, iblock: int) -> None:
        super(Block, self).__init__()
        self.config = config

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias)
        self.silu = nn.SiLU()
        self.mlp_c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        if (config.use_nViT == 0):
            self.rmsnorm_att = RMSNorm(config.n_embd)
            self.rmsnorm_mlp = RMSNorm(config.n_embd)

        if (config.use_nViT == 1):
            self.attn_alpha_init_value = torch.scalar_tensor(0.05, dtype=torch.float32)
            self.attn_alpha_init_scaling = torch.scalar_tensor(config.base_scale, dtype=torch.float32)
            self.attn_alpha = torch.nn.Parameter(self.attn_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.mlp_alpha_init_value = torch.scalar_tensor(0.05, dtype=torch.float32)
            self.mlp_alpha_init_scaling = torch.scalar_tensor(config.base_scale, dtype=torch.float32)
            self.mlp_alpha = torch.nn.Parameter(self.mlp_alpha_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.sqk_init_value = torch.scalar_tensor(1.0, dtype=torch.float32)
            self.sqk_init_scaling = torch.scalar_tensor(config.base_scale, dtype=torch.float32)
            self.sqk = torch.nn.Parameter(self.sqk_init_scaling*torch.ones(self.config.n_embd, dtype=torch.float32))

            self.suv_init_value = torch.scalar_tensor(1.0, dtype=torch.float32)
            self.suv_init_scaling = torch.scalar_tensor(1.0, dtype=torch.float32)
            self.suv = torch.nn.Parameter(self.suv_init_scaling*torch.ones(2 * 4 * config.n_embd, dtype=torch.float32))


    def justnorm(self, x: torch.Tensor) -> torch.Tensor:
        #return F.normalize(x, p=2, dim=-1)
        res = x / x.norm(p=2, dim=-1, keepdim=True)
        return res

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, C = h.size()  # batch, sequence_length, embedding_dim

        if (self.config.use_nViT == 0):
            h = self.rmsnorm_att(h)

        # Project to q, k, v
        q = self.query(h)  # [B, T, C]
        k = self.key(h)    # [B, T, C]
        v = self.value(h)  # [B, T, C]

        # Split embedding dim into heads: [B, T, C] -> [B, H, T, D]
        q = rearrange(q, "b t (h d) -> b h t d", h=self.config.n_head)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.config.n_head)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.config.n_head)

        if (self.config.use_nViT == 1):
            sqk = (self.sqk * (self.sqk_init_value/self.sqk_init_scaling))
            sqk = rearrange(sqk, "(h d) -> 1 h 1 d", h=self.config.n_head)
            q = sqk * self.justnorm(q)
            k = sqk * self.justnorm(k)

        head_size = C // self.config.n_head
        sqrt_head_dim = head_size ** 0.5
        softmax_scale = 1.0 / sqrt_head_dim if self.config.use_nViT == 0 else sqrt_head_dim

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # Get attention output [B, H, T, D]
        attn_output = cast(torch.Tensor,flex_attention(q, k, v, scale=softmax_scale))

        # Merge heads back: [B, H, T, D] -> [B, T, C]
        h_att = rearrange(attn_output, "b h t d -> b t (h d)")

        # Project attention output
        h_att = self.att_c_proj(h_att)

        if (self.config.use_nViT == 0):
            h = h + h_att
        if (self.config.use_nViT == 1):
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = self.justnorm(h)
            B_norm = self.justnorm(h_att)

            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        # MLP block
        if (self.config.use_nViT == 0):
            h = self.rmsnorm_mlp(h)

        uv = self.c_fc(h)
        if (self.config.use_nViT == 1):
            suv = (self.suv * ((self.suv_init_value/self.suv_init_scaling) * (self.config.n_embd ** 0.5)))
            uv = suv * uv

        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)

        if (self.config.use_nViT == 0):
            h = h + h_mlp
        if (self.config.use_nViT == 1):
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = self.justnorm(h)
            B_norm = self.justnorm(h_mlp)

            res = A_norm + lr * (B_norm - A_norm)
            h = self.justnorm(res)

        return h


class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight


class CrossAttentionBlock(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config

        self.local_norm = RMSNorm(config.n_embd)
        self.global_norm = RMSNorm(config.n_embd)

        self.q_local = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_global = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_global = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, local: torch.Tensor, global_: torch.Tensor) -> torch.Tensor:
        local = self.local_norm(local)
        global_ = self.global_norm(global_)

        q = self.q_local(local)
        k = self.k_global(global_)
        v = self.v_global(global_)

        # Split heads
        q = rearrange(q, "b t (h d) -> b h t d", h=self.config.n_head)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.config.n_head)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.config.n_head)

        # Cross attention
        scale = (self.config.n_embd // self.config.n_head) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = rearrange(out, "b h t d -> b t (h d)")

        return self.proj(out)

class ViT(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # Local patch embedding (no padding needed)
        self.local_patch_embed = nn.Conv2d(
            config.channels,
            config.n_embd,
            kernel_size=config.local_patch_size,
            stride=config.local_patch_size,
        )

        # Calculate padding for global patches to align with local patches
        # This ensures global patches are centered on local patches
        global_padding = (config.global_patch_size - config.local_patch_size) // 2
        self.global_patch_embed = nn.Sequential(
            nn.ReflectionPad2d(global_padding),
            nn.Conv2d(
                config.channels,
                config.n_embd,
                kernel_size=config.global_patch_size,
                stride=config.local_patch_size,  # Match local patch stride for alignment
            )
        )

        # Calculate number of patches for position embeddings
        n_local_patches = (config.image_size // config.local_patch_size) ** 2
        self.local_pos_embed = nn.Parameter(torch.zeros(1, n_local_patches, config.n_embd))
        self.global_pos_embed = nn.Parameter(torch.zeros(1, n_local_patches, config.n_embd))

        # Kohonen maps
        if config.use_kohonen:
            self.local_kohonen = KohonenMap(
                config.n_embd,
                config.kohonen_nodes // 2,
                config.kohonen_alpha,
            )
            self.global_kohonen = KohonenMap(
                config.n_embd,
                config.kohonen_nodes // 2,
                config.kohonen_alpha,
            )
            self.map_balance = nn.Parameter(torch.tensor(config.map_balance_weight))

        # Cross attention for combining local and global features
        self.cross_attention = CrossAttentionBlock(config)

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.n_embd, config.local_patch_size * config.local_patch_size * config.channels),
            nn.Tanh(),
        )

        # Transformer blocks
        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, il) for il in range(config.n_layer)]),
        ))

        # Output head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.num_classes),
        )

        if self.config.use_nViT == 1:
            self.sz = torch.nn.Parameter(
                self.config.sz_init_scaling * torch.ones(config.num_classes, dtype=torch.float32),
            )

        # Initialize weights
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        if self.config.use_nViT == 1 and isinstance(module, nn.Linear):
            torch.nn.init.constant_(self.sz, self.config.sz_init_value)

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str) -> torch.optim.AdamW:
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        if self.config.use_nViT == 1:
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_dict.items() if "sz" not in n and p.dim() >= 2], "weight_decay": weight_decay},
                {"params": [p for n, p in param_dict.items() if "sz" not in n and p.dim() < 2], "weight_decay": 0.0},
                {"params": [self.sz], "weight_decay": 0.0},
            ]
        else:
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_dict.items() if p.dim() >= 2], "weight_decay": weight_decay},
                {"params": [p for n, p in param_dict.items() if p.dim() < 2], "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, betas=betas, fused=True if device_type == "cuda" else False)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> tuple[float, float]:
        """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # First estimate the number of flops we do per iteration.
        # See: https://github.com/pytorch/pytorch/issues/110656
        N = sum(p.numel() for p in self.parameters())
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.image_size//cfg.local_patch_size * cfg.image_size//cfg.local_patch_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # Express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu, flops_achieved

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Get local and global patches
        local_patches = self.local_patch_embed(img)  # No padding needed
        global_patches = self.global_patch_embed(img)  # Includes reflection padding

        # Reshape and add position embeddings
        local_patches = local_patches.flatten(2).transpose(1, 2)
        global_patches = global_patches.flatten(2).transpose(1, 2)

        local_patches = local_patches + self.local_pos_embed
        global_patches = global_patches + self.global_pos_embed

        aux_losses = {}

        if self.config.use_kohonen:
            # Apply Kohonen maps with current learning rate
            lr = self.scheduler(self.step, self.total_steps) if hasattr(self, "scheduler") else 1.0

            # Process local and global features
            local_repr, local_indices = self.local_kohonen(local_patches)
            global_repr, global_indices = self.global_kohonen(global_patches)

            # Update Kohonen nodes during training
            if self.training:
                self.local_kohonen.update_nodes(local_patches, local_indices, lr)
                self.global_kohonen.update_nodes(global_patches, global_indices, lr)

            # Combine representations using learnable balance
            balance_weight = torch.sigmoid(self.map_balance)
            patches = balance_weight * local_repr + (1 - balance_weight) * global_repr

            # Calculate Kohonen map losses
            aux_losses["kohonen_local"] = F.mse_loss(local_repr, local_patches)
            aux_losses["kohonen_global"] = F.mse_loss(global_repr, global_patches)
        else:
            # Without Kohonen maps, use cross attention directly
            patches = self.cross_attention(local_patches, global_patches)

        # Apply transformer blocks
        for block in self.transformer.h:
            patches = block(patches)

        # Classification head
        x = patches.mean(dim=1)
        logits = self.mlp_head(x)

        # Reconstruction loss
        reconstructed = self.reconstruction_head(patches)
        aux_losses["reconstruction"] = F.mse_loss(
            reconstructed,
            img.unfold(2, self.config.local_patch_size, self.config.local_patch_size)
               .unfold(3, self.config.local_patch_size, self.config.local_patch_size)
               .flatten(1, 2),
        )

        if self.config.use_nViT == 1:
            sz = self.sz * (self.config.sz_init_value / self.config.sz_init_scaling)
            logits = sz * logits

        return logits, aux_losses

    @property
    def num_params(self) -> int:
        """Return number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())
