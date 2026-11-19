"""Frozen DINOv3 encoder, continuous-time latent predictor (the world model), template-warping mask head."""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[1]
DINO_REPO = Path.home() / ".cache/torch/hub/facebookresearch_dinov3_main"
DINO_W = PROJ / "weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FrozenEncoder(nn.Module):
    """uint8 (N, P, P) -> layer-normed patch tokens (N, (P/16)^2, 768)."""

    def __init__(self):
        super().__init__()
        self.net = torch.hub.load(str(DINO_REPO), "dinov3_vitb16", source="local", weights=str(DINO_W))
        self.net.eval().requires_grad_(False)
        self.register_buffer("mean", MEAN, persistent=False)
        self.register_buffer("std", STD, persistent=False)
        self.dim = 768

    def train(self, mode=True):  # stays in eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        x = x.float().div(255).unsqueeze(1).expand(-1, 3, -1, -1)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            z = self.net.forward_features((x - self.mean) / self.std)["x_norm_patchtokens"]
        return z.float()


class TimeEmbed(nn.Module):
    """Continuous time (seconds, any sign) -> d. Log-spaced frequencies cover 0.05-20 Hz."""

    def __init__(self, d, n_freq=32):
        super().__init__()
        self.register_buffer("freq", 2 * math.pi * torch.logspace(math.log10(0.05), math.log10(20), n_freq))
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, tau):
        a = tau[..., None] * self.freq
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class Predictor(nn.Module):
    """World model: tokens of K context frames at times tau_ctx -> tokens of a frame at tau_tgt.

    Joint space-time attention over (K + Q) x N tokens; query tokens carry only position + target time.
    Residual: the output is the last context frame's tokens plus a predicted change (zero at init).
    """

    def __init__(self, in_dim=768, d=384, depth=6, heads=6, n_tok=256, dropout=0.0):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, 1, n_tok, d))
        self.query = nn.Parameter(torch.zeros(1, 1, 1, d))
        self.temb = TimeEmbed(d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, in_dim)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.zeros_(self.out.weight), nn.init.zeros_(self.out.bias)

    def forward(self, z_ctx, tau_ctx, tau_tgt):
        """z_ctx (B,K,N,C), tau_ctx (B,K), tau_tgt (B,) or (B,Q) -> (B,Q,N,C)."""
        B, K, N, _ = z_ctx.shape
        if tau_tgt.dim() == 1:
            tau_tgt = tau_tgt[:, None]
        Q = tau_tgt.shape[1]
        x = self.inp(z_ctx) + self.pos + self.temb(tau_ctx)[:, :, None]
        q = (self.query + self.pos).expand(B, Q, N, -1) + self.temb(tau_tgt)[:, :, None]
        h = self.blocks(torch.cat([x, q], 1).flatten(1, 2))
        return z_ctx[:, -1:] + self.out(self.norm(h[:, K * N:])).view(B, Q, N, -1)


def _conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class MaskHead(nn.Module):
    """Predict the target mask in a frame from its tokens by warping the first-frame (template) mask.

    Output logits = logit(warped template) + residual. The displacement is predicted at 64x64 and
    upsampled; it is returned for smoothness regularisation.
    """

    def __init__(self, in_dim=768, d=256, n_side=16, out_px=256):
        super().__init__()
        self.n_side, self.out_px = n_side, out_px
        self.fuse = nn.Linear(3 * in_dim + 1, d)
        self.pos = nn.Parameter(torch.zeros(1, n_side * n_side, d))
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.mix = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.up = nn.Sequential(_conv(d, 128), nn.Upsample(scale_factor=2, mode="bilinear"), _conv(128, 64),
                                nn.Upsample(scale_factor=2, mode="bilinear"), _conv(64, 64))
        self.flow = nn.Conv2d(64, 2, 3, padding=1)
        self.res = nn.Sequential(_conv(64 + 1, 32), nn.Conv2d(32, 1, 3, padding=1))
        nn.init.zeros_(self.flow.weight), nn.init.zeros_(self.flow.bias)
        nn.init.zeros_(self.res[-1].weight), nn.init.zeros_(self.res[-1].bias)
        nn.init.trunc_normal_(self.pos, std=0.02)
        g = torch.linspace(-1, 1, out_px)
        self.register_buffer("grid", torch.stack(torch.meshgrid(g, g, indexing="xy"), -1)[None], persistent=False)

    def forward(self, z, z_ref, ref_mask):
        """z, z_ref (B,N,C); ref_mask (B,P,P) {0,1} -> logits (B,P,P), flow (B,2,64,64) in normalised units."""
        B, N, _ = z.shape
        m_tok = F.adaptive_avg_pool2d(ref_mask[:, None].float(), self.n_side).flatten(1)[..., None]
        h = self.mix(self.fuse(torch.cat([z, z_ref, z - z_ref, m_tok], -1)) + self.pos)
        f = self.up(h.transpose(1, 2).reshape(B, -1, self.n_side, self.n_side))
        flow = self.flow(f)
        flow_up = F.interpolate(flow, size=self.out_px, mode="bilinear", align_corners=True)
        warped = F.grid_sample(ref_mask[:, None].float(), self.grid + flow_up.permute(0, 2, 3, 1),
                               mode="bilinear", padding_mode="zeros", align_corners=True)
        f_up = F.interpolate(f, size=self.out_px, mode="bilinear", align_corners=False)
        logits = torch.logit(warped.clamp(1e-4, 1 - 1e-4)) + self.res(torch.cat([f_up, warped], 1))
        return logits[:, 0], flow


def seg_loss(logits, target):
    t = target.float()
    p = logits.sigmoid()
    inter = (p * t).flatten(1).sum(1)
    dice = 1 - (2 * inter + 1) / (p.flatten(1).sum(1) + t.flatten(1).sum(1) + 1)
    return F.binary_cross_entropy_with_logits(logits, t) + dice.mean()


def smooth_loss(flow):
    return (flow.diff(dim=-1) ** 2).mean() + (flow.diff(dim=-2) ** 2).mean()
