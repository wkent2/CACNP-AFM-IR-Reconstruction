#!/usr/bin/env python3
"""
conv_cnp_afmir_prob.py — Convolutional-Attentive Conditional Neural Process
                         (CACNP) for AFM-IR hyperspectral cube reconstruction
                         WITH calibrated predictive uncertainty for adaptive sampling.
"""

import os
import math
from statistics import NormalDist
from scipy.signal import find_peaks
import argparse
from typing import List, Optional, Sequence, Tuple

import numpy as np
import math as _math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anp_afmir import load_cube_library, load_cube
from cnp_afmir  import compute_metrics  # reuse metric helper


# =============================================================================
# Input normalization (shared by training dataset AND test/deployment)
# =============================================================================
NORM_MODES = ("cube_minmax", "context_minmax", "context_zscore")


def hex_seed_mask(H, Wsp, target_frac, rng):
    """Approximate hexagonal seed grid as a boolean (H, Wsp) mask.

    Adds points on a triangular lattice (rows offset by half a column spacing) 
    whose spacing is chosen so the number of points is approximately target_frac * H * Wsp. 
    The whole lattice is given a small random pixel offset so different cubes do 
    not all seed the exact same pixels, while remaining a regular hex grid 
    (matching how a hex grid is laid down on an arbitrary scene at acquisition time).

    Args:
        H, Wsp      : spatial dims (height, W_spatial)
        target_frac : desired fraction of pixels in the seed (e.g. 0.001 = 0.1%)
        rng         : numpy Generator for the lattice offset
    Returns:
        (H, Wsp) bool numpy array with ~target_frac*H*Wsp True entries.
    """

    n_pix = H * Wsp                                     # Determine number of spatial pixels
    n_target = max(1, int(round(target_frac * n_pix)))  # Determine number of target pixels
    # hex lattice: row spacing dy, column spacing dx, with dy = dx * sqrt(3)/2.
    # n_points ~ (H/dy) * (Wsp/dx) = H*Wsp / (dx*dy) = H*Wsp / (dx^2 * sqrt3/2).
    # solve for dx given the target count.

    dx = _math.sqrt((n_pix) / (n_target * (_math.sqrt(3) / 2.0))) # Solve for x spacing
    dx = max(dx, 1.5)                                             # Clip if too small
    dy = dx * _math.sqrt(3) / 2.0                                 # Solve for y spacing

    # random sub-pixel/pixel offset for the whole lattice (keeps it a hex grid)
    off_r = float(rng.uniform(0, dy))
    off_c = float(rng.uniform(0, dx))

    mask = np.zeros((H, Wsp), dtype=bool)                         # Initialize mask
    r = off_r
    row_idx = 0

    while r < H:
        # offset alternate rows by half a column spacing -> triangular (hexagonal) lattice
        c = off_c + (dx / 2.0 if (row_idx % 2) else 0.0)          # Determine row offset
        while c < Wsp:                                            # Iterate until row is complete
            ri = int(round(r)); ci = int(round(c))                # Get integer position
            if 0 <= ri < H and 0 <= ci < Wsp:                     # If in bounds, mask point
                mask[ri, ci] = True
            c += dx                                               # Move to next x position
        r += dy                                                   # Move to next y position
        row_idx += 1                                              # Move to next row
    # Ensure at least one point
    if not mask.any():
        mask[H // 2, Wsp // 2] = True                             # Mark center point
    return mask


def normalize_cube(cube_raw, obs_mask, mode="cube_minmax", eps=1e-8):
    """
    Normalize a hyperspectral cube for the model, using one of three schemes.

    Args:
        cube_raw : (H, Wsp, W) RAW (unnormalized) intensities. For cube_minmax,
                   may already be [0,1]-normalized (idempotent). For the context
                   modes it MUST be raw, since the statistics come from the
                   observed-pixel intensities.
        obs_mask : (H, Wsp) bool — the observed/context pixels. Required for the
                   context_* modes; ignored for cube_minmax.
        mode     : one of NORM_MODES:
                   - "cube_minmax"    : (x - cube.min) / (cube.max - cube.min).
                       Uses WHOLE-cube stats. Needs the full cube (not available
                       from sparse context alone) -> training/eval convenience.
                   - "context_minmax" : (x - ctx.min) / (ctx.max - ctx.min),
                       stats from OBSERVED pixels only. Matches deployment, where
                       only context is seen. Bounded-ish [0,1] -> pairs with
                       scaled_sigmoid output.
                   - "context_zscore" : (x - ctx.mean) / ctx.std, stats from
                       OBSERVED pixels only. Robust to baseline offset (e.g. a
                       nonzero mV floor) and not dominated by a single extreme
                       pixel. UNBOUNDED -> pair with --mean_activation none.
        eps      : numerical floor for the denominator.

    Returns:
        cube_norm : (H, Wsp, W) float32, normalized per `mode`.
        stats     : dict of the (a, b) such that raw = a + b * norm, so callers
                    (test/deploy) can DENORMALIZE predictions back to input units.
                    cube/context_minmax: raw = lo + (hi-lo)*norm
                    context_zscore     : raw = mean + std*norm
    """
    x = np.asarray(cube_raw, dtype=np.float64)

    # Standard min-max normalization: cube = cube-min/(max-min)
    if mode == "cube_minmax":
        lo = float(x.min()); hi = float(x.max())
        denom = (hi - lo) if (hi - lo) > eps else 1.0
        out = (x - lo) / denom
        stats = {"mode": mode, "offset": lo, "scale": denom}

    # Min-max, but using only context voxels
    elif mode == "context_minmax":
        if obs_mask is None or obs_mask.sum() == 0:
            raise ValueError("context_minmax requires a non-empty obs_mask")
        ctx_vals = x[np.asarray(obs_mask, dtype=bool)]      # (n_ctx, W) -> flat
        lo = float(ctx_vals.min()); hi = float(ctx_vals.max())
        denom = (hi - lo) if (hi - lo) > eps else 1.0
        out = (x - lo) / denom
        stats = {"mode": mode, "offset": lo, "scale": denom}

    # Normalize cube according to context z-score metric
    # This method is usefulf for avoiding outliers compressing signal
    elif mode == "context_zscore":
        if obs_mask is None or obs_mask.sum() == 0:
            raise ValueError("context_zscore requires a non-empty obs_mask")
        ctx_vals = x[np.asarray(obs_mask, dtype=bool)]
        mu = float(ctx_vals.mean()); sd = float(ctx_vals.std())
        sd = sd if sd > eps else 1.0
        out = (x - mu) / sd
        stats = {"mode": mode, "offset": mu, "scale": sd}

    # Error handling 
    else:
        raise ValueError(f"unknown norm mode '{mode}'; choose from {NORM_MODES}")

    return out.astype(np.float32), stats


def denormalize(norm_vals, stats):
    """Invert normalize_cube: raw = offset + scale * norm. For test/deployment
    to map predictions back into input (e.g. mV) units."""
    return stats["offset"] + stats["scale"] * np.asarray(norm_vals)


# =============================================================================
# 1.  Building blocks  (unchanged from conv_cnp_afmir.py)
# =============================================================================

def _odd(n: int) -> int:
    """Round up to the nearest odd integer (kernel sizes must be odd)."""
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def _smear_at_sigma(x: torch.Tensor, sigma: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Depthwise separable Gaussian blur of x at a (scalar tensor) bandwidth sigma.
    
    Smearining is necessary if not using NN-filled input to propagate context information. 

    """
    B, C, H, W = x.shape                                              # Get data shape 
    ks = kernel_size                                                  # Get kernel size
    r  = ks // 2                                                      # Get kernel radius
    coords = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)  # Initialize radial corrdinates
    k1d    = torch.exp(-0.5 * (coords / sigma) ** 2)                  # Compute kernel values
    k1d    = k1d / k1d.sum()                                          # Noramlize kernel values

    w_h = k1d.view(1, 1, 1, ks).expand(C, 1, 1, ks).contiguous()      # Smear data in final dimension
    x   = F.conv2d(x, w_h, padding=(0, r), groups=C)
    w_v = k1d.view(1, 1, ks, 1).expand(C, 1, ks, 1).contiguous()      # Smear data in penultimate dimension
    x   = F.conv2d(x, w_v, padding=(r, 0), groups=C)
    return x


class GaussianSmear(nn.Module):
    """
    Learnable single-scale isotropic Gaussian blur for the ConvCNP set convolution.

    Sets gaussian smear radius as a learnable parameter and uses 
    _smear_at_sigma (see above) to smear context input. 

    Useful if unsure of appropriate Gaussian smear radius. 
    """

    def __init__(self, sigma_init: float = 1.5, kernel_size: int = 9):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.log_sigma   = nn.Parameter(torch.tensor(math.log(float(sigma_init))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma).clamp(min=0.1)
        return _smear_at_sigma(x, sigma, self.kernel_size)


class MultiScaleSmear(nn.Module):
    """
    Multi-scale learnable Gaussian set-convolution. Similar to GaussianSmear (see 
    above), but now producing multiple smeared representations of context. 

    Applies S Gaussian blurs at S DIFFERENT learnable bandwidths to the SAME input
    (density + masked spectra), and returns the S blurred copies. Each scale's
    bandwidth is an independent learned parameter (log-sigma).

    Why multi-scale: a single bandwidth forces a tradeoff — wide kernels reach
    between sparse context points (good at low context) but over-smooth (bad at
    high context); narrow kernels stay sharp but cannot bridge gaps. Providing
    several scales (plus each scale's density channel) should let the downstream U-Net
    pick which reach to trust — effectively an adaptive bandwidth without an 
    adaptive kernel.

    Per-scale kernel sizes are sized to the *initial* sigma (>= ~6 sigma) so the
    widest Gaussian is not truncated; sizes are fixed at construction even though
    sigma is learned, so if a learned sigma grows far beyond its init the kernel
    could clip — initialise the top scale's sigma near the largest reach you want.

    forward(x) -> Tensor of shape (B, S * C, H, W): the S blurred copies
    concatenated along the channel axis, in scale order.

    In practice, optimizing this module and training the multiple kernel length 
    scales proved difficult and unhelpful. The model would struggle to decide
    which length scale to trust. Perhaps unsurprising, considering that there
    is no context fraction-dependent parameter. 
    """

    def __init__(self, sigmas_init, kernel_sizes=None, kernel_cap: int = 97):
        """Initializes learnable parameters and extracts initial length scales from inputs"""
        super().__init__()
        sigmas_init = [float(s) for s in sigmas_init] # Load initial sigmas 
        self.n_scales = len(sigmas_init)              # Get number of kernels
        # Set one learnable log-sigma per scale
        self.log_sigmas = nn.ParameterList(
            [nn.Parameter(torch.tensor(math.log(s))) for s in sigmas_init]
        )
        # Set per-scale kernel size: ~6 sigma + 1, odd, capped 
        if kernel_sizes is None:
            kernel_sizes = [min(kernel_cap, _odd(6.0 * s + 1)) for s in sigmas_init]
        self.kernel_sizes = [int(k) for k in kernel_sizes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the smearing operation"""
        outs = []
        for log_sigma, ks in zip(self.log_sigmas, self.kernel_sizes):
            sigma = torch.exp(log_sigma).clamp(min=0.1)
            outs.append(_smear_at_sigma(x, sigma, ks))
        return torch.cat(outs, dim=1)   # (B, S*C, H, W)

    def current_sigmas(self):
        """Returns current kernel length scales. Useful for logging during training."""
        return [float(torch.exp(ls).clamp(min=0.1).item()) for ls in self.log_sigmas]


def _num_groups(channels: int) -> int:
    """Largest GroupNorm group count that evenly divides channels."""
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """
    Basic convolutional layer building blocks for U-net. 

    Two 3x3 convs (GroupNorm -> GELU) with a residual shortcut.

    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Learns residuals on top of input."""
        return self.conv(x) + self.skip(x)


class ImageGuidance(nn.Module):
    """
    Turn structure image(s) into a full-resolution guidance tensor that
    is concatenated into the encoder input, so the image rides the U-Net skip
    connections to full resolution (the pansharpening 'detail injection' idea).

    Each image i = (plane_i, band_i). A shared per-pixel MLP embeds
    [intensity | learned band-embedding(band_i)] -> out_ch features; contributions
    are DEEP-SETS SUMMED over images, so an arbitrary number of images (in any
    order) produce a fixed-shape guidance tensor. Bands are on the fixed W-grid,
    so band identity is a learned nn.Embedding (not a continuous encoding).
    """
    def __init__(self, W: int, emb_dim: int = 16, out_ch: int = 16,
                 agg: str = "sum", max_images: int = 4, include_raw: bool = False):
        super().__init__()
        self.band_emb = nn.Embedding(W, emb_dim)             # Setup embedding
        self.out_ch   = out_ch                               # Extract output channels
        self.max_images = int(max_images)
        
        # include_raw: also hand the encoder the UNTRANSFORMED intensity plane in
        # each slot, alongside the learned 16-ch encoding, so the U-Net (and its
        # skip connections) see the full-resolution image pixels directly rather
        # than only an embedded result. Slot width becomes 1 + out_ch.
        # Only meaningful with agg="concat": summing raw planes would conflate the
        # images, which is the very thing concat exists to avoid.
        self.include_raw = bool(include_raw)

        # Check guidance arguments match up
        if self.include_raw and agg != "concat":
            raise ValueError("include_raw requires guidance_agg='concat' "
                             "(summing raw planes would conflate the images)")

        # agg: how the per-image contributions are pooled (deep-sets).
        #   "sum"  — Sum guidance images (tensors) together. Potential problem with
        #            this method is that the magnitude of the tensor grows with the 
        #            number of structure images. Ultimately, loses individual image
        #            features, since they are all combined to 1 input. 
        #   "mean" — Divides guidance by the number of VALID images, so the magnitude is
        #            N-invariant. Identical to "sum" when N == 1. Ultimately, loses 
        #            individual image features, since they are all combined to 1 input.           
        # "concat" — Each image keeps its OWN out_ch channels in a
        #            dedicated slot (max_images slots, padded slots zeroed), so the
        #            encoder sees every image at full resolution WITHOUT conflating
        #            them. Preserves cross-image structure the sum destroys, at the
        #            cost of a fixed max count and order-dependence (order = the
        #            variance rank, so the most-discriminative image is slot 0).
        #            Guidance width becomes max_images * out_ch.
        # sum/mean can be flipped on a loaded checkpoint; concat changes the input
        # width, so it must be chosen at construction (and matched when loading).
        self.agg = str(agg)
        self.f = nn.Sequential(
            nn.Conv2d(1 + emb_dim, out_ch, 1), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 1))

    @property
    def slot_width(self):
        """Returns the channels one image slot occupies: out_ch, plus 1 if the raw plane rides
        along."""
        return self.out_ch + (1 if self.include_raw else 0)

    @property
    def guidance_width(self):
        """Return the channels the guidance tensor contributes to the encoder input."""
        if self.agg == "concat":
            return self.slot_width * self.max_images
        return self.out_ch

    def _embed(self, plane, band):
        """ Embeds a guidance image image """
        emb  = self.band_emb(band)                               # (B, emb_dim)
        emap = emb.view(band.shape[0], -1, 1, 1).expand(-1, -1, plane.shape[-2],
                                                        plane.shape[-1])
        return self.f(torch.cat([plane, emap], dim=1))          # (B, out_ch, H, W)

    def forward(self, image_planes: torch.Tensor, image_bands: torch.Tensor,
                image_valid: Optional[torch.Tensor] = None):
        # image_planes: (B, n_img, H, W); image_bands: (B, n_img);
        # image_valid: (B, n_img) bool for multi-image padding (None => all valid)
        B, n_img, H, Wd = image_planes.shape                   # Get data shape
        if image_bands.dim() == 1:
            image_bands = image_bands.unsqueeze(1)

        if self.agg == "concat":
            # Concatenate embedded structure images
            # Each image -> its OWN out_ch slot; fixed max_images slots, padded
            # slots are zeros.
            slots = []
            for i in range(self.max_images):
                if i < n_img:
                    # Embed image
                    c = self._embed(image_planes[:, i:i+1], image_bands[:, i])
                    # Including raw image (you should), concatenate it
                    if self.include_raw:
                        # RAW intensity plane rides in the same slot, untransformed,
                        # at full resolution -> reaches the encoder (and the skips)
                        # without passing through f.
                        c = torch.cat([image_planes[:, i:i+1], c], dim=1)
                    if image_valid is not None:
                        c = c * image_valid[:, i].view(B, 1, 1, 1).to(c.dtype)
                else:
                    c = image_planes.new_zeros(B, self.slot_width, H, Wd)
                slots.append(c)
            # NOTE on padding: the data is z-scored, so 0 is a MEANINGFUL value (the
            # mean) — an all-zero raw plane would read as "a flat mid-valued image".
            # Validity stays unambiguous because a padded slot also has an all-zero
            # EMBEDDING block, which a real image never produces (the band embedding
            # alone is non-zero).
            return torch.cat(slots, dim=1) # Concatenate guidance (B, max_images*slot_width, H, W)

        # The code below runs if guidance is set to 'sum' or 'mean'
        G = image_planes.new_zeros(B, self.out_ch, H, Wd)        # Initialize guidance
        for i in range(n_img):
            plane = image_planes[:, i:i+1]                       # Get image (B,1,H,W)
            emb   = self.band_emb(image_bands[:, i])             # Embed images (B, emb_dim)
            emap  = emb.view(B, -1, 1, 1).expand(-1, -1, H, Wd)  # Expand embedding to image shape (B, emb_dim, H, W)
            contrib = self.f(torch.cat([plane, emap], dim=1))    # Concat image to embedding (B, out_ch, H, W)
            if image_valid is not None:                          # Zero padded images
                contrib = contrib * image_valid[:, i].view(B, 1, 1, 1).to(contrib.dtype)
            G = G + contrib                                      # Pool images
        if self.agg == "mean":
            # divide by the number of VALID images (per sample), avoids magnitude scaling 
            # with the number of structure images
            if image_valid is not None:
                n = image_valid.sum(dim=1).clamp(min=1).view(B, 1, 1, 1).to(G.dtype)
            else:
                n = torch.full((B, 1, 1, 1), float(n_img), device=G.device, dtype=G.dtype)
            G = G / n
        return G                                                 # (B, out_ch, H, W)


class FiLMGenerator(nn.Module):
    """
    Feature-wise Linear Modulation 

    Condition the encoder layers on which bands are imaged (a global gain on
    channels), via FiLM: e <- gamma * e + beta at each encoder stage. The
    conditioning z is a DEEP-SETS SUM of learned band embeddings over the imaged
    set, so it is count/order-agnostic. Initialised to identity (gamma=1, beta=0)
    so training starts as a no-op and learns to modulate.

    This module had little impact on model performance, so it is not used in the 
    published version of this methdology. 
    """
    def __init__(self, W: int, enc_channels: List[int], emb_dim: int = 16,
                 hidden: int = 128, agg: str = "sum"):
        super().__init__()
        self.agg = str(agg)          # same sum/mean choice as ImageGuidance
        self.band_emb = nn.Embedding(W, emb_dim)                 # Establish learnable band embeddings
        self.chs = list(enc_channels)                            # Get number of channels
        total = 2 * sum(self.chs)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, total))       # Define MLP for modifiying features
        nn.init.zeros_(self.mlp[-1].weight)                      # Identity at initialization
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, image_bands: torch.Tensor,
                image_valid: Optional[torch.Tensor] = None):
        if image_bands.dim() == 1:
            image_bands = image_bands.unsqueeze(1)
        emb = self.band_emb(image_bands)                        # Embed image bands (B, n_img, emb_dim)
        if image_valid is not None:                             # Mask padded images, if there
            emb = emb * image_valid.unsqueeze(-1).to(emb.dtype)
        z = emb.sum(dim=1)                                      # Sum embeded images (B, emb_dim) 
        if self.agg == "mean":                                  # Average embeddings, if doing so, to avoid
                                                                # structure image count magnitude dependence
            if image_valid is not None:
                n = image_valid.sum(dim=1).clamp(min=1).unsqueeze(-1).to(z.dtype)
            else:
                n = torch.full((z.shape[0], 1), float(image_bands.shape[1]),
                               device=z.device, dtype=z.dtype)
            z = z / n
        p = self.mlp(z)                                         # Determine modulation signal (B, 2*sum(chs))
        out, off = {}, 0
        for name, C in zip(["e0", "e1", "e2", "e3"], self.chs): # Modulate channels
            g = 1.0 + p[:, off:off+C];        off += C          # gamma = 1 + delta
            b =       p[:, off:off+C];        off += C          # beta
            out[name] = (g.view(-1, C, 1, 1), b.view(-1, C, 1, 1))
        return out


class UNet(nn.Module):
    """
    U-net core of CACNP: U-Net encoder-decoder with 3 levels of 2x spatial downsampling 
    and skip connections.  Pads spatial dims to a multiple of 8 (reflect) then crops back.
    For a square 128x128 grid the round-trips are exact and the bottleneck sees
    a global receptive field (128 -> 64 -> 32 -> 16).

    Uses ConvBlock defined above as the essential building block. 

    Cross attention with context pixles (keys and values) is applied to decoder features (queries). 
    """

    # valid attention stage names, coarse -> fine
    ATTN_STAGES = ("bottleneck", "dec2", "dec1", "dec0")

    def __init__(self, in_ch: int, enc_channels: List[int], dropout: float = 0.0,
                 attn_modules: Optional[dict] = None, dec0_ch: Optional[int] = None):
        super().__init__()
        C0, C1, C2, C3 = enc_channels

        self.dec0_ch = int(dec0_ch) if dec0_ch else C0 # Specify final decoder block 
                                                       # if different from entry
        self.enc0 = ConvBlock(in_ch, C0, dropout)       
        self.enc1 = ConvBlock(C0,    C1, dropout)
        self.enc2 = ConvBlock(C1,    C2, dropout)
        self.enc3 = ConvBlock(C2,    C3, dropout)   # bottleneck

        self.pool = nn.MaxPool2d(2)

        self.dec2 = ConvBlock(C3 + C2, C2, dropout)
        self.dec1 = ConvBlock(C2 + C1, C1, dropout)
        self.dec0 = ConvBlock(C1 + C0, self.dec0_ch, dropout)

        # cross-attention modules keyed by stage name (empty dict => no attention)
        self.attn = nn.ModuleDict(attn_modules or {})

    def _apply_attn(self, name, feat, ctx):
        """Apply cross attention between context spectra and U-net features"""
        if ctx is not None and name in self.attn:
            feat = feat + self.attn[name](feat, *ctx)   # residual (zero-init out)
        return feat

    def forward(self, x: torch.Tensor,
                ctx: Optional[tuple] = None,
                film: Optional[dict] = None) -> torch.Tensor:
        """
        Takes in the NN-filled context hyperspectral cube + distance map + embedded 
        structure images. So C = Wspec + 1 + 4*struct_embed

        ctx = (ctx_spec, ctx_pos, key_valid) gathered observed voxels, or None.
        film = {"e0":(gamma,beta), ...} FiLM params per encoder stage, or None.
        """
        _, _, H, W = x.shape

        pH = (8 - H % 8) % 8
        pW = (8 - W % 8) % 8
        if pH > 0 or pW > 0:
            x = F.pad(x, (0, pW, 0, pH), mode="reflect")

        def _film(feat, name):
            """This is not used in published version."""
            if film is not None and name in film:
                g, b = film[name]
                feat = g * feat + b
            return feat

        e0 = _film(self.enc0(x), "e0")                # (B,C0,H,W)
        e1 = _film(self.enc1(self.pool(e0)), "e1")    # (B,C1,H/2,W/2)
        e2 = _film(self.enc2(self.pool(e1)), "e2")    # (B,C2,H/4,W/4)
        e3 = _film(self.enc3(self.pool(e2)), "e3")    # bottleneck (B, C3, H/8, W/8)
        e3 = self._apply_attn("bottleneck", e3, ctx)

        up2 = F.interpolate(e3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2  = self.dec2(torch.cat([up2, e2], dim=1))  # (B, C2, H/4, W/4)
        d2  = self._apply_attn("dec2", d2, ctx)

        up1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1  = self.dec1(torch.cat([up1, e1], dim=1))  # (B, C1, H/2, W/2)
        d1  = self._apply_attn("dec1", d1, ctx)

        up0 = F.interpolate(d1, size=e0.shape[2:], mode="bilinear", align_corners=False)
        d0  = self.dec0(torch.cat([up0, e0], dim=1))  # (B, C0, H, W)
        d0  = self._apply_attn("dec0", d0, ctx)

        return d0[:, :, :H, :W]


# =============================================================================
# 2.  Probabilistic ConvCNP  
# =============================================================================

def _fourier_pos(coords: torch.Tensor, n_bands: int) -> torch.Tensor:
    """
    Sinusoidal Fourier positional encoding. 

    Input: coords : (..., 2) normalized to [0, 1]  ->  (..., 4 * n_bands)

    Gives the attention a multi-frequency notion of location so it
    can reason about proximity without a hard-coded distance bias. Accepts any
    leading dims (e.g. (N,2) query grid or (B,K,2) gathered context positions).
    """

    freqs = (2.0 ** torch.arange(n_bands, device=coords.device, dtype=coords.dtype)) * math.pi
    ang   = coords.unsqueeze(-1) * freqs                       # (..., 2, n_bands)
    enc   = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    return enc.reshape(*coords.shape[:-1], -1)                 # (..., 4 * n_bands)


@torch.no_grad()
def build_context_tokens(obs_mask: torch.Tensor,
                         ctx_spectra_grid: torch.Tensor):
    """
    Builds context tokens for attention. 

    Gathers the observed voxels into a padded key/value set (built once per
    forward and shared by every attention stage). 

    Returns:
        ctx_spec  : (B, K, W)  clean spectra at observed voxels, zero-padded.
        ctx_pos   : (B, K, 2)  normalized (y, x) coords of those voxels, padded.
        key_valid : (B, K) bool True for real observations, False for padding.
    where K = max observed count in the batch (dynamic; one .item() sync/forward).
    """
    B, W, H, Wd = ctx_spectra_grid.shape                       # Gets data shape
    N = H * Wd                                                 # Gets number of pixels 
    flat_mask = obs_mask.reshape(B, N)                         # (B, N) bool
    counts = flat_mask.sum(dim=1)                              # (B,)
    K = int(counts.max().clamp(min=1).item())                  # Number of context pixels

    # Get flat mask for pixels 
    vals, idx = flat_mask.to(ctx_spectra_grid.dtype).topk(K, dim=1)   # (B, K)
    key_valid = vals > 0.5                                     # (B, K)

    # Gather spectra at selected pixels
    spec_flat = ctx_spectra_grid.reshape(B, W, N)
    ctx_spec  = torch.gather(spec_flat, 2, idx.unsqueeze(1).expand(B, W, K))
    ctx_spec  = ctx_spec.permute(0, 2, 1).contiguous()        # (B, K, W)

    # Get coordinates for each context pixel
    ys = (torch.arange(H,  device=obs_mask.device, dtype=ctx_spectra_grid.dtype) + 0.5) / H
    xs = (torch.arange(Wd, device=obs_mask.device, dtype=ctx_spectra_grid.dtype) + 0.5) / Wd
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)       # (N, 2)
    ctx_pos = pos[idx]                                          # (B, K, 2)

    # zero padded rows (masked out anyway; keeps projections from seeing junk)
    ctx_spec = ctx_spec * key_valid.unsqueeze(-1).to(ctx_spec.dtype)
    return ctx_spec, ctx_pos, key_valid


# Neutral wavenumber coordinate for tokens/queries that do NOT sit at a single
# band (full-spectrum context voxels, and the spatial queries). Placed OUTSIDE
# [0,1] so its Fourier encoding is distinct from any real imaged band's nu in
# [0,1]. Image tokens carry a real nu in [0,1]; everything else carries this.
NEUTRAL_NU = -1.0


def build_image_tokens(image_planes: torch.Tensor,
                       image_bands:  torch.Tensor,
                       W: int,
                       n_subsample: int,
                       deterministic: bool,
                       image_valid: Optional[torch.Tensor] = None):
    """
    Turn full-field structure image(s) into attention tokens.

    Each imaged pixel becomes ONE token carrying its SCALAR intensity and its 3D
    position (x, y, nu_band). The intensity is embedded by a DEDICATED projection
    in the attention (img_k / img_v), sized so image tokens compete with the
    dense full-spectrum context values (they no longer share v_in / k_in, which
    made them ~10x weaker). Images are spatially smooth, so we subsample to
    n_subsample tokens per image (random while training for coverage; evenly
    strided at eval for reproducibility).

    Returns:
        img_val   : (B, n_img*M, 1) scalar intensities.
        img_pos   : (B, n_img*M, 3) normalized (y, x, nu) positions.
        img_valid : (B, n_img*M) bool, all True.

    This module did not improve performance in any meaninful way, so it is not used
    in the published model. 
    """
    B, n_img, H, Wd = image_planes.shape
    N = H * Wd
    dev, dt = image_planes.device, image_planes.dtype
    if image_bands.dim() == 1:
        image_bands = image_bands.unsqueeze(1)                    # (B, n_img)

    ys = (torch.arange(H,  device=dev, dtype=dt) + 0.5) / H
    xs = (torch.arange(Wd, device=dev, dtype=dt) + 0.5) / Wd
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pos2d = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1) # (N, 2)

    M = min(n_subsample, N)
    if M >= N:
        idx = torch.arange(N, device=dev)
    elif deterministic:
        idx = torch.linspace(0, N - 1, M, device=dev).round().long()
    else:
        idx = torch.randperm(N, device=dev)[:M]

    vals, poss = [], []
    for i in range(n_img):
        plane = image_planes[:, i].reshape(B, N)[:, idx]          # (B, M)
        band  = image_bands[:, i]                                 # (B,)
        nu    = (band.to(dt) / W).view(B, 1, 1).expand(B, M, 1)   # (B, M, 1)
        p3    = torch.cat([pos2d[idx].unsqueeze(0).expand(B, -1, -1), nu], dim=-1)
        vals.append(plane.unsqueeze(-1))                          # (B, M, 1)
        poss.append(p3)                                           # (B, M, 3)

    img_val   = torch.cat(vals, dim=1)                            # (B, n_img*M, 1)
    img_pos   = torch.cat(poss, dim=1)                            # (B, n_img*M, 3)
    if image_valid is not None:
        # multi_image padding: each image's M tokens inherit that image's
        # validity, so padded (zero) slots are masked out in the attention.
        # Token order is image-major (all M of image 0, then image 1, ...), which
        # this expand-then-reshape matches exactly.
        iv = image_valid.to(torch.bool).to(dev)                   # (B, n_img)
        img_valid = iv.unsqueeze(-1).expand(B, n_img, M).reshape(B, n_img * M)
    else:
        img_valid = torch.ones(B, img_val.shape[1], dtype=torch.bool, device=dev)
    return img_val, img_pos, img_valid


class CrossAttentionStage(nn.Module):
    """
    Residual cross-attention at one U-Net stage (bottleneck or a decoder level).

    Queries  : the stage's feature cells (B, C, Hf, Wf) -> (Hf*Wf) tokens, each
               carrying its feature + its cell-centre position. Query resolution =
               the stage resolution, so cost scales with Hf*Wf.
    Keys/Vals: the context pixels only (gathered, padded)
               keys = clean spectrum + position
               values = clean spectrum. 
               Padding is masked out. What gets routed is always a real, 
               unaveraged observed spectrum.

    Output is added redisually.
    """

    def __init__(self, c_feat: int, W: int, d_model: int = 256,
                 n_heads: int = 4, n_pos_bands: int = 8,
                 spectral_pos: bool = False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.c_feat        = c_feat                               # Channel features
        self.n_heads       = n_heads                              # Number of attention heads
        self.d_model       = d_model                              # Dimension of attn representation
        self.d_head        = d_model // n_heads                   # Dimension per head
        self.n_pos_bands   = n_pos_bands                          # Number of position bands
        self.spectral_pos  = spectral_pos                         # Whether or not spectral position
                                                                  # is used
        d_pos = (6 if spectral_pos else 4) * n_pos_bands

        self.q_in = nn.Linear(c_feat + d_pos, d_model)            # Queries
        self.k_in = nn.Linear(W + d_pos,      d_model)            # Keys
        self.v_in = nn.Linear(W,              d_model)            # Values

        # Used if using image tokens. (Not used in published model )
        if spectral_pos:
            img_e = 32
            self.img_k = nn.Linear(1 + d_pos, d_model)            # [intensity | pos3d]
            self.img_v = nn.Sequential(nn.Linear(2, img_e),       # [intensity, nu]
                                       nn.GELU(),
                                       nn.Linear(img_e, d_model))
        else:
            self.img_k = None
            self.img_v = None
        self.out  = nn.Linear(d_model,        c_feat)             # Output MLP
        nn.init.zeros_(self.out.weight)      
        nn.init.zeros_(self.out.bias) 
        self._pos_cache: dict = {}                                # Saves positions

    def _grid_pos(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """Gets grid positions"""
        key = (H, W, device, dtype)
        if key not in self._pos_cache:
            ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
            xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            self._pos_cache[key] = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)
        return self._pos_cache[key]                          # (H*W, 2)

    def forward(self, feat: torch.Tensor,
                ctx_spec: torch.Tensor,
                ctx_pos:  torch.Tensor,
                key_valid: torch.Tensor,
                img_bundle=None) -> torch.Tensor:
        B, C, Hf, Wf = feat.shape                                # Get U-net layer data shape
        dev, dt = feat.device, feat.dtype                        # Extract device and dtype
        Nq = Hf * Wf                                             # Determine number of queries

        # Construct Queries: stage features + cell-centre positions 
        qcoord = self._grid_pos(Hf, Wf, dev, dt)                 # Get query positions (Nq, 2)
        if self.spectral_pos:                                    # append neutral spectral
            nu = torch.full((qcoord.shape[0], 1), NEUTRAL_NU, device=dev, dtype=dt)
            qcoord = torch.cat([qcoord, nu], dim=-1)             # (Nq, 3)
        qpos = _fourier_pos(qcoord, self.n_pos_bands)            # Encode positions (Nq, d_pos)
        qf   = feat.permute(0, 2, 3, 1).reshape(B, Nq, C)        # Reshape queries
        q    = self.q_in(torch.cat([qf, qpos.unsqueeze(0).expand(B, -1, -1)], dim=-1)) # Run queries through MLP

        # Construct keys and values: gathered observed voxels (dense-spectrum projection)
        kpos = _fourier_pos(ctx_pos, self.n_pos_bands)           # Encode CTX positions (B, K, d_pos)
        k    = self.k_in(torch.cat([ctx_spec, kpos], dim=-1))    # Run keys through MLP
        v    = self.v_in(ctx_spec)                               # Run values through MLP
        valid = key_valid                                        # Extract valid keys

        # Construct image tokens (NOT USED IN PUBLISHED MODEL)
        if img_bundle is not None and self.img_k is not None:
            img_val, img_pos, img_valid = img_bundle             # (B,M,1),(B,M,3),(B,M)
            ipos = _fourier_pos(img_pos, self.n_pos_bands)       # Encode image positions (B, M, d_pos)
            ik   = self.img_k(torch.cat([img_val, ipos], dim=-1))# Run image keys through MLP (B,M,d_model)
            iv   = self.img_v(torch.cat([img_val, img_pos[..., 2:3]], dim=-1))  # Run image values through MLP (B,M,d_model)
            k     = torch.cat([k, ik], dim=1)                    # Concatenate ctx and image keys
            v     = torch.cat([v, iv], dim=1)                    # Concatenate ctx and image values
            valid = torch.cat([valid, img_valid], dim=1)         # Add new valid image keys
        K = k.shape[1]                                           # Get the number of keys

        # PERFORM ATTENTION
        def split(t):       # (B, N, d_model) -> (B, heads, N, d_head)
            """Splits data of shape d_model into d_head chunks"""
            return t.reshape(B, -1, self.n_heads, self.d_head).transpose(1, 2)


        qh, kh, vh = split(q), split(k), split(v)               # Split data into d_head chunks
        scale  = 1.0 / math.sqrt(self.d_head)                   # Normalize data
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale # Compute dot product of keys and queries 
                                                                # [RAW ATTENTION SCORES] (B,heads,Nq,K)
        vmask  = valid.reshape(B, 1, 1, K)                      # Establish attention mask
        scores = scores.masked_fill(~vmask, float("-inf"))      # Mask out padded attention scores
        attn   = torch.softmax(scores, dim=-1)                  # Apply softmax to scores
        out    = torch.matmul(attn, vh)                         # Compute attended data (B,heads,Nq,d_head)
        out = out.transpose(1, 2).reshape(B, Nq, self.d_model)  # Recombine model (d_head -> d_model)
        out = self.out(out)                                     # Run through MLP (d_model -> c_feat)
        return out.reshape(B, Hf, Wf, C).permute(0, 3, 1, 2)    # Reshape to U-net dimension (B, C, Hf, Wf)


@torch.no_grad()
def nn_fill_and_distance(obs_mask: torch.Tensor,
                        ctx_spectra_grid: torch.Tensor):
    """
    Compute the NN-filled hyperspectral cube from context and distance map. 

    Nearest-observation fill + distance transform (the first part of the encoder input).

    Args:
        obs_mask          : (B, H, W) bool   — True = observed.
        ctx_spectra_grid  : (B, Wv, H, W)    — spectrum at observed pixels, 0 elsewhere.

    Returns:
        filled : (B, Wv, H, W) — every pixel holds the CLEAN spectrum of its nearest
                                 observed pixel (Voronoi fill; no averaging/blending).
        dist   : (B, 1,  H, W) — Euclidean distance to that nearest observation,
                                 normalized by the image diagonal (so ~[0, 1]).

    Nearest-seed assignment uses the Jump Flooding Algorithm: O(log N) passes of
    9-neighbour propagation, vectorized over the batch on whatever device the
    tensors live on. 

    Method was validated exactly against scipy.ndimage.distance_transform_edt
    """
    B, H, Wd = obs_mask.shape                 # Get data shape
    dev = obs_mask.device                     # Move mask to device
    INF = torch.finfo(torch.float32).max      # Stand in for inf value

    # Get y and x coords in [0,1] range
    ys = torch.arange(H,  device=dev).view(1, H, 1).expand(B, H, Wd).float()
    xs = torch.arange(Wd, device=dev).view(1, 1, Wd).expand(B, H, Wd).float()

    # Initialize arrays for tracking closest ctx pixels in x and y coords, and
    # for monitoring distance. 
    seed    = obs_mask
    best_y  = torch.where(seed, ys, torch.zeros_like(ys))
    best_x  = torch.where(seed, xs, torch.zeros_like(xs))
    best_d2 = torch.where(seed, torch.zeros_like(ys), torch.full_like(ys, INF))

    # Determine maximum hop distance 
    K = 1
    while K < max(H, Wd):
        K <<= 1

    # Build list of hop sizes
    steps = []
    s = K // 2
    while s >= 1:
        steps.append(s); s //= 2
    steps = [K] + steps + [2, 1, 2, 1]

    # Create list of 8 NN directions
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if not (dy == 0 and dx == 0)]

    # Fill image 
    for stp in steps:
        for dy, dx in offsets:
            # Extract current belief
            cy   = torch.roll(best_y, shifts=(-dy * stp, -dx * stp), dims=(1, 2))
            cx   = torch.roll(best_x, shifts=(-dy * stp, -dx * stp), dims=(1, 2))
            chas = torch.roll(best_d2 < INF, shifts=(-dy * stp, -dx * stp), dims=(1, 2))
            # Ensure that nieghbors do not wrap around image
            in_b = ((ys + dy * stp) >= 0) & ((ys + dy * stp) < H) & \
                   ((xs + dx * stp) >= 0) & ((xs + dx * stp) < Wd)
            valid = in_b & chas
            # Compute distance
            d2 = (ys - cy) ** 2 + (xs - cx) ** 2
            d2 = torch.where(valid, d2, torch.full_like(d2, INF))
            # Update best answer
            take = d2 < best_d2
            best_d2 = torch.where(take, d2, best_d2)
            best_y  = torch.where(take, cy, best_y)
            best_x  = torch.where(take, cx, best_x)

    # Finalize NN idx
    idx  = best_y.long() * Wd + best_x.long()                 # (B, H, W)
    # Gather ctx spectra
    Wv   = ctx_spectra_grid.shape[1]
    flat = ctx_spectra_grid.reshape(B, Wv, H * Wd)
    idxf = idx.reshape(B, 1, H * Wd).expand(B, Wv, H * Wd)
    filled = torch.gather(flat, 2, idxf).reshape(B, Wv, H, Wd)

    # Normalize distance
    dist = torch.sqrt(best_d2.clamp_min(0.0)).unsqueeze(1) / math.sqrt(H * H + Wd * Wd)

    # Handle cubes with zero obersved pixels: undefined fill -> zeros, max distance.
    no_seed = ~obs_mask.any(dim=2).any(dim=1)                 # (B,)
    if no_seed.any():
        filled[no_seed] = 0.0
        dist[no_seed]   = 1.0
    return filled, dist


class HeadCrossAttention(nn.Module):
    """

    Cross attention for sharpening final spectral predictions at output head. 

    Full-resolution spectral sharpening: content-only cross-attention that
    operates on the reconstructed spectrum at the output.

    The query is the head's reconstructed spectrum at each pixel; keys and values
    are the OBSERVED context spectra. Each reconstructed pixel attends to the
    measured spectra it most resembles (by spectral CONTENT, no positional
    encoding, so no nn-fill spatial bias) and pulls their sharp peak structure.
    The attended result is mapped back to full W-band width and ADDED as a
    residual to the reconstructed spectrum, so it sharpens the head output.

    SDPA is used so the (Nq x K) score matrix is never materialized (Nq = H*W is
    up to 16384 at native resolution). This avoids model memory and computation
    requirements exploding. 

    forward
    -------
    spec      : (B, W, H, Wd) float. Reconstructed mean spectrum (the head output).
    ctx_spec  : (B, K, W)     float. Clean observed spectra (keys/values), padded.
    key_valid : (B, K)        bool.  True for real observations, False for padding.
    returns   : (B, W, H, Wd) float. spec + sharpening residual (full resolution).
    """
    def __init__(self, W: int, d_model: int = 320, n_heads: int = 4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"head_attn_dim ({d_model}) must be divisible by "
                             f"head_attn_heads ({n_heads})")
        self.W       = W                    # Spectral dimension length
        self.n_heads = n_heads              # Number of attention heads
        self.d_head  = d_model // n_heads   # Dimension of each head
        self.d_model = d_model              # Dimension of attention
        self.q_in = nn.Linear(W, d_model)   # reconstructed spectrum -> query (no position)
        self.k_in = nn.Linear(W, d_model)   # observed spectrum      -> key   (no position)
        self.v_in = nn.Linear(W, d_model)   # observed spectrum      -> value
        self.out  = nn.Linear(d_model, W)   # attended -> full-W residual on the spectrum
        nn.init.zeros_(self.out.weight)     # identity at init (residual)
        nn.init.zeros_(self.out.bias)

    def forward(self, spec, ctx_spec, key_valid):
        B, W, H, Wd = spec.shape                                    # Get data shape
        Nq = H * Wd                                                 # Number of queries (reconstructed pixels) 
        K  = ctx_spec.shape[1]                                      # Number of context tokens
        q = self.q_in(spec.permute(0, 2, 3, 1).reshape(B, Nq, W))   # MLP for queries (B, Nq, d_model)
        k = self.k_in(ctx_spec)                                     # MLP for keys (B, K,  d_model)
        v = self.v_in(ctx_spec)                                     # MLP for values (B, K,  d_model)
        qh = q.reshape(B, Nq, self.n_heads, self.d_head).transpose(1, 2)
        kh = k.reshape(B, K,  self.n_heads, self.d_head).transpose(1, 2)
        vh = v.reshape(B, K,  self.n_heads, self.d_head).transpose(1, 2)
        attn_mask = key_valid.view(B, 1, 1, K)                      # Mask attention values
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask)  # Perform attention (B,h,Nq,d_head)
        o = o.transpose(1, 2).reshape(B, Nq, self.d_model)          # (B, Nq, d_model)
        o = self.out(o)                                             # (B, Nq, W)  full-band residual
        o = o.permute(0, 2, 1).reshape(B, W, H, Wd)                 # (B, W, H, Wd)
        return spec + o                                             # Add residual to recon spectrum


class CACNP(nn.Module):
    """
    Convolutional-Attentive Conditional Neural Process.

    Encoder: density channel + masked spectra +structure images -> NN-filled + embedded structure images -> 
     -> 1x1 spectral encoder -> U-Net.

    Decoder: two 3x3 conv emit mu and sigma heads. Mu head recevies attention at the very end.

    forward() returns (mu, sigma), each (B, W, H, W_sp).
    """

    def __init__(
        self,
        W:             int,                            # Spectral dimension size
        base_channels: int       = 64,                 # Base U-net channels
        enc_channels:  List[int] = None,               # Spectral encoder channels
        kernel_sigma:  float     = 1.5,                # Kernel sigma (if using Gaussian smear)
        kernel_size:   int       = 9,                  # Kernel size (if using Gaussian smear)
        dropout:       float     = 0.0,                # Dropout frac for U-net layers
        sigma_min:     float     = 1e-3,               # Floor on sigma head predictions
        norm_mask:     bool      = True,               # Normalization Mask
        scales:        Optional[List[float]] = None,   # Multi-scale bandwidths for Gaussian smear; None=single-scale
        spectral_mlp:  bool      = False,              # Two-layer 1x1 MLP encoder
        mlp_hidden:    int       = 256,                # MLP hidden dimensions
        norm_smear:    bool      = False,              # Soft per-scale density normalization
        norm_eps:      float     = 0.05,               # Eps fraction of per-scale kernel peak
        mean_activation: str     = "sigmoid",          # Activation on output [sigmoid|scaled_sigmoid|linear|none]
        sig_margin:    float     = 0.05,               # Margin for scaled_sigmoid
        input_mode:    str       = "smear",            # How to treat context spectra ["smear" | "nn_fill"]
        image_bands:   Optional[List[int]] = None,     # Fixed full-field image band indices
        dynamic_image_band: bool = False,              # Per-sample band chosen at runtime
        image_tokens:  bool      = False,              # Route image(s) as 3D attention tokens?
        image_subsample: int     = 1024,               # Number of image tokens per image (subsampled)
        dec0_channels: Optional[int] = None,           # Width of final decoder block (spectral latent)
        guidance_agg:  str       = "sum",              # How to treat structure image guidance ["sum" | "mean" | "concat"] 
        max_images:    int       = 4,                  # Maximum slots for guidance_agg="concat"
        guidance_raw:  bool      = False,              # Also pass RAW image planes?
        decoder_skip:  str       = "none",             # Add decoder skip? [none | spectral_in | guidance]
        image_guidance: bool     = False,              # Piece A: full-res guidance tensor -> encoder
        film:          bool      = False,              # Piece B: FiLM-condition encoder on imaged bands
        multi_image:   bool      = False,              # Multiple structure images? ataset supplies 1..N variance-selected bands
        guidance_ch:   int       = 16,                 # Guidance tensor channels 
        band_emb_dim:  int       = 16,                 # Learned band-embedding dim (embedding dimension)
        use_attention: bool      = False,              # Do bottleneck cross-attention? 
        attn_stages:   Optional[List[str]] = None,     # Which U-Net stages get attention?
        attn_heads:    int       = 4,                  # How many attention heads?
        attn_dim:      int       = 256,                # d_model of the attention
        attn_pos_bands: int      = 8,                  # How many dimensions for Fourier positional embedding?
        head_kernel:   int       = 3,                  # Spatial kernel of the output heads (1 = per-pixel; 3 = 3x3 receptive field)
        head_attn:     bool      = False,              # Do attention at native head resolution?
        head_attn_dim: int       = 320,                # d_model of the head attention (>= W avoids a value bottleneck)
        head_attn_heads: int     = 4,                  # How many heads for the head attention?
    ):
        super().__init__()
        self.W         = W                             # Spectral dimension size
        self.sigma_min = sigma_min                     # Minimum sigma 
        self.norm_mask = norm_mask                     # Normalize mask
        self.norm_smear = norm_smear                   # Normalize smear
        self.norm_eps   = norm_eps                     # Norm eps
        self.mean_activation = mean_activation         # Mean activation
        self.sig_margin = sig_margin                   # Margin for scaled sigmoid
        self.input_mode = input_mode                   # Gaussian smear or NN-filled

        # Specifying image bands (static)
        if image_bands is not None:
            image_bands = list(image_bands)
            for b in image_bands:
                if not (0 <= b < W):
                    raise ValueError(f"image_band {b} out of range [0,{W})")
            if input_mode != "nn_fill":
                raise NotImplementedError("image_bands currently requires input_mode='nn_fill'")
        # If using DYNAMIC image bands
        if dynamic_image_band:
            if input_mode != "nn_fill":
                raise NotImplementedError("dynamic_image_band requires input_mode='nn_fill'")
            if image_bands is not None:
                raise ValueError("use either fixed image_bands or dynamic_image_band, not both")
        
        # Below lines check how structure images are used...
        self.image_tokens    = bool(image_tokens)     # Run structure images through attn?
        self.image_subsample = int(image_subsample)   # How much to subsample attn images
        self.image_guidance  = bool(image_guidance)   # Using image guidance?
        self.film            = bool(film)             # Using FiLM guidance?
        self.multi_image     = bool(multi_image)      # Multiple structure images?

        # Checking compatible inputs
        if image_tokens:
            if input_mode != "nn_fill":
                raise NotImplementedError("image_tokens requires input_mode='nn_fill'")
            if image_bands is None and not dynamic_image_band and not multi_image:
                raise ValueError("image_tokens needs image_bands, dynamic_image_band, "
                                 "or multi_image to select which band(s) to image")
        if image_guidance or film:
            if input_mode != "nn_fill":
                raise NotImplementedError("image_guidance/film require input_mode='nn_fill'")
            if image_bands is None and not dynamic_image_band and not multi_image:
                raise ValueError("image_guidance/film need image_bands, dynamic_image_band, "
                                 "or multi_image")
        
        self.image_bands = image_bands                             # If using static image bands, what are they?
        self.dynamic_image_band = bool(dynamic_image_band)         # Are bands going to be selected dynamically?
        
        self.multi_scale = scales is not None and len(scales) > 1  # Are multiple smearing scales being used? (only matters if smearing input)

        # Check that encoder channels are specified and handles it if not
        if enc_channels is None:
            enc_channels = [base_channels,     base_channels * 2,
                            base_channels * 4, base_channels * 8]

        # NN-FILL or GAUSSIAN SMEAR? Determine appropriate input channel size. 
        if input_mode == "nn_fill":
            self.smear      = None   # Smear not needed
            self.multi_scale = False # Smear scales not needed
            self.n_scales   = 1
            self.register_buffer("density_peak", torch.zeros(1), persistent=False)
            
            # Determine input size depending on how density is included
            # Older verison of method fed in 2W channels to show which bands 
            # had structure images. 
            use_image_conv = ((image_bands is not None) or dynamic_image_band) \
                             and not image_tokens and not image_guidance
            in_ch = (2 * W) if use_image_conv else (W + 1) 

            # If image guidance, add extra channels
            if image_guidance:                     
                slot_w = guidance_ch + (1 if guidance_raw else 0)
                g_mult = max_images if guidance_agg == "concat" else 1
                in_ch += (slot_w if guidance_agg == "concat" else guidance_ch) * g_mult
        else:
            # Smearing input
            if self.multi_scale:
                # Use multiple scales
                self.smear = MultiScaleSmear(sigmas_init=scales)  # Use multiscale smear
                self.n_scales = len(scales)                       # How many scales?
                sigmas_for_peak = [float(s) for s in scales]      # What are the initial scales?
            else:
                # Use a single scale
                single_sigma = scales[0] if scales else kernel_sigma 
                self.smear = GaussianSmear(sigma_init=single_sigma, kernel_size=kernel_size)
                self.n_scales = 1
                sigmas_for_peak = [float(single_sigma)]

            # Per-scale density peak. What is the maximum peak value in each scale?
            # This ended up not being very valuable when implemented. 
            peaks = []
            for s in sigmas_for_peak:
                c1d = 1.0 / (math.sqrt(2.0 * math.pi) * max(s, 0.1))
                peaks.append(float(c1d * c1d))
            # Registered as a buffer so it moves with .to(device) and is saved.
            self.register_buffer("density_peak", torch.tensor(peaks), persistent=False)

            # What is the size of the spectral encoder
            in_ch = self.n_scales * (W + 1)

        # What is the spectral encoder arch?
        # Each option uses GroupNorm and GELU activations
        if spectral_mlp:
            # Two-layer 1x1 MLP: in_ch -> mlp_hidden -> base_channels
            self.spectral_in = nn.Sequential(
                nn.Conv2d(in_ch, mlp_hidden, 1, bias=False),
                nn.GroupNorm(_num_groups(mlp_hidden), mlp_hidden),
                nn.GELU(),
                nn.Conv2d(mlp_hidden, base_channels, 1, bias=False),
                nn.GroupNorm(_num_groups(base_channels), base_channels),
                nn.GELU(),
            )
        else:
            # Single-layer 1x1 MLP
            self.spectral_in = nn.Sequential(
                nn.Conv2d(in_ch, base_channels, 1, bias=False),
                nn.GroupNorm(_num_groups(base_channels), base_channels),
                nn.GELU(),
            )

        # Which U-net stages get cross attention with spectral context?
        if attn_stages is None:
            attn_stages = ["bottleneck"] if use_attention else []
        attn_stages = list(attn_stages) 
        for s in attn_stages:
            if s not in UNet.ATTN_STAGES:
                raise ValueError(f"attn_stage '{s}' not in {UNet.ATTN_STAGES}")
        self.use_attention = len(attn_stages) > 0
        self.attn_stages   = attn_stages
        if self.image_tokens and not self.use_attention:
            raise ValueError("image_tokens requires attention stages (--attn_stages) "
                             "— the image is routed THROUGH attention.")

        # Channel width at each stage's feature map (queries live here)
        dec0_ch = int(dec0_channels) if dec0_channels else enc_channels[0]
        self.dec0_ch = dec0_ch
        stage_ch = {
            "bottleneck": enc_channels[3],
            "dec2":       enc_channels[2],
            "dec1":       enc_channels[1],
            "dec0":       dec0_ch,
        }

        # Instantiate and compile attention modules
        attn_modules = {
            s: CrossAttentionStage(c_feat=stage_ch[s], W=W, d_model=attn_dim,
                                   n_heads=attn_heads, n_pos_bands=attn_pos_bands,
                                   spectral_pos=self.image_tokens)
            for s in attn_stages
        }

        # Instantiate U-Net
        self.unet = UNet(base_channels, enc_channels, dropout=dropout,
                         attn_modules=attn_modules, dec0_ch=dec0_ch)

        # Instantiate guidance or FiLM if using
        self.guidance = (ImageGuidance(W, band_emb_dim, guidance_ch, agg=guidance_agg,
                                       max_images=max_images, include_raw=guidance_raw)
                         if image_guidance else None)
        self.film_gen = (FiLMGenerator(W, enc_channels, band_emb_dim, agg=guidance_agg)
                         if film else None)

        # What kind of data is allowed to skip within the network?
        self.decoder_skip = str(decoder_skip)
        if self.decoder_skip == "spectral_in":
            skip_ch = base_channels
        elif self.decoder_skip == "guidance":
            if not image_guidance:
                raise ValueError("decoder_skip='guidance' requires --image_guidance")
            skip_ch = guidance_ch * (max_images if guidance_agg == "concat" else 1) \
                      + (max_images if (guidance_agg == "concat" and guidance_raw) else 0)
        elif self.decoder_skip == "none":
            skip_ch = 0
        else:
            raise ValueError(f"unknown decoder_skip {decoder_skip!r}")

        # What is the skip channel
        self.skip_ch = skip_ch
        head_in = dec0_ch + skip_ch

        # What are the output heads' kernel sizes?
        self.head_kernel = int(head_kernel)
        if self.head_kernel < 1 or self.head_kernel % 2 == 0:
            raise ValueError(f"head_kernel must be a positive odd int, got {head_kernel}")
        _pad = self.head_kernel // 2
        _pad_mode = "reflect" if _pad > 0 else "zeros"

        # Make output heads
        def _make_head():
            if spectral_mlp:
                return nn.Sequential(
                    nn.Conv2d(head_in, mlp_hidden, self.head_kernel, bias=False,
                              padding=_pad, padding_mode=_pad_mode),
                    nn.GroupNorm(_num_groups(mlp_hidden), mlp_hidden),
                    nn.GELU(),
                    nn.Conv2d(mlp_hidden, W, self.head_kernel,
                              padding=_pad, padding_mode=_pad_mode),
                )
            return nn.Conv2d(head_in, W, self.head_kernel,
                             padding=_pad, padding_mode=_pad_mode)

        # Instantiate output heads 
        self.mean_decoder  = _make_head()   # -> mean_raw  (then mean activation)
        self.sigma_decoder = _make_head()   # -> scale_raw (then softplus+floor)

        # Instantiate output head cross_attention
        self.head_attn = (HeadCrossAttention(W, head_attn_dim, head_attn_heads)
                          if head_attn else None)

    def forward(
        self,
        obs_mask:         torch.Tensor,   # (B, H, W_sp) bool — True = observed
        ctx_spectra_grid: torch.Tensor,   # (B, W, H, W_sp) — spectrum at observed pixels, 0 elsewhere
        image_planes:     Optional[torch.Tensor] = None,  # (B, n_img, H, W_sp) imaged planes
        image_bands:      Optional[torch.Tensor] = None,  # (B,) or (B,n_img) band index
        image_valid:      Optional[torch.Tensor] = None,  # (B, n_img) bool, multi-image padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._forward(obs_mask, ctx_spectra_grid, image_planes, image_bands, image_valid)

    def _bands_tensor(self, image_bands, B, device):
        """Imaged band index/indices as (B, n_img). Multi-image passes a 2D
        (B, n_img) tensor already; dynamic passes (B,); fixed uses self.image_bands."""
        if image_bands is not None and torch.is_tensor(image_bands) and image_bands.dim() == 2:
            return image_bands.to(device).long()
        if self.dynamic_image_band:
            return image_bands.to(device).long().view(B, -1)
        t = torch.as_tensor(self.image_bands, device=device).long()
        return t.view(1, -1).expand(B, -1)

    def _add_image_tokens(self, ctx, image_planes, image_bands, image_valid=None):
        """
        Build the image-token bundle and attach it to the attention context.
        Returns (ctx_spec, ctx_pos3d, key_valid, img_bundle) where img_bundle =
        (img_val (B,M,1), img_pos (B,M,3), img_valid (B,M)). Context positions
        gain a neutral nu; image tokens keep their real nu. Image tokens are NOT
        merged into ctx_spec — they get their own projection in the attention.
        """
        if image_planes is None:
            raise ValueError("image_tokens mode: image_planes must be passed to forward")
        ctx_spec, ctx_pos, key_valid = ctx
        B = ctx_spec.shape[0]
        dev, dt = ctx_spec.device, ctx_spec.dtype

        if self.multi_image or self.dynamic_image_band:
            # Runtime bands: multi_image supplies (B, max_images) variance-selected
            # bands (padded, masked via image_valid); dynamic supplies (B, 1).
            bands_t = image_bands.to(dev).long().view(B, -1)
        else:
            bands_t = torch.as_tensor(self.image_bands, device=dev).long()
            bands_t = bands_t.view(1, -1).expand(B, -1)               # (B, n_img)

        img_bundle = build_image_tokens(
            image_planes, bands_t, self.W, self.image_subsample,
            deterministic=not self.training, image_valid=image_valid)

        # Context positions -> 3D with neutral nu (so d_pos matches image tokens)
        nu = torch.full((B, ctx_pos.shape[1], 1), NEUTRAL_NU, device=dev, dtype=dt)
        ctx_pos3d = torch.cat([ctx_pos, nu], dim=-1)                  # (B, K, 3)
        return (ctx_spec, ctx_pos3d, key_valid, img_bundle)

    def _forward(
        self,
        obs_mask:         torch.Tensor,
        ctx_spectra_grid: torch.Tensor,
        image_planes:     Optional[torch.Tensor] = None,
        image_bands:      Optional[torch.Tensor] = None,
        image_valid:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.input_mode == "nn_fill":
            # Nearest-observation fill + distance map. The distance map i
            #s an explicit "how far from data am I" signal.
            filled, dist = nn_fill_and_distance(obs_mask, ctx_spectra_grid)
            G = None                                   # kept for the decoder skip
            if self.image_guidance:
                # Plain nn_fill (W+1) + full-res guidance tensor from the
                # image(s), concatenated so the image rides the U-Net skips.
                rep = torch.cat([filled, dist], dim=1)            # Concatenate distance map(B, W+1, H, W_sp)
                bands_t = self._bands_tensor(image_bands, filled.shape[0], filled.device) # Get band tensor
                G = self.guidance(image_planes, bands_t, image_valid)  # (B, guidance_ch, H, W)
                rep = torch.cat([rep, G], dim=1)                  # Concatenate guidance (B, W+1+g, H, W)
            elif self.image_tokens:
                # Image is routed through 3D attention, NOT the conv input: the
                # encoder sees plain nn_fill (W+1); image tokens are added to the
                # attention key set below.
                rep = torch.cat([filled, dist], dim=1)            # (B, W+1, H, W_sp)
            elif self.dynamic_image_band:
                # One imaged band PER SAMPLE, chosen at runtime; overwrite that
                # sample's value + coverage planes in place (fixed 2W width).
                if image_planes is None or image_bands is None:
                    raise ValueError("dynamic_image_band: pass image_planes (B,1,H,W) "
                                     "and image_bands (B,) to forward")
                B, Wv, H, Wd = filled.shape                              # Get data shape
                value    = filled.clone()                                # Make a copy
                coverage = dist.expand(B, Wv, H, Wd).clone()             # Change shape
                bidx = torch.arange(B, device=filled.device)             # Get batch indexes
                value[bidx, image_bands]    = image_planes[:, 0, :, :]   # Extract structure image values (B,H,W)
                coverage[bidx, image_bands] = 0.0
                rep = torch.cat([value, coverage], dim=1)                # Concatenate input (B, 2W, H, W)
            elif self.image_bands is not None:
                if image_planes is None:
                    raise ValueError("image_bands set but no image_planes passed to forward")
                B, Wv, H, Wd = filled.shape                              # Get data shape
                # Value tensor: NN-filled spectra, imaged bands overwritten by the
                # measured plane (that band becomes truth everywhere).
                value = filled.clone()
                # Coverage tensor (per band): spatial distance-to-nearest-spectrum
                # for every band, then imaged bands set to 0 (observed everywhere).
                coverage = dist.expand(B, Wv, H, Wd).clone()
                for i, b in enumerate(self.image_bands):
                    value[:, b, :, :]    = image_planes[:, i, :, :]      # Get image band value
                    coverage[:, b, :, :] = 0.0                           # Get coverage 
                rep = torch.cat([value, coverage], dim=1)                # Concatenate coverage (B, 2W, H, W)
            else:
                rep = torch.cat([filled, dist], dim=1)                   # Simple concatenate distance(B, W+1, H, W_sp)
        else:
            # Build set-convolution input: density channel + masked spectra
            density = obs_mask.float().unsqueeze(1)                    # (B, 1, H, W_sp)
            inp     = torch.cat([density, ctx_spectra_grid], dim=1)    # (B, W+1, H, W_sp)
            C = inp.shape[1]                                           # W+1

            # Smear. Single-scale -> (B, W+1, H, W). Multi-scale -> (B, S*(W+1), H, W),
            # blurred copies concatenated in scale order.
            inp_blur = self.smear(inp)

            # Per scale, split into [density_blur | func_blur] and optionally normalise
            # the signal by the smeared density. Three modes:
            #   norm_smear (recommended): SOFT per-scale division func/(density+eps_s)
            #     with eps_s scaled to the scale's kernel-peak density. This cancels
            #     the context-fraction-dependent magnitude drift (the cause of the
            #     light->dark inversion at observed pixels) AND degrades smoothly in
            #     the low-density limit -> no square artifacts.
            #   norm_mask (legacy hard division, eps=1e-6): cancels drift but the
            #     near-0/0 limit at footprint edges produces the square artifacts.
            #   neither: raw func_blur (un-normalized) -> scale/context drift.
            reps = []
            for s in range(self.n_scales):
                block        = inp_blur[:, s * C:(s + 1) * C]
                density_blur = block[:, :1]
                func_blur    = block[:, 1:]

                if self.norm_smear:
                    eps_s = self.norm_eps * self.density_peak[s]
                    func  = func_blur / (density_blur + eps_s)
                elif self.norm_mask:
                    func = func_blur / (density_blur + 1e-6)
                else:
                    func = func_blur

                reps.append(density_blur)
                reps.append(func)
            rep = torch.cat(reps, dim=1)            # (B, S*(W+1), H, W_sp)

        h_in = self.spectral_in(rep)                # Embed spectra
        h   = h_in
        ctx = None

        # Build context attention tokens
        if self.use_attention or self.head_attn is not None:
            ctx = build_context_tokens(obs_mask, ctx_spectra_grid)
            if self.image_tokens:
                ctx = self._add_image_tokens(ctx, image_planes, image_bands, image_valid)
        film = None
        # If using FiLM, construct guidance tensors
        if self.film:
            bands_t = self._bands_tensor(image_bands, h.shape[0], h.device)
            film = self.film_gen(bands_t, image_valid)
        
        # Run embedded input through U-Net
        h   = self.unet(h, ctx=ctx, film=film)
        
        # If doing decoder skip, figure what to pass
        if self.decoder_skip == "spectral_in":
            h = torch.cat([h, h_in], dim=1)
        elif self.decoder_skip == "guidance":
            if G is None:
                raise RuntimeError("decoder_skip='guidance' but no guidance tensor "
                                   "was built (is --image_guidance set?)")
            h = torch.cat([h, G], dim=1)
        
        # Decode mean spectrum (output head main)
        mean_raw  = self.mean_decoder(h)                           # (B, W, H, W_sp)
        
        # If doing attention on output, do it here
        if self.head_attn is not None and ctx is not None:
            mean_raw = self.head_attn(mean_raw, ctx[0], ctx[2])
        
        # Sigma head 
        scale_raw = self.sigma_decoder(h)                          # (B, W, H, W_sp)

        # Output mean activation (selectable). Variants differ in whether they can
        # reach the data rails (0 and 1) and in their gradient behaviour there:
        #   sigmoid        : smooth, bounded (0,1), SATURATES -> can't reach 0 or 1
        #                    without infinite pre-activation (suspected range cap).
        #   scaled_sigmoid : (1+2m)*sigmoid - m -> CAN reach 0 and 1 at finite
        #                    pre-activation; smooth everywhere. m = sig_margin.
        #   linear         : hard clamp to [0,1] -> reaches rails exactly, but zero
        #                    gradient outside [0,1] (the original clamp behaviour).
        #   none           : identity, UNBOUNDED -> full gradient everywhere, may
        #                    predict <0 or >1 (clamp only at inference for display).
        act = self.mean_activation
        if act == "sigmoid":
            mu = torch.sigmoid(mean_raw)
        elif act == "scaled_sigmoid":
            m  = self.sig_margin
            mu = (1.0 + 2.0 * m) * torch.sigmoid(mean_raw) - m
        elif act == "linear":
            mu = mean_raw.clamp(0.0, 1.0)
        elif act == "none":
            mu = mean_raw
        else:
            raise ValueError(f"unknown mean_activation '{act}'")

        # Sigma output activations
        sigma = F.softplus(scale_raw) + self.sigma_min             # (B, W, H, W_sp) > 0
        
        # Output mean and sigma (DONE!)
        return mu, sigma

    def freeze_mean_pathway(self):
        """
        Freeze parameters for only training the sigma output head. This was 
        used during training to train both heads seperately. 

        Freeze every parameter that contributes to the MEAN, leaving ONLY the
        sigma_decoder trainable. Used in staged (Stage-2) uncertainty training:
        the mean is locked to its MSE-trained solution so NLL cannot hedge it,
        and sigma learns to predict the (now fixed) residual error magnitude.

        Frozen: smear, spectral_in, unet, mean_decoder (everything shared +
        the mean head). Also sets them to eval() so BatchNorm/Dropout-style
        layers are deterministic. Trainable: sigma_decoder only.
        """
        modules = []
        if self.smear is not None:
            modules.append(("smear", self.smear))
        modules += [("spectral_in", self.spectral_in),
                    ("unet", self.unet),
                    ("mean_decoder", self.mean_decoder)]
        for name, module in modules:
            for p in module.parameters():
                p.requires_grad = False
            module.eval()
        for p in self.sigma_decoder.parameters():
            p.requires_grad = True
        self.sigma_decoder.train()

    @torch.no_grad()
    def get_density_maps(self, obs_mask):
        """This is used for getting the density channel maps"""

        # Is Gaussian smear being used?
        if self.smear is None:                       # nn_fill: distance map instead
            _, dist = nn_fill_and_distance(
                obs_mask, obs_mask.float().unsqueeze(1))
            return [dist]
        
        density = obs_mask.float().unsqueeze(1)

        blurred = self.smear(density)

        if self.n_scales == 1:
            return [blurred]

        return [
            blurred[:, i:i+1]
            for i in range(self.n_scales)
        ]


# =============================================================================
# 3.  Dataset  (unchanged from conv_cnp_afmir.py)
# =============================================================================

class GridCNPDataset(Dataset):
    """
    Grid-level dataset for ConvCNP training.

    Each task returns a FULL cube (H, W_sp, W_wv) and a random binary observation
    mask (H, W_sp).  Context size is sampled (uniform or log-uniform) between
    n_context_min and n_context — train across a WIDE range, including very sparse
    contexts, so the model learns to be uncertain where observations are scarce.
    """

    def __init__(
        self,
        cube_data_list:      list,                          # Generation parametrs
        n_context:           int,                           # Number of context pixels
        n_tasks_per_epoch:   int,                           # Number of cubes per epoch
        val_fraction:        float = 0.10,                  # Validation set fraction
        split:               str   = "train",               # Which split we are generating
        seed:                int   = 42,                    # Random context seet
        n_context_min:       Optional[int] = None,          # Minimum number of context points
        context_dist:        str = 'log_uni',               # Distribution for context draws
        mode_frac:           float = 0.05,                  # Context distribution mode
        tail_frac:           float = 0.20,                  # Context distribution tail
        tail_quantile:       float = 0.99,                  # Context distribution tail quantile
        norm_mode:           str           = "cube_minmax", # Normalization mode
    ):
        # Ensure valid normalization input 
        if norm_mode not in NORM_MODES:                     # Reject unknown norm_mode
            raise ValueError(f"norm_mode must be one of {NORM_MODES}")  # Error out

        self.norm_mode = norm_mode                    # Assign normalization input
        rng     = np.random.default_rng(seed)         # Establish rng
        n_total = len(cube_data_list)                 # Total cubes
        n_val   = max(1, int(n_total * val_fraction)) # Number of validation cubes
        perm    = rng.permutation(n_total)            # Get n_total permutations
        val_set = set(perm[:n_val].tolist())          # Permute validation

        self.cube_indices        = [i for i in range(n_total)
                                    if (i in val_set) == (split == "val")]  # This split's cube indices
        self.cube_data_list      = cube_data_list             # Store cube data list
        self.n_context           = n_context                  # Store max context size
        self.n_context_min       = n_context_min if n_context_min is not None else n_context  # Store min context size
        self.n_tasks             = n_tasks_per_epoch          # Store tasks per epoch
        self.seed                = seed                       # Store base seed
        self.context_dist        = context_dist               # Store context-size distribution type
        self.mode_frac           = mode_frac                  # Store log-normal mode fraction
        self.tail_frac           = tail_frac                  # Store log-normal tail fraction
        self.tail_quantile       = tail_quantile              # Store log-normal tail quantile

        first     = cube_data_list[self.cube_indices[0]]      # Grab first cube of this split
        self.H    = int(first.shape[0])                       # Store cube height
        self.Wsp  = int(first.shape[1])                       # Store cube width
        self.W    = int(first.shape[2])                       # Store number of wavenumber channels
        n_pix     = self.H * self.Wsp                         # Total pixels per cube

        for ci in self.cube_indices:                          # Check every cube in this split
            cd = cube_data_list[ci]                           # Get this cube
            if int(cd.shape[0]) != self.H or int(cd.shape[1]) != self.Wsp:  # Flag mismatched spatial shape
                raise ValueError(
                    f"Cube {ci} has spatial shape ({cd.shape[0]}, {cd.shape[1]}) "
                    f"but expected ({self.H}, {self.Wsp}). All cubes must match."
                )                                             # Error out on shape mismatch

        sampling = "log-uniform" if log_uniform_context else "uniform"  # Label sampling mode for print
        print(f"  GridCNPDataset [{split:5s}]: {len(self.cube_indices)} cubes  "
              f"(n_context={self.n_context_min}-{n_context}/{n_pix} pixels, "
              f"W_wv={self.W}, ctx_sampling={sampling})")      # Print dataset summary

    def __len__(self) -> int:                                  # Dataset length = tasks per epoch
        return self.n_tasks                                    # Return tasks per epoch

    def __getitem__(self, idx: int):                            # Build one training task
        rng = np.random.default_rng(self.seed + idx * 31337 + 7)# Per-item deterministic rng
        ci  = rng.choice(self.cube_indices)                     # Randomly pick a cube for this split
        cd  = self.cube_data_list[ci]                           # Get the chosen cube
        H, Wsp, Wv = cd.shape                                   # Unpack cube shape
        n_pix = H * Wsp                                         # Total pixels in this cube

        # Sample context and build observation mask
        if self.context_dist == 'log_uni':                      # Log-uniform context-size sampling
            log_lo = np.log(max(self.n_context_min, 1))         # Lower log bound
            log_hi = np.log(self.n_context + 1)                 # Upper log bound
            n_ctx = int(np.clip(
                np.round(np.exp(rng.uniform(log_lo, log_hi))),
                self.n_context_min, self.n_context,
            ))                                                  # Draw and clip context size
        elif self.context_dist == 'log_norm':                   # Log-normal context-size sampling
            mu, sigma, _, _ = lognormal_from_mode_tail(
                n_pix=n_pix,
                mode_frac=self.mode_frac,
                tail_frac=self.tail_frac,
                tail_quantile=self.tail_quantile,
            )                                                   # Fit log-normal params from mode/tail

            n_ctx = int(
                _rejection_truncate(
                    lambda r, k: np.round(r.lognormal(mean=mu, sigma=sigma, size=k)),
                    self.n_context_min,
                    self.n_context,
                    1,
                    rng,
                )[0]
            )                                                    # Draw context size, rejecting out-of-range
        else:                                                    # Uniform context-size sampling
            n_ctx = int(rng.integers(self.n_context_min, self.n_context + 1))  # Draw context size uniformly
        n_ctx = min(n_ctx, n_pix)                                # Cap context size at total pixels

        flat_perm   = rng.permutation(n_pix)                     # Shuffle all pixel indices
        obs_flat    = np.zeros(n_pix, dtype=bool)                # Start with nothing observed
        obs_flat[flat_perm[:n_ctx]] = True                       # Mark first n_ctx shuffled pixels observed
        obs_mask    = obs_flat.reshape(H, Wsp)                   # Reshape to 2D context mask

        # Normalize cube
        cube_raw = cd.load_vals_raw().reshape(H, Wsp, Wv)        # Load raw spectra, reshape to cube
        cube, _stats = normalize_cube(cube_raw, obs_mask, mode=self.norm_mode)  # Normalize cube using this mask

        return (torch.from_numpy(cube),              # (H, Wsp, W_wv) float32
                torch.from_numpy(obs_mask))          # (H, Wsp) bool


def grid_cnp_collate(batch):
    """Collate full-grid samples. All cubes share (H, Wsp, W_wv); no padding.
    Sample tuples: (cube, mask), (cube, mask, band), or (cube, mask, bands_pad,
    n_valid) for multi-image variance selection."""
    n = len(batch[0])
    if n == 4:
        cubes, masks, bands, nval = zip(*batch)
        return (torch.stack(cubes), torch.stack(masks),
                torch.stack(bands), torch.stack(nval))
    if n == 3:
        cubes, masks, bands = zip(*batch)
        return torch.stack(cubes), torch.stack(masks), torch.stack(bands)
    cubes, masks = zip(*batch)
    return torch.stack(cubes), torch.stack(masks)


# =============================================================================
# 4.  Lightning module  
# =============================================================================

class CACNPLightningModule(pl.LightningModule):
    """
    Lightning wrapper for the probabilistic CACNP.

    Logged each epoch:
        {stage}/nll              -- the training objective (lower is better)
        {stage}/mse              -- mean-only MSE on targets (comparability)
        {stage}/mse_pixel        -- band-averaged spatial MSE (matches old logs)
        val/coverage_2sigma      -- empirical P(|y-mu| <= 2 sigma); ~0.954 if calibrated
        val/mean_sigma           -- mean target sigma; watch for collapse toward 0
        train/smear_sigma_px     -- learned set-conv bandwidth
    """

    def __init__(
        self,
        W:                int,                          # Number of wavenumber channels
        base_channels:    int       = 64,               # First U-Net channel width
        enc_channels:     List[int] = None,             # Four encoder channel widths
        kernel_sigma:     float     = 1.5,              # Initial smear bandwidth (px)
        kernel_size:      int       = 9,                # Gaussian smear kernel size
        dropout:          float     = 0.0,              # U-Net dropout rate
        sigma_min:        float     = 1e-3,             # Floor on predicted sigma
        lr:               float     = 3e-4,             # Base learning rate
        weight_decay:     float     = 1e-5,             # AdamW weight decay
        lr_warmup_epochs: int       = 5,                # Linear LR warmup length
        max_epochs:       int       = 200,              # Total epochs (for cosine schedule)
        norm_mask:        bool      =True,              # Hard density normalization of smear
        scales:           Optional[List[float]] = None, # Multi-scale smear bandwidths
        spectral_mlp:     bool      = False,            # Two-layer 1x1 MLP encoder/heads
        mlp_hidden:       int       = 256,              # Hidden width of that MLP
        lambda_ctx:       float     = 0.1,              # Weight on context-sigma penalty
        norm_smear:       bool      = False,            # Soft per-scale density normalization
        norm_eps:         float     = 0.05,             # Eps fraction for soft normalization
        lambda_mse:       float     = 0.0,              # Weight on target MSE term
        lambda_img:       float     = 0.0,              # Weight on structure-image level match
        lambda_grad:      float     = 0.0,              # Weight on imaged-band gradient match
        lambda_grad_full: float     = 0.0,              # Weight on all-band gradient match
        lambda_tv:        float     = 0.0,              # Weight on isotropic spatial TV
        tv_eps:           float     = 1e-6,             # Eps inside the TV sqrt
        mean_activation:  str       = "sigmoid",        # Output mean activation type
        sig_margin:       float     = 0.05,             # Margin for scaled_sigmoid
        mse_only:         bool      = False,            # Legacy switch forcing mean_mse
        train_mode:       str       = "joint_nll",      # Which training regime to run
        init_from:        Optional[str] = None,         # Checkpoint to warm-start weights from
        gen_params:       Optional[dict] = None,        # Generation ranges (logged to hparams)
        onthefly:         bool      = False,            # Whether data is generated live
        norm_mode:        str       = "cube_minmax",    # Cube normalization scheme
        input_mode:       str       = "smear",          # Encoder front end: smear or nn_fill
        image_bands:      Optional[List[int]] = None,   # Fixed imaged band indices
        dynamic_image_band: bool  = False,              # Pick one imaged band per cube at runtime
        image_peak_height: float  = 0.2,                # find_peaks height for dynamic selection
        image_tokens:     bool      = False,            # Route images as attention tokens
        image_subsample:  int       = 1024,             # Image tokens per image
        dec0_channels:    Optional[int] = None,         # Width of final decoder block
        guidance_agg:     str       = "sum",            # How multiple images are pooled
        max_images:       int       = 4,                # Slots for concat guidance
        guidance_raw:     bool      = False,            # Also pass raw image planes
        decoder_skip:     str       = "none",           # Which tensor to skip to the heads
        image_guidance:   bool      = False,            # Concatenate guidance into encoder input
        film:             bool      = False,            # FiLM-condition encoder on imaged bands
        multi_image:      bool      = False,            # Dataset supplies 1..N variance-picked bands
        image_var_thresh: float     = 0.2,              # Variance-peak threshold for band picking
        use_attention:    bool      = False,            # Enable cross-attention stages
        attn_stages:      Optional[List[str]] = None,   # Which U-Net stages get attention
        attn_heads:       int       = 4,                # Attention heads per stage
        attn_dim:         int       = 256,              # Attention d_model
        attn_pos_bands:   int       = 8,                # Fourier positional bands
        head_kernel:      int       = 1,                # Output-head conv kernel size
        head_attn:        bool      = False,            # Enable head-resolution attention
        head_attn_dim:    int       = 320,              # Head attention d_model
        head_attn_heads:  int       = 4,                # Head attention heads
        # Adaptive-context training (transfer-learn on adaptively-sampled ctx) 
        adaptive_context:      bool  = False,           # Master switch
        adaptive_points_per_step: int = 1,              # Pixels added per acquisition round
        adaptive_start_frac_min: float = 0.001,         # Hex seed start fraction: min (0.1%)
        adaptive_start_frac_max: float = 0.01,          # Hex seed start fraction: max (1%)
        adaptive_acq_reduce:   str   = "mean",          # Sigma reduction over bands: mean|max
        adaptive_refresh_epochs: int = 5,               # Refresh frozen acquirer every N epochs
    ):
        super().__init__()                              # Init the Lightning base class
        self.save_hyperparameters()                     # Record all args into hparams

        # Adaptive-context training configuration
        self.adaptive_context      = bool(adaptive_context)           # Store adaptive master switch
        self.adaptive_points_per_step = int(adaptive_points_per_step) # Store points per round
        self.adaptive_start_frac_min = float(adaptive_start_frac_min) # Store min seed fraction
        self.adaptive_start_frac_max = float(adaptive_start_frac_max) # Store max seed fraction
        self.adaptive_acq_reduce   = str(adaptive_acq_reduce)         # Store sigma reduction mode
        self.adaptive_frozen_acquirer = True                          # Freeze acquierer (refreshed periodically)
        self.adaptive_refresh_epochs = int(adaptive_refresh_epochs)   # Store refresh interval
        object.__setattr__(self, "_frozen_acquirer", None)            # Built lazily; not a submodule

        # Specify loss function parameters
        self.lambda_ctx = lambda_ctx                    # Store context-sigma weight
        self.lambda_mse = lambda_mse                    # Store target-MSE weight
        self.lambda_img = lambda_img                    # Store image-level weight
        self.lambda_grad = lambda_grad                  # Store imaged-band gradient weight
        self.lambda_grad_full = lambda_grad_full        # Store all-band gradient weight
        self.lambda_tv = lambda_tv                      # Store TV weight
        self.tv_eps = tv_eps                            # Store TV epsilon
        
        # What training mode to use
        if mse_only:                                    # Legacy flag takes precedence
            train_mode = "mean_mse"                     # Force Stage-1 regime
        if train_mode not in ("mean_mse", "sigma_nll", "joint_nll"):  # Validate regime
            raise ValueError(f"unknown train_mode '{train_mode}'")    # Error out
        
        self.train_mode = train_mode                        # Store training regime
        self.image_bands = list(image_bands) if image_bands else None  # Store fixed imaged bands
        self.dynamic_image_band = bool(dynamic_image_band)  # Store dynamic-band flag
        self.image_peak_height  = float(image_peak_height)  # Store peak-height threshold
        self.image_tokens       = bool(image_tokens)        # Store image-token flag
        self.image_guidance     = bool(image_guidance)      # Store guidance flag
        self.film               = bool(film)                # Store FiLM flag
        self.multi_image        = bool(multi_image)         # Store multi-image flag
        self.image_var_thresh   = float(image_var_thresh)   # Store variance threshold
        self.mse_only   = (train_mode == "mean_mse")        # Derived legacy flag

        if enc_channels is None:                            # No explicit encoder widths given
            enc_channels = [base_channels,     base_channels * 2,
                            base_channels * 4, base_channels * 8]  # Default 1x,2x,4x,8x widths

        self.model = CACNP(                             # Build the underlying network
            W=W,                                        # Wavenumber channels
            base_channels=base_channels,                # First U-Net width
            enc_channels=enc_channels,                  # Encoder widths
            kernel_sigma=kernel_sigma,                  # Smear bandwidth
            kernel_size=kernel_size,                    # Smear kernel size
            dropout=dropout,                            # Dropout rate
            sigma_min=sigma_min,                        # Sigma floor
            norm_mask=norm_mask,                        # Hard density normalization
            scales=scales,                              # Multi-scale bandwidths
            spectral_mlp=spectral_mlp,                  # MLP encoder/heads
            mlp_hidden=mlp_hidden,                      # MLP hidden width
            norm_smear=norm_smear,                      # Soft density normalization
            norm_eps=norm_eps,                          # Soft-normalization eps
            mean_activation=mean_activation,            # Mean activation type
            sig_margin=sig_margin,                      # scaled_sigmoid margin
            input_mode=input_mode,                      # Encoder front end
            image_bands=image_bands,                    # Fixed imaged bands
            dynamic_image_band=dynamic_image_band,      # Dynamic band selection
            image_tokens=image_tokens,                  # Image-token routing
            image_subsample=image_subsample,            # Tokens per image
            dec0_channels=dec0_channels,                # Final decoder width
            guidance_agg=guidance_agg,                  # Guidance pooling mode
            max_images=max_images,                      # Guidance slots
            guidance_raw=guidance_raw,                  # Include raw image planes
            decoder_skip=decoder_skip,                  # Decoder skip source
            image_guidance=image_guidance,              # Guidance concatenation
            film=film,                                  # FiLM conditioning
            multi_image=multi_image,                    # Multi-image mode
            use_attention=use_attention,                # Enable attention
            attn_stages=attn_stages,                    # Attention stages
            attn_heads=attn_heads,                      # Attention heads
            attn_dim=attn_dim,                          # Attention d_model
            attn_pos_bands=attn_pos_bands,              # Fourier position bands
            head_kernel=head_kernel,                    # Head conv kernel size
            head_attn=head_attn,                        # Head attention on/off
            head_attn_dim=head_attn_dim,                # Head attention d_model
            head_attn_heads=head_attn_heads,            # Head attention heads
        )

        # Checks if starting from pretrained model, and if so loads weights
        if init_from is not None:                       # A warm-start checkpoint was given
            self._load_init_weights(init_from)          # Load its matching weights

        # Checks if only training sigma head, and if so freezes everything else
        if self.train_mode == "sigma_nll":              # Stage-2 regime
            self.model.freeze_mean_pathway()            # Lock everything but the sigma head

    def _load_init_weights(self, ckpt_path):            # Warm-start weight loader
        ckpt = torch.load(ckpt_path, map_location="cpu")# Load checkpoint onto CPU
        sd = ckpt.get("state_dict", ckpt)               # Grab its state_dict
        # Strip a leading "model." if present to match our submodule naming
        own = self.state_dict()                         # Copy of this module's own state
        loaded, skipped = 0, 0                          # Counters for the report
        for k, v in sd.items():                         # Walk every source tensor
            if k in own and own[k].shape == v.shape:    # Key and shape both match
                own[k] = v; loaded += 1                 # Take it and count it
            else:                                       # Missing key or shape mismatch
                skipped += 1                            # Count it as skipped
        self.load_state_dict(own, strict=False)         # Apply the merged state
        print(f"  init_from: loaded {loaded} tensors, skipped {skipped} "
              f"(shape/key mismatch — e.g. fresh sigma head) from {ckpt_path}")  # Report result

    
    def _image_planes(self, cube_bhwc, dyn_bands=None):  # Extract imaged band plane(s)
        """Full-field image plane(s) from a cube (B,H,W,Wv).
        Dynamic mode: dyn_bands (B,) -> per-sample plane (B,1,H,W) + bands (B,).
        Fixed mode: (B,n_img,H,W) + None. No image: (None, None)."""
        if self.dynamic_image_band:                        # One band per sample, chosen at runtime
            B = cube_bhwc.shape[0]                         # Batch size
            bands = dyn_bands.to(cube_bhwc.device).long()  # Per-sample band indices
            chw = cube_bhwc.permute(0, 3, 1, 2)            # (B,Wv,H,W)
            planes = chw[torch.arange(B, device=chw.device), bands]   # (B,H,W)
            return planes.unsqueeze(1).contiguous(), bands # Add channel axis, return with bands
        if not self.image_bands:                           # No imaged bands configured
            return None, None                              # Nothing to return
        return cube_bhwc[..., self.image_bands].permute(0, 3, 1, 2).contiguous(), None  # Fixed bands

    def forward(self, obs_mask, ctx_spectra_grid, image_planes=None, image_bands=None):  # Thin wrapper
        return self.model(obs_mask, ctx_spectra_grid,
                          image_planes=image_planes, image_bands=image_bands)  # Delegate to CACNP

    @staticmethod
    def _build_ctx_grid(cubes, obs_masks):              # Zero out unobserved spectra
        """(B,H,W,Wv),(B,H,W) -> ctx_grid (B,Wv,H,W) with zeros at unobserved."""
        return cubes.permute(0, 3, 1, 2) * obs_masks.unsqueeze(1).float()  # Move Wv first, mask

    # ------------------------------------------------------------------
    # Adaptive-context training
    # ------------------------------------------------------------------
    def _ensure_frozen_acquirer(self):                  # Build/refresh the frozen acquirer
        """Lazily build/refresh the frozen acquirer that drives adaptive sampling.

        A frozen copy (not the live model) decouples the acquisition distribution
        from the model being trained, removing the moving-target instability of
        sampling by the live sigma while it changes. It is REFRESHED every
        adaptive_refresh_epochs epochs so the acquisition uncertainty tracks the
        improving model (a lagged target) -- without a refresh, the sigma driving
        acquisition would never learn to be accurate for the new context
        distribution.

        The acquirer is stored via object.__setattr__ so it is NOT registered as
        an nn.Module submodule and therefore NOT written into the checkpoint
        state_dict.
        """
        need = (self._frozen_acquirer is None)              # Need one if none exists yet
        if (not need) and self.adaptive_refresh_epochs > 0: # Otherwise check refresh schedule
            need = (int(self.current_epoch) > 0 and
                    int(self.current_epoch) % self.adaptive_refresh_epochs == 0 and
                    getattr(self, "_frozen_epoch", -1) != int(self.current_epoch))  # Due this epoch?
        if need:                                            # Time to (re)build the acquirer
            import copy                                     # For deepcopy
            acq = copy.deepcopy(self.model)                 # Snapshot the current model
            for p in acq.parameters():                      # Walk its parameters
                p.requires_grad_(False)                     # Freeze them
            acq.eval()                                      # Put it in eval mode
            object.__setattr__(self, "_frozen_acquirer", acq)   # not a submodule
            self._frozen_epoch = int(self.current_epoch)    # Record when it was built
        return self._frozen_acquirer                        # Hand back the acquirer
 
    @torch.no_grad()
    def _acq_sigma(self, acquirer, cube_bhwc, mask_bhw):  # Per-pixel acquisition score
        """One forward pass of the ACQUIRER on a batch, returning per-pixel sigma
        reduced over wavenumbers. No grad: this is data preparation, not the loss.

        cube_bhwc : (B,H,Wsp,Wv) normalised cube (same normalisation as _step input)
        mask_bhw  : (B,H,Wsp) bool current context
        returns   : (B,H,Wsp) float acquisition score (higher = sample next)
        """
        ctx = self._build_ctx_grid(cube_bhwc, mask_bhw) # Masked context grid
        # image planes selected exactly as in _step's no-image / multi-image path.
        if self.multi_image:                            # Multi-image band selection path
            # replicate select_image_bands per sample; cheap (variance peak on obs)
            B, H, Wsp, Wv = cube_bhwc.shape             # Unpack cube shape
            planes = []                                 # (unused placeholder list)
            bands = []                                  # Per-sample padded band lists
            valids = []                                 # Per-sample validity flags
            maxN = 1                                    # Track max bands across batch
            per = []                                    # Per-sample selected bands
            for b in range(B):                          # Loop over batch
                sel = self.select_image_bands(cube_bhwc[b], mask_bhw[b], n_images=1)  # Pick bands
                per.append(sel)                         # Store this sample's picks
                maxN = max(maxN, len(sel))              # Update padding width
            for b in range(B):                          # Second pass to pad
                sel = per[b] + [0] * (maxN - len(per[b]))  # Pad band list with zeros
                bands.append(sel)                       # Store padded bands
                valids.append([1] * len(per[b]) + [0] * (maxN - len(per[b])))  # Mark real vs padded
            bands_t = torch.tensor(bands, device=cube_bhwc.device, dtype=torch.long)  # (B,maxN)
            valid_t = torch.tensor(valids, device=cube_bhwc.device, dtype=torch.bool)  # Validity mask
            chw = cube_bhwc.permute(0, 3, 1, 2)         # (B,Wv,H,Wsp)
            idx = bands_t.view(B, maxN, 1, 1).expand(B, maxN, H, Wsp)  # Broadcast band indices
            img_planes = torch.gather(chw, 1, idx)      # Gather the imaged planes
            _, sigma = acquirer(mask_bhw, ctx, image_planes=img_planes,
                                image_bands=bands_t, image_valid=valid_t)  # Acquirer forward
        else:                                           # Fixed / dynamic / no-image path
            img_planes, img_bands = self._image_planes(cube_bhwc, None)  # Get planes if any
            _, sigma = acquirer(mask_bhw, ctx, image_planes=img_planes,
                                image_bands=img_bands, image_valid=None)  # Acquirer forward
        # sigma: (B,Wv,H,Wsp) -> reduce over Wv
        if self.adaptive_acq_reduce == "max":           # Max-over-bands reduction
            acq = sigma.max(dim=1).values               # Take per-pixel max sigma
        else:                                           # Mean-over-bands reduction
            acq = sigma.mean(dim=1)                     # Take per-pixel mean sigma
        return acq                                            # (B,H,Wsp)

    @torch.no_grad()
    def build_adaptive_mask(self, cubes, obs_masks):    # Grow an adaptive context mask
        """Replace each sample's random context with an ADAPTIVELY-grown one.

        Procedure (pure adaptive; matches the deployment acquisition):
          * Start from a hexagonal seed grid whose fraction is drawn uniformly per cube
            in [adaptive_start_frac_min, adaptive_start_frac_max] (0.1%..1%);
          * iterate: run the frozen acquirer, add the top adaptive_points_per_step
            highest-sigma unobserved pixels per sample, until reaching that
            sample's original context count (its random n_ctx, log_norm-distributed,
            becomes the adaptive STOP count, preserving the context-size range).
        every sample is converted (no random-context mixing): the model trains on
        exactly the uncertainty-selected context distribution it faces at
        deployment. The reconstruction loss is evaluated on unobserved pixels,
        which under adaptive sampling are predominantly region interiors -- so the
        gradient is naturally concentrated on interior propagation, which is the
        skill this run is meant to fix.

        Returns a new (B,H,Wsp) bool mask tensor.
        """
        B, H, Wsp, Wv = cubes.shape                     # Unpack batch cube shape
        device = cubes.device                           # Device to build masks on
        n_pix = H * Wsp                                 # Pixels per cube
        acquirer = self._ensure_frozen_acquirer()       # Get the frozen acquirer
        if acquirer is None:                            # None available
            acquirer = self.model                       # live-model acquisition

        target_counts = obs_masks.view(B, -1).sum(dim=1).tolist()   # per-sample stop

        # Per-batch rng for hex seed offsets + start-fraction draws (deterministic
        # in global_step so a resumed run is reproducible)
        np_rng = np.random.default_rng(int(self.global_step) * 2654435761 % (2**31))  # Seeded rng

        new_mask = obs_masks.clone()                    # Start from a copy of the input masks
        for b in range(B):                              # Handle each sample separately
            stop = int(target_counts[b])                # This sample's target context count
            # Per-cube start fraction uniform in [min, max] (0.1%..1%)
            sf = float(np_rng.uniform(self.adaptive_start_frac_min,
                                      self.adaptive_start_frac_max))  # Draw seed fraction
            seed = hex_seed_mask(H, Wsp, sf, np_rng)    # (H,Wsp) bool numpy
            m = torch.from_numpy(seed).to(device)       # Move seed mask to device
            # If the hex seed already meets/exceeds the stop count, just use it
            if int(m.sum()) >= stop:                    # Seed already big enough
                new_mask[b] = m                         # Use it as-is
                continue                                # Skip the growth loop
            # Grow adaptively to 'stop' via the frozen acquirer's sigma
            while int(m.sum()) < stop:                  # Until the target count is reached
                acq = self._acq_sigma(acquirer,
                                      cubes[b:b+1], m.unsqueeze(0))[0]   # (H,Wsp)
                acq = acq.masked_fill(m, float("-inf")) # Never re-pick observed pixels
                k = min(self.adaptive_points_per_step, stop - int(m.sum()))  # Points to add now
                flat = acq.view(-1)                     # Flatten for topk
                topk = torch.topk(flat, k).indices      # Highest-sigma unobserved pixels
                m.view(-1)[topk] = True                 # Mark them observed
            new_mask[b] = m                             # Store this sample's grown mask
        return new_mask                                 # Return the adaptive masks

    def _step(self, batch, stage: str):                 # Shared train/val step
        # Adaptive-context training: replace random ctx with adaptive ctx 
        # Only during training, and only when enabled. Acquisition is no-grad; only
        # the loss forward below carries gradients. Validation stays randoim so the
        # metric is comparable across runs (change adaptive_context to also convert
        # val if you want to monitor adaptive-context calibration during training).
        if self.adaptive_context and stage == "train":      # Adaptive mode, training only
            _cubes = batch[0]                               # Cubes from the batch
            _masks = batch[1]                               # Original random masks
            _new = self.build_adaptive_mask(_cubes, _masks) # Build adaptive masks
            batch = (_cubes, _new) + tuple(batch[2:])       # Swap them into the batch

        img_valid = None                                    # Default: no image padding mask
        if len(batch) == 4:                                 # Multi-image batch layout
            # Multi-image: bands_pad (B,max_N), n_valid (B,)
            cubes, obs_masks, bands_pad, n_valid = batch       # Unpack four-element batch
            ctx_grid = self._build_ctx_grid(cubes, obs_masks)  # Masked context grid
            B, H, Wsp, Wv = cubes.shape                        # Unpack cube shape
            maxN = bands_pad.shape[1]                          # Padded image-slot count
            # Gather imaged planes at the (padded) bands: (B, maxN, H, W)
            chw = cubes.permute(0, 3, 1, 2)                             # (B,Wv,H,W)
            idx = bands_pad.view(B, maxN, 1, 1).expand(B, maxN, H, Wsp) # Broadcast band indices
            img_planes = torch.gather(chw, 1, idx)                      # (B,maxN,H,W)
            img_bands  = bands_pad                                      # (B,maxN)
            img_valid  = (torch.arange(maxN, device=cubes.device)[None, :]
                          < n_valid[:, None])                           # (B,maxN) bool
        elif len(batch) == 3:                                  # Dynamic-band batch layout
            cubes, obs_masks, dyn_bands = batch                # Unpack three-element batch
            ctx_grid = self._build_ctx_grid(cubes, obs_masks)  # Masked context grid
            img_planes, img_bands = self._image_planes(cubes, dyn_bands)  # Per-sample plane
        else:                                                  # Plain batch layout
            cubes, obs_masks = batch                           # Unpack two-element batch
            ctx_grid = self._build_ctx_grid(cubes, obs_masks)  # Masked context grid
            img_planes, img_bands = self._image_planes(cubes, None)  # Fixed / no images

        # Run data through model
        mu, sigma = self.model(obs_masks, ctx_grid,
                               image_planes=img_planes, image_bands=img_bands,
                               image_valid=img_valid)   # Forward pass -> predictive params

        true      = cubes.permute(0, 3, 1, 2)           # Get GT hyperspectral cubes (B,Wv,H,W)

        # ------------------------------------------------------------------
        # Masks
        # ------------------------------------------------------------------
        ctx = obs_masks                                 # Observed (context) pixels
        tgt = ~obs_masks                                # Unobserved (target) pixels

        ctx_exp = ctx.unsqueeze(1).expand_as(mu)        # Context mask broadcast over bands
        tgt_exp = tgt.unsqueeze(1).expand_as(mu)        # Target mask broadcast over bands

        n_ctx = ctx_exp.sum()                           # Count of context voxels
        n_tgt = tgt_exp.sum()                           # Count of target voxels

        # ------------------------------------------------------------------
        # Training objective (selected by train_mode)
        # ------------------------------------------------------------------
        dist = torch.distributions.Normal(mu, sigma)    # Predictive Gaussian
        nll_full = -dist.log_prob(true)                 # Per-voxel negative log-likelihood

        if self.train_mode == "mean_mse":               # Stage-1 regime
            # Stage 1: full-cube MSE on the mean only. Sigma head present but
            # excluded from the loss (untrained, not meaningful). Monitors
            # val/recon_mse for checkpoint/early-stop.
            loss = (mu - true).pow(2).mean()            # Full-cube mean MSE

        elif self.train_mode == "sigma_nll":            # Stage-2 regime
            # Stage 2: pure full-cube NLL with the mean pathway FROZEN, so only
            # sigma_decoder learns. With mu fixed, NLL reduces to a well-posed
            # regression of sigma onto the residual magnitude (optimal sigma =
            # |y - mu|); no mean hedging is possible. No context-sigma penalty
            # here — we want sigma to honestly track error everywhere.
            loss = nll_full.mean()                      # Pure full-cube NLL

        else:  # "joint_nll" — comparison baseline: both heads, joint NLL
            loss = nll_full.mean() + self.lambda_ctx * sigma[ctx_exp].mean()  # NLL + context-sigma
            if self.lambda_mse > 0.0 and n_tgt > 0:     # Optional target-MSE term
                mse_loss = (mu - true).pow(2)[tgt_exp].mean()  # MSE on unobserved voxels
                loss = loss + self.lambda_mse * mse_loss  # Add it to the loss

        # ------------------------------------------------------------------
        # Structure-image consistency
        # ------------------------------------------------------------------
        img_mse = None                                  # Default: term not computed
        if getattr(self, "lambda_img", 0.0) > 0.0 and img_planes is not None:  # Term enabled
            Bm, _, Hm, Wm = mu.shape                                           # Prediction shape
            ib = img_bands if img_bands.dim() > 1 else img_bands.view(Bm, 1)   # Ensure 2D bands
            ip = img_planes if img_planes.dim() == 4 else img_planes.unsqueeze(1)  # Ensure 4D planes
            idx_i = ib.view(Bm, ib.shape[1], 1, 1).expand(-1, -1, Hm, Wm)      # Broadcast band indices
            mu_at = torch.gather(mu, 1, idx_i)                                 # (B, N, H, W)
            se = (mu_at - ip).pow(2)                                           # Squared error vs the image
            if img_valid is not None:                                          # ignore padded slots
                wv = img_valid.view(Bm, -1, 1, 1).to(se.dtype)                 # Validity weights
                denom = (wv.sum() * Hm * Wm).clamp(min=1.0)                    # Valid-voxel count
                img_mse = (se * wv).sum() / denom                              # Masked mean
            else:                                                              # No padding to mask
                img_mse = se.mean()                                            # Plain mean
            loss = loss + self.lambda_img * img_mse                            # Add weighted term to the loss

        # ------------------------------------------------------------------
        # Structure-image GRADIENT consistency (edge-matching)
        # ------------------------------------------------------------------
        grad_cons = None                                # Default: term not computed
        if getattr(self, "lambda_grad", 0.0) > 0.0 and img_planes is not None:  # Term enabled
            Bm, _, Hm, Wm = mu.shape                                            # Prediction shape
            ib = img_bands if img_bands.dim() > 1 else img_bands.view(Bm, 1)    # Ensure 2D bands
            ip = img_planes if img_planes.dim() == 4 else img_planes.unsqueeze(1)  # Ensure 4D planes
            idx_i = ib.view(Bm, ib.shape[1], 1, 1).expand(-1, -1, Hm, Wm)       # Broadcast band indices
            mu_at = torch.gather(mu, 1, idx_i)                                  # (B, n_img, H, W)
            # forward-difference spatial gradients (band-wise)
            mu_dh = mu_at[:, :, 1:, :] - mu_at[:, :, :-1, :]                    # (B, n_img, H-1, W)
            mu_dw = mu_at[:, :, :, 1:] - mu_at[:, :, :, :-1]                    # (B, n_img, H, W-1)
            ip_dh = ip[:, :, 1:, :]    - ip[:, :, :-1, :]                       # Image gradient along H
            ip_dw = ip[:, :, :, 1:]    - ip[:, :, :, :-1]                       # Image gradient along W
            se_dh = (mu_dh - ip_dh).pow(2)                                      # Squared gradient error along H
            se_dw = (mu_dw - ip_dw).pow(2)                                      # Squared gradient error along W
            if img_valid is not None:                                           # ignore padded slots
                wv = img_valid.view(Bm, -1, 1, 1).to(se_dh.dtype)               # Validity weights
                denom_dh = (wv.sum() * (Hm - 1) * Wm).clamp(min=1.0)            # Valid count along H
                denom_dw = (wv.sum() * Hm * (Wm - 1)).clamp(min=1.0)            # Valid count along W
                grad_cons = (se_dh * wv).sum() / denom_dh \
                          + (se_dw * wv).sum() / denom_dw                       # Masked mean of both
            else:                                                               # No padding to mask
                grad_cons = se_dh.mean() + se_dw.mean()                         # Plain mean of both
            loss = loss + self.lambda_grad * grad_cons                          # Add weighted term to the loss

        # ------------------------------------------------------------------
        # ALL-BAND spatial gradient consistency (lambda_grad_full).
        # ------------------------------------------------------------------
        grad_cons_full = None                                    # Default: term not computed
        if getattr(self, "lambda_grad_full", 0.0) > 0.0:         # Term enabled
            mu_dh_f  = mu[:, :, 1:, :]   - mu[:, :, :-1, :]      # (B, C, H-1, W)
            mu_dw_f  = mu[:, :, :, 1:]   - mu[:, :, :, :-1]      # (B, C, H, W-1)
            tr_dh_f  = true[:, :, 1:, :] - true[:, :, :-1, :]    # Truth gradient along H
            tr_dw_f  = true[:, :, :, 1:] - true[:, :, :, :-1]    # Truth gradient along W
            grad_cons_full = (mu_dh_f - tr_dh_f).pow(2).mean() \
                           + (mu_dw_f - tr_dw_f).pow(2).mean()   # Mean squared gradient error
            loss = loss + self.lambda_grad_full * grad_cons_full # Add weighted term

        # ------------------------------------------------------------------
        # Total-variation regularization (spatial only)
        # ------------------------------------------------------------------
        tv_term = None                                  # Default: term not computed
        if getattr(self, "lambda_tv", 0.0) > 0.0:       # Term enabled
            dh = mu[:, :, 1:, :] - mu[:, :, :-1, :]     # (B, W, H-1, W_sp)
            dw = mu[:, :, :, 1:] - mu[:, :, :, :-1]     # (B, W, H, W_sp-1)
            # crop to the common (H-1, W_sp-1) region so the two gradients align
            dh = dh[:, :, :, :-1]                       # Trim dh's width by one
            dw = dw[:, :, :-1, :]                       # Trim dw's height by one
            tv_term = torch.sqrt(dh.pow(2) + dw.pow(2) + self.tv_eps).mean()  # Isotropic TV
            loss = loss + self.lambda_tv * tv_term      # Add weighted term to the loss

        # ------------------------------------------------------------------
        # Diagnostics (logs data for monitoring training progress)
        # ------------------------------------------------------------------
        with torch.no_grad():                           # Metrics only, no gradients

            # ----- NLL -----
            nll_ctx = (
                nll_full[ctx_exp].mean()                # Mean NLL on context voxels
                if n_ctx > 0                            # Only if any exist
                else torch.zeros((), device=mu.device)  # Otherwise zero
            )

            nll_tgt = (
                nll_full[tgt_exp].mean()                # Mean NLL on target voxels
                if n_tgt > 0                            # Only if any exist
                else torch.zeros((), device=mu.device)  # Otherwise zero
            )

            # ----- Sigma -----
            sigma_ctx = (
                sigma[ctx_exp].mean()                   # Mean sigma on context voxels
                if n_ctx > 0                            # Only if any exist
                else torch.zeros((), device=mu.device)  # Otherwise zero
            )

            sigma_tgt = (
                sigma[tgt_exp].mean()                   # Mean sigma on target voxels
                if n_tgt > 0                            # Only if any exist
                else torch.zeros((), device=mu.device)  # Otherwise zero
            )

            # ----- MSE -----
            se = (mu - true).pow(2)                     # Per-voxel squared error

            mse_ctx = (
                se[ctx_exp].mean()                      # Mean MSE on context voxels
                if n_ctx > 0                            # Only if any exist
                else torch.zeros((), device=mu.device)  # Otherwise zero
            )

            mse_tgt = (
                se[tgt_exp].mean()                      # Mean MSE on target voxels
                if n_tgt > 0                            # Only if any exist
                else torch.zeros((), device=mu.device)  # Otherwise zero
            )

            mse = se.mean()                             # Full-cube mean MSE

            # ----- Pixel-averaged spectral MSE -----
            pred_mean = mu.mean(dim=1)        # (B,H,W)
            true_mean = true.mean(dim=1)                # Band-averaged truth

            mse_pixel = (pred_mean - true_mean).pow(2).mean()  # Spatial MSE of band means

            # ----- Calibration -----
            within2 = (
                (true >= mu - 2 * sigma) &              # Truth above the lower 2-sigma bound
                (true <= mu + 2 * sigma)                # Truth below the upper 2-sigma bound
            ).float()                                   # As 0/1 for averaging

            cov2_ctx = (
                within2[ctx_exp].mean()                 # Coverage on context voxels
                if n_ctx > 0                            # Only if any exist
                else torch.zeros((), device=mu.device)  # Otherwise zero
            )

            cov2_tgt = (
                within2[tgt_exp].mean()                 # Coverage on target voxels
                if n_tgt > 0                            # Only if any exist
                else torch.zeros((), device=mu.device)  # Otherwise zero
            )

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if tv_term is not None:                         # TV term was computed
            self.log(f"{stage}/tv", tv_term.detach(),
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)  # Log TV
        if img_mse is not None:                         # Image-level term was computed
            self.log(f"{stage}/img_mse", img_mse.detach(),
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)  # Log img_mse
        if grad_cons is not None:                       # Imaged-band gradient term was computed
            self.log(f"{stage}/grad_cons", grad_cons.detach(),
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)  # Log grad_cons
        if grad_cons_full is not None:                  # All-band gradient term was computed
            self.log(f"{stage}/grad_cons_full", grad_cons_full.detach(),
                     on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)  # Log grad_cons_full
        self.log(f"{stage}/nll",
                 nll_full.mean(),
                 on_step=False, on_epoch=True, prog_bar=True)  # Log full-cube NLL

        # Full-cube reconstruction MSE — the monitored quantity in mse_only mode
        # (and a useful diagnostic always). Logged as recon_mse so it is distinct
        # from mse_tgt / mse_ctx splits below.
        self.log(f"{stage}/recon_mse",
                 mse,
                 on_step=False, on_epoch=True, prog_bar=True)  # Log reconstruction MSE

        self.log(f"{stage}/mse",
                 mse,
                 on_step=False, on_epoch=True, prog_bar=True)  # Log same MSE under /mse

        self.log(f"{stage}/mse_pixel",
                 mse_pixel,
                 on_step=False, on_epoch=True, prog_bar=False) # Log band-averaged spatial MSE

        # ----- Context metrics -----
        self.log(f"{stage}/nll_ctx",
                 nll_ctx,
                 on_step=False, on_epoch=True)          # Log context NLL

        self.log(f"{stage}/mse_ctx",
                 mse_ctx,
                 on_step=False, on_epoch=True)          # Log context MSE

        self.log(f"{stage}/sigma_ctx",
                 sigma_ctx,
                 on_step=False, on_epoch=True)          # Log context sigma

        # ----- Target metrics -----
        self.log(f"{stage}/nll_tgt",
                 nll_tgt,
                 on_step=False, on_epoch=True)          # Log target NLL

        self.log(f"{stage}/mse_tgt",
                 mse_tgt,
                 on_step=False, on_epoch=True)          # Log target MSE

        self.log(f"{stage}/sigma_tgt",
                 sigma_tgt,
                 on_step=False, on_epoch=True)          # Log target sigma

        if stage == "val":                              # Validation-only calibration metrics
            self.log("val/coverage_2sigma_ctx",
                     cov2_ctx,
                     on_step=False, on_epoch=True)      # Log context 2-sigma coverage

            self.log("val/coverage_2sigma_tgt",
                     cov2_tgt,
                     on_step=False, on_epoch=True)      # Log target 2-sigma coverage

        return loss                                     # Hand the loss back to Lightning

    def training_step(self, batch, _):                  # Lightning training hook
        return self._step(batch, "train")               # Run the shared step in train mode

    def validation_step(self, batch, _):                # Lightning validation hook
        return self._step(batch, "val")                 # Run the shared step in val mode

    def on_train_epoch_end(self):                       # End-of-epoch logging hook
        smear = self.model.smear                        # Get the smear module
        if smear is None:                              # nn_fill: no learnable radii
            return                                      # Nothing to log
        if hasattr(smear, "current_sigmas"):          # multi-scale
            for i, s in enumerate(smear.current_sigmas()):  # Each scale's bandwidth
                self.log(f"train/smear_sigma_px_{i}", s,
                         on_step=False, on_epoch=True, prog_bar=False)  # Log per-scale sigma
        else:                                          # single-scale
            sigma = torch.exp(smear.log_sigma).item()   # Recover sigma from log-parameter
            self.log("train/smear_sigma_px", sigma,
                     on_step=False, on_epoch=True, prog_bar=False)  # Log single sigma

    # ---- inference / adaptive-sampling interface ----
    @torch.no_grad()
    def select_image_bands(self, cube: torch.Tensor, obs_mask: torch.Tensor,
                           n_images: int = 1) -> List[int]:  # Pick imaged bands at test time
        """
        Multi-image models: pick the imaged band(s) the same way training did —
        rank bands by the variance of the sampled (observed) spectra (find_peaks on
        the min-max-normalised band variance) and take the top-n_images peaks.
        """
        from synth_onthefly import variance_peak_bands           # Import the shared ranking helper
        sampled = cube[obs_mask].detach().cpu().numpy()          # (n_obs, W)
        ranked = variance_peak_bands(sampled, self.image_var_thresh)  # Rank bands by variance peaks
        if len(ranked) == 0:                                     # fallback: top variance
            ranked = np.array([int(sampled.var(axis=0).argmax())]) if sampled.shape[0] >= 2 \
                     else np.array([int(cube.shape[-1] // 2)])   # Highest-variance band, or middle
        n = max(1, min(int(n_images), len(ranked)))              # Clamp request to what's available
        return [int(b) for b in ranked[:n]]                      # Return the top-n band indices

    def predict_with_uncertainty(
        self,
        cube:     torch.Tensor,             # (H, W_sp, W_wv) float32 in [0,1]
        obs_mask: torch.Tensor,             # (H, W_sp) bool — currently observed pixels
        image_band: Optional[int] = None,   # dynamic mode: which band is imaged
        n_images:   Optional[int] = None,   # multi-image: how many bands to image
        image_bands: Optional[Sequence[int]] = None,  # multi-image: explicit bands
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct a single cube from its observed pixels.

        In dynamic-image mode, `image_band` selects the imaged wavenumber. If not
        given, it is chosen the same way as in training: probe a random pixel, find
        its peaks (height=image_peak_height), pick one (argmax fallback).

        Returns:
            mu    : (H, W_sp, W_wv) predicted mean spectra
            sigma : (H, W_sp, W_wv) predictive standard deviation
        """
        self.eval()                                           # Switch to eval mode
        device = next(self.parameters()).device               # Device the model lives on
        cube   = cube.to(device).unsqueeze(0)                 # Reformat cube (1,H,Wsp,Wv)
        mask   = obs_mask.to(device).unsqueeze(0)             # Reformat mask (1,H,Wsp)
        ctx    = self._build_ctx_grid(cube, mask)             # Reformat ctx (1,Wv,H,Wsp)

        img_valid = None                                      # Default: no image padding mask
        if self.multi_image:                                  # Multi-image path
            # Replicate the training-time selection on this cube's context
            sel = ([int(b) for b in image_bands] if image_bands is not None
                   else self.select_image_bands(cube[0], mask[0],
                                                n_images=(n_images or 1)))  # Explicit or selected
            bands_t = torch.tensor(sel, device=device, dtype=torch.long).view(1, -1)  # (1,N)
            N = bands_t.shape[1]                                 # Number of imaged bands
            chw = cube.permute(0, 3, 1, 2)                       # (1,Wv,H,Wsp)
            idx = bands_t.view(1, N, 1, 1).expand(1, N, cube.shape[1], cube.shape[2])  # Broadcast idx
            img_planes = torch.gather(chw, 1, idx)               # (1,N,H,Wsp)
            img_bands  = bands_t                                 # Band indices to pass to the model
            img_valid  = torch.ones(1, N, dtype=torch.bool, device=device)  # All slots valid
        elif self.dynamic_image_band:                            # Dynamic single-band path
            if image_band is None:                               # match training selection
                H, Wsp, Wv = cube.shape[1], cube.shape[2], cube.shape[3] # Unpack shape
                spec = cube[0, H // 2, Wsp // 2].detach().cpu().numpy()  # Probe the centre pixel
                pk, _ = find_peaks(spec, height=self.image_peak_height)  # Find its peaks
                image_band = int(pk[len(pk) // 2]) if len(pk) > 0 else int(spec.argmax())  # Pick one
            bands = torch.tensor([int(image_band)], device=device, dtype=torch.long)  # As a tensor
            img_planes, img_bands = self._image_planes(cube, bands)  # Extract that plane
        else:                                                        # Fixed-band / no-image path
            img_planes, img_bands = self._image_planes(cube)     # Extract fixed planes if any
        mu, sigma = self.model(mask, ctx, image_planes=img_planes, image_bands=img_bands,
                               image_valid=img_valid)            # Forward pass
        mu    = mu.squeeze(0).permute(1, 2, 0).contiguous()      # (H,Wsp,Wv)
        sigma = sigma.squeeze(0).permute(1, 2, 0).contiguous()   # (H,Wsp,Wv)
        return mu, sigma                                         # Return predictive mean and sigma

    @torch.no_grad()
    def acquisition_map(
        self,
        cube:      torch.Tensor,   # (H, W_sp, W_wv)
        obs_mask:  torch.Tensor,   # (H, W_sp) bool
        reduce:    str = "mean",   # "mean" or "max" over wavenumbers
        mask_observed: bool = True,
    ) -> torch.Tensor:
        """
        Per-pixel acquisition score. Used for greedy adaptive sampling.

        Reduces predictive sigma over the wavenumber axis to a single number per
        pixel, then (optionally) sets already-observed pixels to -inf so argmax
        never re-selects them.  Measure next at argmax of this map.

        Returns:
            acq : (H, W_sp) float
        """
        _, sigma = self.predict_with_uncertainty(cube, obs_mask)   # (H,Wsp,Wv)
        if reduce == "max":                                        # Max-over-bands reduction
            acq = sigma.max(dim=-1).values                         # Per-pixel max sigma
        else:                                                      # Mean-over-bands reduction
            acq = sigma.mean(dim=-1)                               # Per-pixel mean sigma
        if mask_observed:                                          # Exclude already-measured pixels
            acq = acq.masked_fill(obs_mask.to(acq.device), float("-inf"))  # Make them unpickable
        return acq                                                 # Return the acquisition map

    def configure_optimizers(self):                                # Lightning optimizer/scheduler hook
        # Only optimize parameters that require grad. In sigma_nll the
        # mean pathway is frozen, so this is just the sigma_decoder.
        trainable = [p for p in self.parameters() if p.requires_grad]   # Unfrozen parameters only
        opt = torch.optim.AdamW(trainable,
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay) # AdamW optimizer

        total_steps  = self.trainer.estimated_stepping_batches          # Total optimizer steps planned
        warmup_steps = int(self.hparams.lr_warmup_epochs
                           * (total_steps // max(1, self.hparams.max_epochs)))  # Steps in warmup

        def lr_lambda(step):                                            # LR multiplier as a function of step
            if step < warmup_steps:                                 # Still warming up
                return float(step) / max(1, warmup_steps)           # Linear ramp from 0 to 1
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)  # Post-warmup fraction
            return 0.5 * (1.0 + math.cos(math.pi * progress))       # Cosine decay to 0

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)  # Wrap it as a scheduler
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}  # Step-wise scheduling

    def count_parameters(self) -> int:                  # Count trainable parameters
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # Sum unfrozen sizes


# =============================================================================
# 5.  Version + logger helpers  (unchanged)
# =============================================================================

def _next_version(results_dir: str, prefix: str = "conv_cnp_prob_version_") -> int:
    """Makes directory for saving model outputs"""
    os.makedirs(results_dir, exist_ok=True)
    existing = [d for d in os.listdir(results_dir)
                if d.startswith(prefix) and
                os.path.isdir(os.path.join(results_dir, d))]
    if not existing:
        return 0
    nums = []
    for d in existing:
        try:
            nums.append(int(d[len(prefix):]))
        except ValueError:
            pass
    return (max(nums) + 1) if nums else 0


def _make_logger(results_dir: str, version: int):
    """Sets up logger"""
    try:
        import tensorboard  # noqa: F401
        return pl.loggers.TensorBoardLogger(
            save_dir = results_dir,
            name     = f"conv_cnp_prob_version_{version}",
            version  = 0,
        )
    except ModuleNotFoundError:
        pass
    try:
        import tensorboardX  # noqa: F401
        return pl.loggers.TensorBoardLogger(
            save_dir = results_dir,
            name     = f"conv_cnp_prob_version_{version}",
            version  = 0,
        )
    except ModuleNotFoundError:
        pass
    print("  [logger] tensorboard not found — falling back to CSVLogger")
    return pl.loggers.CSVLogger(
        save_dir = results_dir,
        name     = f"conv_cnp_prob_version_{version}",
        version  = 0,
    )


# =============================================================================
# 6.  Training function
# =============================================================================

def train(args):

    # Initiate random seed
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    print("\n[1/5] Loading data...")

    # Check if we are using on-the-fly generation (should be True)
    use_onthefly = bool(getattr(args, "onthefly", False))

    if use_onthefly:
        # Generate cubes on-the-fly

        # Import necessary libraries
        from synth_onthefly import (
            OnTheFlySynthDataset,
            make_worker_init_fn, load_spectrum_library,
        )

        # Check all of the arguments are correct
        if not args.spectrum_library:
            raise ValueError("--onthefly requires --spectrum_library (an (N, W) .npy)")
        if not args.wavenumbers:
            raise ValueError("--onthefly requires --wavenumbers (a (W,) .npy on the "
                             "same grid as the spectrum library)")


        spectrum_library = load_spectrum_library(args.spectrum_library) # Load spectrum library
        wavenumbers = np.load(args.wavenumbers).astype(np.float32)      # Load wavenumber array
        W = int(spectrum_library.shape[1])                              # Get the length of wavenumber array

        # Assemble synthetic data generation ranges from CLI (logged into hparams below)
        # Check documentation for details on all of these inputs. 
        gen_params = dict(
            n_phases              = list(args.gen_n_phases),
            feature_size          = list(args.gen_feature_size),
            boundary_curvature    = list(args.gen_boundary_curvature),
            phase_fraction        = list(args.gen_phase_fraction),
            morphology_types      = list(args.gen_morphology_types),
            boundary_width        = list(args.gen_boundary_width),
            mixing_types          = list(args.gen_mixing_types),
            depth_edge_mix        = list(args.gen_depth_edge_mix),
            depth_decay_length    = list(args.gen_depth_decay_length),
            depth_probing_depth   = (list(args.gen_depth_probing_depth)
                                     if args.gen_depth_probing_depth is not None else None),
            depth_thickness_range = list(args.gen_depth_thickness_range),
            depth_thickness_profile = args.gen_depth_thickness_profile,
            relative_intensity    = (list(args.gen_relative_intensity)
                                     if args.gen_relative_intensity is not None else None),
            clip_output           = (not args.gen_no_clip),
            noise_std             = list(args.gen_noise_std),
            intra_phase_variation = list(args.gen_intra_phase_variation),
            spectra_shift_std     = float(args.gen_spectra_shift_std),
            spectra_width_jitter  = float(args.gen_spectra_width_jitter),
            spectra_amp_jitter    = float(args.gen_spectra_amp_jitter),
        )

        # Print some of the generation parameters in the log file
        print(f"  On-the-fly generation ENABLED")
        print(f"  Spectrum library: {spectrum_library.shape} (N spectra x W) from "
              f"{args.spectrum_library}")
        print(f"  Image size: {args.image_size[0]} x {args.image_size[1]}")
        print(f"  Wavenumber channels (W): {W}")
        print(f"  Generation ranges:")
        for k, v in gen_params.items():
            print(f"      {k:22s} {v}")
        cube_data_list = None  # not used in this path
    else:
        # Use pregenerated training data

        if args.data_dir:
            # Load pregenerating LIBRARY of hyperspectral cubes
            cube_data_list, wavenumbers = load_cube_library(
                args.data_dir, wavenumbers_path=args.wavenumbers
            )
        else:
            # Train on SINGLE HYPERSPECTRAL CUBE (only use for debugging)
            cube = load_cube(args.data_path, args.h5_key)
            cube_data_list = [cube]
            wavenumbers    = None
            if args.wavenumbers:
                wavenumbers = np.load(args.wavenumbers).astype(np.float32)

        W = int(cube_data_list[0].shape[2])
        print(f"  Spatial grid: {cube_data_list[0].shape[0]} x {cube_data_list[0].shape[1]}")
        print(f"  Wavenumber channels (W): {W}")
        gen_params = None   # not used in the fixed-library path

    print("[2/5] Building datasets...")
    n_val_tasks = max(50, args.n_tasks_per_epoch // 10)

    #--------------------------------------------------
    # Create trainign and validation datasets
    #--------------------------------------------------
    if use_onthefly:
        # Creat training dataset
        train_ds = OnTheFlySynthDataset(
            spectrum_library    = spectrum_library,
            wavenumbers         = wavenumbers,
            image_size          = tuple(args.image_size),
            n_tasks_per_epoch   = args.n_tasks_per_epoch,
            param_ranges        = gen_params,
            norm_mode           = args.norm_mode,
            n_context           = args.n_context,
            n_context_min       = args.n_context_min,
            context_dist        = args.context_dist,
            mode_frac           = args.mode_frac,
            tail_frac           = args.tail_frac,
            tail_quantile       = args.tail_quantile,
            seed                = args.seed,
            split               = "train",
            dynamic_image_band  = args.dynamic_image_band,
            image_peak_height   = args.image_peak_height,
            multi_image         = args.multi_image,
            texture_prob        = args.gen_texture_prob,
            texture_sigma       = args.gen_texture_sigma,
            texture_amp         = args.gen_texture_amp,
            texture_clip        = args.gen_texture_clip,
            texture_scales      = args.gen_texture_scales,
            scratch_prob        = args.gen_scratch_prob,
            scratch_count       = args.gen_scratch_count,
            outlier_n_range     = tuple(args.gen_outlier_n),
            outlier_boost_frac  = args.gen_outlier_boost_frac,
            outlier_peak_boost  = tuple(args.gen_outlier_peak_boost),
            outlier_shift_max   = args.gen_outlier_shift_max,
            scratch_width       = args.gen_scratch_width,
            scratch_depth       = args.gen_scratch_depth,
            context_mode        = args.context_mode,
            context_mix         = args.context_mix,
            adaptive_tau        = args.adaptive_tau,
            adaptive_noise_floor= args.adaptive_noise_floor,
            image_var_thresh    = args.image_var_thresh,
            image_count_weights = args.image_count_weights,
            max_images          = args.max_images,
        )

        # Create validation dataset
        val_ds = OnTheFlySynthDataset(
            spectrum_library    = spectrum_library,
            wavenumbers         = wavenumbers,
            image_size          = tuple(args.image_size),
            n_tasks_per_epoch   = n_val_tasks,
            param_ranges        = gen_params,
            norm_mode           = args.norm_mode,
            n_context           = args.n_context,
            n_context_min       = args.n_context_min,
            context_dist        = args.context_dist,
            mode_frac           = args.mode_frac,
            tail_frac           = args.tail_frac,
            tail_quantile       = args.tail_quantile,
            seed                = args.seed + 1,
            split               = "val",      # fixed per-index seeds -> stable val set
            dynamic_image_band  = args.dynamic_image_band,
            image_peak_height   = args.image_peak_height,
            multi_image         = args.multi_image,
            texture_prob        = args.gen_texture_prob,
            texture_sigma       = args.gen_texture_sigma,
            texture_amp         = args.gen_texture_amp,
            texture_clip        = args.gen_texture_clip,
            texture_scales      = args.gen_texture_scales,
            scratch_prob        = args.gen_scratch_prob,
            scratch_count       = args.gen_scratch_count,
            outlier_n_range     = tuple(args.gen_outlier_n),
            outlier_boost_frac  = args.gen_outlier_boost_frac,
            outlier_peak_boost  = tuple(args.gen_outlier_peak_boost),
            outlier_shift_max   = args.gen_outlier_shift_max,
            scratch_width       = args.gen_scratch_width,
            scratch_depth       = args.gen_scratch_depth,
            context_mode        = args.context_mode,
            context_mix         = args.context_mix,
            adaptive_tau        = args.adaptive_tau,
            adaptive_noise_floor= args.adaptive_noise_floor,
            image_var_thresh    = args.image_var_thresh,
            image_count_weights = args.image_count_weights,
            max_images          = args.max_images,
        )
        worker_init = make_worker_init_fn(args.seed)
    else:
        # Create training dataset from pregenerated library
        train_ds = GridCNPDataset(
            cube_data_list      = cube_data_list,
            n_context           = args.n_context,
            n_context_min       = args.n_context_min,
            n_tasks_per_epoch   = args.n_tasks_per_epoch,
            val_fraction        = args.val_fraction,
            split               = "train",
            seed                = args.seed,
            context_dist        = args.context_dist,
            mode_frac           = args.mode_frac,
            tail_frac           = args.tail_frac,
            tail_quantile       = args.tail_quantile,
            norm_mode           = args.norm_mode,
        )
        # Create training dataset from pregenerated library
        val_ds = GridCNPDataset(
            cube_data_list      = cube_data_list,
            n_context           = args.n_context,
            n_context_min       = args.n_context_min,
            n_tasks_per_epoch   = n_val_tasks,
            val_fraction        = args.val_fraction,
            split               = "val",
            seed                = args.seed + 1,
            context_dist        = args.context_dist,
            mode_frac           = args.mode_frac,
            tail_frac           = args.tail_frac,
            tail_quantile       = args.tail_quantile,
            norm_mode           = args.norm_mode,
        )
        worker_init = None

    # Initialize training and validation dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=grid_cnp_collate,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init,
        persistent_workers=False,
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=grid_cnp_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=(4 if args.num_workers > 0 else None),
    )
    print(f"  Train tasks/epoch: {len(train_ds)}  |  Val tasks: {len(val_ds)}")

    # Determine encoder channels
    print("[3/5] Building model...")
    enc_channels = args.enc_channels if args.enc_channels else [
        args.base_channels,
        args.base_channels * 2,
        args.base_channels * 4,
        args.base_channels * 8,
    ]

    # Used if Gaussian smearing context pixels (should be None in the case of NN-filling input)
    scales = args.scales if args.scales else None

    # Initialize lightning module (includes the model itself)
    model = CACNPLightningModule(
        W                = W,
        base_channels    = args.base_channels,
        enc_channels     = enc_channels,
        kernel_sigma     = args.kernel_sigma,
        kernel_size      = args.kernel_size,
        dropout          = args.dropout,
        sigma_min        = args.sigma_min,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        lr_warmup_epochs = args.lr_warmup_epochs,
        max_epochs       = args.max_epochs,
        norm_mask        = args.norm_mask,
        scales           = scales,
        spectral_mlp     = args.spectral_mlp,
        mlp_hidden       = args.mlp_hidden,
        lambda_ctx       = args.lambda_ctx,
        norm_smear       = args.norm_smear,
        norm_eps         = args.norm_eps,
        mean_activation  = args.mean_activation,
        sig_margin       = args.sig_margin,
        lambda_mse       = args.lambda_mse,
        lambda_img       = args.lambda_img,
        lambda_grad      = args.lambda_grad,
        lambda_grad_full = args.lambda_grad_full,
        lambda_tv        = args.lambda_tv,
        tv_eps           = args.tv_eps,
        mse_only         = args.mse_only,
        train_mode       = args.train_mode,
        init_from        = args.init_from,
        gen_params       = gen_params,
        onthefly         = use_onthefly,
        norm_mode        = args.norm_mode,
        input_mode       = args.input_mode,
        image_bands      = args.image_bands,
        dynamic_image_band = args.dynamic_image_band,
        image_peak_height  = args.image_peak_height,
        image_tokens     = args.image_tokens,
        image_subsample  = args.image_subsample,
        dec0_channels    = args.dec0_channels,
        guidance_agg     = args.guidance_agg,
        max_images       = args.max_images,
        guidance_raw     = args.guidance_raw,
        decoder_skip     = args.decoder_skip,
        image_guidance   = args.image_guidance,
        film             = args.film,
        multi_image      = args.multi_image,
        image_var_thresh = args.image_var_thresh,
        use_attention    = args.use_attention,
        attn_stages      = args.attn_stages,
        attn_heads       = args.attn_heads,
        attn_dim         = args.attn_dim,
        attn_pos_bands   = args.attn_pos_bands,
        head_kernel      = args.head_kernel,
        head_attn        = args.head_attn,
        head_attn_dim    = args.head_attn_dim,
        head_attn_heads  = args.head_attn_heads,
        adaptive_context      = args.adaptive_context,
        adaptive_points_per_step = args.adaptive_points_per_step,
        adaptive_start_frac_min = args.adaptive_start_frac_min,
        adaptive_start_frac_max = args.adaptive_start_frac_max,
        adaptive_acq_reduce   = args.adaptive_acq_reduce,
        adaptive_refresh_epochs  = args.adaptive_refresh_epochs,
    )

    # Print some model parameters to the log file
    print(f"  Parameters:           {model.count_parameters():,}")
    print(f"  U-Net enc channels:   {enc_channels}")
    print(f"  Output head:          2*W = {2 * W} channels (mean | scale)")
    print(f"  sigma floor:          {args.sigma_min}")
    if scales and len(scales) > 1:
        print(f"  Smear:                MULTI-SCALE sigmas (init) = {scales} px")
    else:
        print(f"  Smear sigma (init):   {args.kernel_sigma} px  (kernel_size={args.kernel_size})")
    print(f"  spectral_in MLP:      {args.spectral_mlp}  (hidden={args.mlp_hidden})")
    print(f"  input_mode:           {args.input_mode}")
    if args.image_bands:
        print(f"  fixed image bands:    {args.image_bands}  "
              f"(input = [value | per-band coverage] = 2W, fixed width)")
    if args.dynamic_image_band:
        print(f"  dynamic image band:   ON  (per-cube random peak, "
              f"find_peaks height={args.image_peak_height}; input = 2W)")
    _stages = args.attn_stages if args.attn_stages is not None else (["bottleneck"] if args.use_attention else [])
    if _stages:
        print(f"  attention stages:     {_stages}  (heads={args.attn_heads}, "
              f"d_model={args.attn_dim}, d_head={args.attn_dim // args.attn_heads}, "
              f"pos_bands={args.attn_pos_bands}; gathered-voxel keys)")
    else:
        print(f"  attention stages:     none")
    print(f"  norm_mask:            {args.norm_mask}")
    print(f"  norm_smear (soft):    {args.norm_smear}  (eps={args.norm_eps})")
    print(f"  lambda_mse:           {args.lambda_mse}  (0 = pure NLL)")
    print(f"  lambda_img:           {args.lambda_img}  (structure-image level consistency)")
    print(f"  lambda_grad:          {args.lambda_grad}  (structure-image gradient/edge consistency)")
    print(f"  lambda_grad_full:     {args.lambda_grad_full}  (ALL-BAND gradient consistency vs ground truth)")
    print(f"  lambda_tv:            {args.lambda_tv}  (isotropic spatial TV on mu)")
    print(f"  head_kernel:          {args.head_kernel}  (output-head spatial kernel; 1 = per-pixel)")
    print(f"  head_attn:            {args.head_attn}  (content-only SDPA attention at the head)")
    print(f"  mean_activation:      {args.mean_activation}  (margin={args.sig_margin})")
    print(f"  norm_mode:            {args.norm_mode}")
    print(f"  train_mode:           {model.train_mode}  (init_from={args.init_from})")
    print(f"  trainable params:     {model.count_parameters():,}")

    # Make sure to save the model training results in a new version folder
    version = _next_version(args.results_dir)
    out_dir = os.path.join(args.results_dir, f"conv_cnp_prob_version_{version}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[4/5] Output directory: {out_dir}")

    # In MSE-only mode val/nll is not the objective and sigma is untrained, so
    # monitor the reconstruction MSE instead for both checkpointing and early stop.
    monitor_metric = "val/recon_mse" if model.train_mode == "mean_mse" else "val/nll"
    print(f"  monitoring:           {monitor_metric}")

    # Setup model callbacks for monitoring training parameters. 
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath   = out_dir,
            filename  = "best",
            monitor   = monitor_metric,
            mode      = "min",
            save_last = True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]
    # Is early stopping being used?
    if not getattr(args, "no_early_stop", False):
        callbacks.insert(1, pl.callbacks.EarlyStopping(
            monitor  = monitor_metric,
            patience = args.patience,
            mode     = "min",
        ))

    # On-the-fly: Ensure per-epoch seed so that new cubes are generated each epoch
    if use_onthefly:
        class _SetEpoch(pl.Callback):
            def __init__(self, ds): self.ds = ds
            def on_train_epoch_start(self, trainer, pl_module):
                if hasattr(self.ds, "set_epoch"):
                    self.ds.set_epoch(trainer.current_epoch)
        callbacks.append(_SetEpoch(train_ds))

    # Lightning 1.x has no strategy="auto"; None is the single-device default there
    # and is accepted by 2.x as well. Map the portable strings to None so scripts
    # that pass --strategy auto/none work on either version. Real strategy names
    # (e.g. "ddp") pass through unchanged.
    _strategy = args.strategy
    if isinstance(_strategy, str) and _strategy.strip().lower() in ("", "auto", "none"):
        _strategy = None

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs          = args.max_epochs,
        accelerator         = "auto",
        devices             = args.devices,
        num_nodes           = args.num_nodes,
        strategy            = _strategy,
        log_every_n_steps   = max(1, len(train_loader) // 5),
        callbacks           = callbacks,
        logger              = _make_logger(args.results_dir, version),
        enable_progress_bar = True,
        precision           = (16 if (torch.cuda.is_available() and args.amp) else 32),
    )

    # If resuming checkpoint, let user know
    resume_ckpt = getattr(args, "resume_from", None)
    if resume_ckpt:
        print(f"\n[5/5] RESUMING training from {resume_ckpt}  "
              f"(max {args.max_epochs} epochs, patience {args.patience})")
    else:
        print(f"\n[5/5] Training  (max {args.max_epochs} epochs, patience {args.patience})")
    
    # TRAIN THE MODEL
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

    # Save best path at best.ckpt
    best_path = os.path.join(out_dir, "best.ckpt")
    print(f"\nBest checkpoint: {best_path}")
    return best_path


# =============================================================================
# 7.  CLI 
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a probabilistic ConvCNP for AFM-IR cube reconstruction "
                    "with calibrated uncertainty for adaptive sampling"
    )

    g = p.add_argument_group("Data")
    g.add_argument("--data_dir",     type=str,   default=None,
                   help="Directory from generate_synth_afmir.py (mutually exclusive with --data_path)")
    g.add_argument("--data_path",    type=str,   default=None,
                   help="Single HDF5 or .npy cube file")
    g.add_argument("--h5_key",       type=str,   default="data")
    g.add_argument("--wavenumbers",  type=str,   default=None)
    g.add_argument("--val_fraction", type=float, default=0.10)

    # ---- on-the-fly synthetic generation (Stage 3: diverse spectra) ----
    g.add_argument("--onthefly", action="store_true",
                   help="Generate synthetic cubes on the fly each batch (morphology "
                        "generated live; phase spectra sampled from --spectrum_library) "
                        "instead of loading a fixed --data_dir library.")
    g.add_argument("--spectrum_library", type=str, default=None,
                   help="Path to a pre-normalized (N, W) .npy matrix of spectra "
                        "(e.g. NIST). Required with --onthefly. W must match the "
                        "wavenumber grid in --wavenumbers.")
    g.add_argument("--image_size", type=int, nargs=2, default=[128, 128],
                   metavar=("X", "Y"),
                   help="Spatial size of on-the-fly generated cubes.")

    g = p.add_argument_group("Generation ranges (on-the-fly only)")
    g.add_argument("--gen_n_phases", type=int, nargs=2, default=[2, 3],
                   metavar=("MIN", "MAX"),
                   help="Min/max number of phases (including background).")
    g.add_argument("--gen_feature_size", type=float, nargs=2, default=[4.0, 40.0],
                   metavar=("MIN", "MAX"), help="Region feature size range (px).")
    g.add_argument("--gen_boundary_curvature", type=float, nargs=2, default=[0.4, 1.0],
                   metavar=("MIN", "MAX"),
                   help="Boundary curvature range (1.0=round, lower=elongated).")
    g.add_argument("--gen_phase_fraction", type=float, nargs=2, default=[0.1, 0.4],
                   metavar=("MIN", "MAX"), help="Area fraction per non-bg phase.")
    g.add_argument("--gen_morphology_types", type=str, nargs="+",
                   default=["dispersed", "voronoi"],
                   choices=["dispersed", "voronoi"],
                   help="Morphology types to sample from per cube.")
    g.add_argument("--gen_boundary_width", type=float, nargs=2, default=[1.5, 2.0],
                   metavar=("MIN", "MAX"), help="Interface mixing width range (px).")
    g.add_argument("--gen_mixing_types", type=str, nargs="+", default=["linear"],
                   help="Boundary mixing model(s) to sample from.")
    g.add_argument("--gen_depth_edge_mix", type=float, nargs=2, default=[0.0, 0.0],
                   metavar=("MIN", "MAX"),
                   help="DEPTH (subsurface) SPECTRAL MIXING: range for the maximum "
                        "background fraction mixed in at a particle boundary, sampled "
                        "uniformly per cube. Emulates the finite AFM-IR probing depth: "
                        "near a particle edge (thin material, subsurface background "
                        "close below) a fraction of the spectrum beneath is mixed in, "
                        "decaying exponentially toward the particle interior. Default "
                        "[0.0, 0.0] DISABLES depth mixing (backward compatible); the "
                        "generator no-ops when edge_mix<=0. Requires --input_mode "
                        "nn_fill. Pair with --gen_depth_decay_length.")
    g.add_argument("--gen_depth_decay_length", type=float, nargs=2, default=[6.0, 6.0],
                   metavar=("MIN", "MAX"),
                   help="DEPTH MIXING decay length range (px), sampled uniformly per "
                        "cube: alpha_bg(d) = edge_mix * exp(-d / decay_length), where d "
                        "is depth into the particle from its boundary. Larger = the "
                        "subsurface contribution reaches deeper. Only active when "
                        "--gen_depth_edge_mix is non-zero. Background phase is fixed at "
                        "0 on the on-the-fly path.")
    g.add_argument("--gen_depth_probing_depth", type=float, nargs=2, default=None,
                   metavar=("MIN", "MAX"),
                   help="PHYSICAL depth mixing: probing depth p (nm) range, sampled "
                        "per cube. Setting this ENABLES the Beer-Lambert model "
                        "(fraction from beneath = exp(-thickness/p)) and supersedes "
                        "the legacy edge model. e.g. 100 500. Unset = off.")
    g.add_argument("--gen_depth_thickness_range", type=float, nargs=2, default=[200.0, 500.0],
                   metavar=("MIN", "MAX"),
                   help="PHYSICAL depth: per-particle peak thickness (nm) range. The "
                        "thickness/probing_depth ratio sets opacity. Used only with "
                        "--gen_depth_probing_depth.")
    g.add_argument("--gen_depth_thickness_profile", type=str, default="sqrt",
                   choices=["sqrt", "linear"],
                   help="PHYSICAL depth thickness profile: 'sqrt' (puddle) or 'linear'.")
    g.add_argument("--gen_relative_intensity", type=float, nargs=2, default=None,
                   metavar=("MIN", "MAX"),
                   help="Per-phase RELATIVE INTENSITY (max peak height) range, sampled "
                        "per phase per cube. e.g. 0.5 1.0. Unset = all phases at 1.0.")
    g.add_argument("--gen_no_clip", action="store_true",
                   help="Disable the [0,1] clip in generation (only needed if "
                        "--gen_relative_intensity goes above 1.0).")
    g.add_argument("--gen_noise_std", type=float, nargs=2, default=[0.032, 0.032],
                   metavar=("MIN", "MAX"),
                   help="Gaussian noise std range (normalized units). Widen "
                        "(e.g. 0.0 0.06) to train across noise levels.")
    g.add_argument("--gen_intra_phase_variation", type=float, nargs=2, default=[0.0, 0.0],
                   metavar=("MIN", "MAX"),
                   help="Within-region amplitude variation range.")
    g.add_argument("--gen_spectra_shift_std", type=float, default=0.0,
                   help="Spectral shift jitter std (keep 0 for pre-normalized NIST).")
    g.add_argument("--gen_spectra_width_jitter", type=float, default=0.0,
                   help="Spectral width jitter (keep 0 for pre-normalized NIST).")
    g.add_argument("--gen_spectra_amp_jitter", type=float, default=0.0,
                   help="Spectral amplitude jitter (keep 0 for pre-normalized NIST).")

    g = p.add_argument_group("Context sampling")
    g.add_argument("--n_context",           type=int,  default=2000,
                   help="Maximum context pixels per task")
    g.add_argument("--n_context_min",       type=int,  default=10,
                   help="Minimum context pixels per task (keep small to train sparse-context uncertainty)")
    g.add_argument("--context_dist", type=str, default="log_uni",
                   choices=["uni", "log_uni", "log_norm"],
                   help="Sample context size uniformly, log uniformly, or log normally.")
    g.add_argument("--mode_frac", type=float, default=0.05,
                   help="Location of distribution mode in log normal context size sampling.")
    g.add_argument("--tail_frac", type=float, default=0.20,
                   help="Location of lower tail in log normal context size sampling.")
    g.add_argument("--tail_quantile", type=float, default=0.99,
                   help="Location of rightward tail for log normal context size sampling.")
    g.add_argument("--norm_mode", type=str, default="cube_minmax",
                   choices=list(NORM_MODES),
                   help="Input normalization scheme: "
                        "cube_minmax (whole-cube [0,1], needs full cube; pairs "
                        "with sigmoid/scaled_sigmoid). "
                        "context_minmax ([0,1] from observed pixels only; matches "
                        "deployment; pairs with scaled_sigmoid). "
                        "context_zscore ((x-mean)/std from observed pixels; robust "
                        "to baseline offset / mV floor; UNBOUNDED -> use with "
                        "--mean_activation none).")

    g = p.add_argument_group("Model architecture")
    g.add_argument("--input_mode", type=str, default="smear",
                   choices=["smear", "nn_fill"],
                   help="Encoder front end. 'smear': multi-scale Gaussian set-conv "
                        "(learnable radii). 'nn_fill': nearest-observation Voronoi "
                        "fill (clean, unaveraged spectra) + distance-transform channel; "
                        "no learnable kernel radii. With nn_fill, --scales / "
                        "--norm_smear / --kernel_* are ignored.")
    g.add_argument("--base_channels", type=int,   default=64,
                   help="Spectral latent dim = first U-Net channel count")
    g.add_argument("--enc_channels",  type=int,   nargs=4, default=None,
                   help="4 U-Net encoder channel widths [C0 C1 C2 C3] "
                        "(default: base_channels x [1, 2, 4, 8])")
    g.add_argument("--kernel_sigma",  type=float, default=1.5,
                   help="Initial Gaussian smear bandwidth in pixels (learned)")
    g.add_argument("--kernel_size",   type=int,   default=9,
                   help="Separable Gaussian kernel size (odd, e.g. 9 or 15)")
    g.add_argument("--dropout",       type=float, default=0.05)
    g.add_argument("--sigma_min",     type=float, default=1e-3,
                   help="Lower floor on predictive sigma for numerical stability")
    g.add_argument("--norm_mask", action="store_true",
                   help="Normalize spectral context by smeared mask (recommended)")
    g.add_argument("--scales", type=float, nargs="+", default=None,
                   help="Multi-scale smear bandwidths in px, e.g. --scales 1 4 16. "
                        "Each scale's sigma is learned. If omitted or single value, "
                        "uses the single-scale GaussianSmear with --kernel_sigma.")
    g.add_argument("--spectral_mlp", action="store_true",
                   help="Use a two-layer 1x1 MLP spectral encoder (in -> mlp_hidden "
                        "-> base_channels) instead of a single 1x1 conv. Recommended "
                        "with multi-scale, since the input is S*(W+1) channels wide.")
    g.add_argument("--mlp_hidden", type=int, default=256,
                   help="Hidden width of the spectral_in MLP (if --spectral_mlp).")
    g.add_argument("--use_attention", action="store_true",
                   help="Enable residual cross-attention. With no --attn_stages, "
                        "defaults to the bottleneck only. Keys/values are the "
                        "gathered observed voxels (clean spectra + positions). "
                        "Zero-init residual so init == no-attention model; frozen "
                        "with the mean pathway in Stage-2 sigma_nll.")
    g.add_argument("--dec0_channels", type=int, default=None,
                   help="Width of the FINAL decoder block = the per-pixel latent the "
                        "output heads reconstruct all W wavenumbers from. Defaults to "
                        "--base_channels. Raising ONLY this widens spectral capacity "
                        "without touching the encoder or the spatial pathway.")
    g.add_argument("--gen_outlier_n", type=int, nargs=2, default=[1, 20],
                   metavar=("MIN", "MAX"),
                   help="ANOMALOUS pixels per cube (uniform, inclusive). Each is a "
                        "tall-peak outlier or a rigid spectral shift (see "
                        "--gen_outlier_boost_frac). Default 1 20 (ON). 0 0 disables.")
    g.add_argument("--gen_outlier_boost_frac", type=float, default=0.5,
                   help="Per anomalous pixel, P(tall-peak outlier) else rigid shift. Default 0.5.")
    g.add_argument("--gen_outlier_peak_boost", type=float, nargs=2, default=[1.5, 3.5],
                   metavar=("MIN", "MAX"),
                   help="Multiplicative boost of a tall-peak outlier's 1-2 tallest peaks "
                        "(exceeds 1; applied after the [0,1] clip).")
    g.add_argument("--gen_outlier_shift_max", type=int, default=1,
                   help="Max |roll| (band indices) for the rigid-shift outlier. Default 1.")
    g.add_argument("--gen_texture_prob", type=float, default=0.0,
                   help="Fraction of cubes given spatially correlated within-region "
                        "intensity texture (multiplicative, broadband, independent "
                        "per phase). 1.0 = every cube.")
    g.add_argument("--gen_texture_sigma", type=float, nargs=2, default=[3.0, 15.0],
                   help="Gaussian blur sigma range (px) for each texture scale.")
    g.add_argument("--gen_texture_amp", type=float, nargs=2, default=[0.02, 0.06],
                   help="Amplitude range per texture scale.")
    g.add_argument("--gen_texture_clip", type=float, default=0.10,
                   help="Hard cap on the texture field, so spectral values never "
                        "change by more than this fraction.")
    g.add_argument("--gen_texture_scales", type=int, default=2,
                   help="Number of blur scales summed per phase (multi-scale).")
    g.add_argument("--gen_scratch_prob", type=float, default=0.0,
                   help="Fraction of cubes given scratches (broadband multiplicative "
                        "linear defects).")
    g.add_argument("--gen_scratch_count", type=int, nargs=2, default=[1, 4],
                   help="Scratches per affected cube.")
    g.add_argument("--gen_scratch_width", type=float, nargs=2, default=[0.8, 2.5],
                   help="Scratch Gaussian half-width in px.")
    g.add_argument("--gen_scratch_depth", type=float, nargs=2, default=[0.20, 0.60],
                   help="Fractional attenuation at the scratch centre.")
    g.add_argument("--context_mode", type=str, default="random",
                   choices=["random", "hex", "adaptive", "mixture"],
                   help="What KIND of context the model trains on. 'random' = uniform "
                        "(original). 'hex' = hexagonal grid (the experiment's opening "
                        "move). 'adaptive' = hex + points added the way the acquisition "
                        "loop adds them (image-aware GT oracle, zero forward passes). "
                        "'mixture' = draw one per cube (see --context_mix). Deployed "
                        "contexts are hex-then-clustered, NOT uniform-random.")
    g.add_argument("--context_mix", type=float, nargs=3, default=[0.40, 0.20, 0.40],
                   metavar=("P_RANDOM", "P_HEX", "P_ADAPTIVE"),
                   help="Per-cube probabilities for --context_mode mixture.")
    g.add_argument("--adaptive_tau", type=float, default=0.064,
                   help="Two phases are indistinguishable in the structure image if "
                        "their endmember intensities differ by less than this. Should "
                        "track the noise (~2 x gen_noise_std).")
    g.add_argument("--adaptive_noise_floor", type=float, default=0.0,
                   help="Subtract this from the oracle error so the sampler does not "
                        "chase pixels that are merely NOISY rather than uncertain.")
    g.add_argument("--multi_image", action="store_true",
                   help="Multi-image training: per cube, rank bands by variance of "
                        "the SAMPLED spectra (find_peaks), image the top-N highest-"
                        "variance peaks, N ~ weighted(1..max_images).")
    g.add_argument("--image_var_thresh", type=float, default=0.2,
                   help="find_peaks height threshold on normalised band variance.")
    g.add_argument("--image_count_weights", type=float, nargs="+",
                   default=[0.6, 0.25, 0.10, 0.05],
                   help="Probabilities for imaging 1,2,3,4 bands (weighted to 1).")
    g.add_argument("--max_images", type=int, default=4)
    g.add_argument("--guidance_agg", type=str, default="sum",
                   choices=["sum", "mean", "concat"],
                   help="How multiple structure images are pooled in the guidance "
                        "tensor (deep-sets). 'sum' (original) makes |G| grow with the "
                        "number of images N — if training was dominated by N=1, a "
                        "4-image G is out-of-distribution. 'mean' makes |G| "
                        "N-invariant (identical to 'sum' at N=1).")
    g.add_argument("--decoder_skip", type=str, default="none",
                   choices=["none", "spectral_in", "guidance"],
                   help="Concatenate a pre-U-Net tensor onto the U-Net output before "
                        "the mean/sigma heads, so the decoders see information not "
                        "yet mixed over a 5x5 neighbourhood by enc0. 'spectral_in' = "
                        "the full per-pixel encoder output; 'guidance' = only the "
                        "structure-image guidance block.")
    g.add_argument("--guidance_raw", action="store_true",
                   help="Also pass the UNTRANSFORMED full-resolution intensity plane "
                        "in each image slot, alongside the learned encoding, so the "
                        "U-Net and its skip connections see the raw image pixels and "
                        "not only an embedded result. Slot width becomes 1+guidance_ch. "
                        "Requires --guidance_agg concat.")
    g.add_argument("--image_guidance", action="store_true",
                   help="Piece A: build a full-resolution guidance tensor from the "
                        "image(s) (band-embedding + deep-sets over images) and "
                        "concatenate into the encoder input, so the image rides the "
                        "U-Net skips. Band-agnostic, arbitrary count. nn_fill only.")
    g.add_argument("--film", action="store_true",
                   help="Piece B: FiLM-condition the encoder features on which "
                        "bands are imaged (deep-sets band embedding -> per-channel "
                        "gamma,beta). Use with --image_guidance.")
    g.add_argument("--image_tokens", action="store_true",
                   help="Route the imaged band(s) through 3D (x,y,nu) cross-attention "
                        "instead of the conv input. Image never touches spectral_in; "
                        "conv input stays plain nn_fill (W+1). Requires attention "
                        "stages. Works with --image_bands or --dynamic_image_band.")
    g.add_argument("--image_subsample", type=int, default=1024,
                   help="Image tokens per image (random-uniform subsample of the "
                        "H*W plane; deterministic stride at eval).")
    g.add_argument("--dynamic_image_band", action="store_true",
                   help="Select the imaged band PER CUBE at runtime: probe a random "
                        "pixel, find its peaks (scipy.find_peaks, height="
                        "--image_peak_height), and image one peak at random. Fixed "
                        "2W [value|coverage] input; the band varies per sample, not "
                        "the width. Mutually exclusive with --image_bands. nn_fill only.")
    g.add_argument("--image_peak_height", type=float, default=0.2,
                   help="Minimum peak height for find_peaks in dynamic band selection.")
    g.add_argument("--image_bands", type=int, nargs="+", default=None,
                   help="Band index/indices imaged as fixed full-field planes "
                        "(e.g. --image_bands 138). Turns the encoder input into the "
                        "arbitrary-imaging representation: [value (W) | per-band "
                        "coverage (W)] = 2W channels, FIXED regardless of how many "
                        "bands are imaged. Each imaged band's value plane is set to "
                        "the measured image and its coverage plane to 0 (observed "
                        "everywhere), written IN PLACE (no channel is added). "
                        "Requires --input_mode nn_fill.")
    g.add_argument("--attn_stages", type=str, nargs="+", default=None,
                   choices=["bottleneck", "dec2", "dec1", "dec0"],
                   help="Which U-Net stages get cross-attention (coarse->fine). "
                        "E.g. '--attn_stages bottleneck dec2' for multi-scale. "
                        "Implies attention on. Cost scales with the stage's query "
                        "resolution: bottleneck 16x16, dec2 32x32, dec1 64x64, "
                        "dec0 128x128.")
    g.add_argument("--attn_heads", type=int, default=4,
                   help="Number of attention heads (default 4).")
    g.add_argument("--attn_dim", type=int, default=256,
                   help="Attention d_model; d_head = attn_dim / attn_heads (default 256 -> 64).")
    g.add_argument("--attn_pos_bands", type=int, default=8,
                   help="Fourier positional-encoding bands for query/key positions.")
    g.add_argument("--head_kernel", type=int, default=1,
                   help="Spatial kernel size of the output-head convolutions (odd). "
                        "1 (default) = original per-pixel head: reads the U-Net latent "
                        "and decoder skip one pixel at a time, cannot move/snap a "
                        "boundary. 3 = 3x3 field per conv (~5x5 effective across the "
                        "two-layer spectral_mlp head), letting the head do local spatial "
                        "reasoning over the full-resolution decoder-skip guidance to "
                        "sharpen a boundary the bilinear upsample softened. Cost is a "
                        "k^2 factor on head params only (no attention memory).")

    g = p.add_argument_group("Training")
    g.add_argument("--head_attn", action="store_true",
                   help="Enable content-only cross-attention at native head "
                        "resolution (SDPA, memory-efficient). Each output cell "
                        "attends to the observed spectra by content similarity "
                        "(no positional encoding) to sharpen peaks. Off by default.")
    g.add_argument("--head_attn_dim", type=int, default=320,
                   help="d_model of the head attention (default 256).")
    g.add_argument("--head_attn_heads", type=int, default=4,
                   help="number of heads in the head attention (default 4).")
    g.add_argument("--n_tasks_per_epoch", type=int,   default=500)
    g.add_argument("--batch_size",        type=int,   default=8,
                   help="Reduce to 2-4 if GPU memory is limited")
    g.add_argument("--max_epochs",        type=int,   default=200)
    g.add_argument("--lr",                type=float, default=3e-4)
    g.add_argument("--weight_decay",      type=float, default=1e-5)
    g.add_argument("--lr_warmup_epochs",  type=int,   default=5)
    g.add_argument("--patience",          type=int,   default=40)
    g.add_argument("--no_early_stop", action="store_true",
                   help="disable EarlyStopping entirely. Use for fine-tuning / "
                        "resume runs where the monitored metric is expected to "
                        "worsen before recovering (early stop would fire during "
                        "the dip, and its restored state defeats a raised --patience).")
    g.add_argument("--num_workers",       type=int,   default=4)
    g.add_argument("--lambda_ctx",        type=float,   default=0.1)
    g.add_argument("--norm_smear", action="store_true",
                   help="SOFT per-scale density normalization: func/(density+eps_s) "
                        "with eps scaled to each scale's kernel peak. Cancels the "
                        "context-fraction magnitude drift without the square "
                        "artifacts of the hard --norm_mask division.")
    g.add_argument("--norm_eps", type=float, default=0.05,
                   help="eps as a fraction of each scale's peak density (--norm_smear). "
                        "Larger = smoother/safer in low-density regions, less exact "
                        "normalization. 0.01-0.1 is reasonable.")
    g.add_argument("--lambda_tv", type=float, default=0.0,
                   help="Weight on ISOTROPIC spatial total-variation of the mean "
                        "prediction: sqrt(dh^2 + dw^2) summed over pixels and bands. "
                        "Promotes piecewise-smooth reconstructions with sharp edges. "
                        "Spatial only (no spectral smoothing); applied to mu. Keep "
                        "SMALL — it competes with within-region texture.")
    g.add_argument("--tv_eps", type=float, default=1e-6,
                   help="Epsilon inside the isotropic-TV sqrt for stable gradients "
                        "where dh=dw=0.")
    g.add_argument("--lambda_img", type=float, default=0.0,
                   help="Weight on STRUCTURE-IMAGE CONSISTENCY: MSE between the "
                        "reconstruction at each imaged band and the structure image "
                        "supplied at that band, over all pixels. Each imaged band is "
                        "only ~1/W of the ordinary loss, so without this the model has "
                        "almost no pressure to reproduce the image it was handed. "
                        "Applied ONLY at the imaged bands (no spectral kernel).")
    g.add_argument("--lambda_grad", type=float, default=0.0,
                   help="Weight on STRUCTURE-IMAGE GRADIENT (EDGE) CONSISTENCY: MSE "
                        "between the spatial gradient of the reconstruction at each "
                        "imaged band and the spatial gradient of the structure image "
                        "at that band, over all pixels. Where lambda_img matches the "
                        "LEVEL (and so barely constrains a blurred or shifted edge), "
                        "this matches the EDGE directly, rewarding sharp, correctly "
                        "placed boundaries. Applied ONLY at the imaged bands (no "
                        "spectral kernel). 0 = off (default). Try 0.1-1.0.")
    g.add_argument("--lambda_grad_full", type=float, default=0.0,
                   help="Weight on ALL-BAND spatial gradient consistency: MSE "
                        "between the spatial gradient of the reconstruction and "
                        "the spatial gradient of the GROUND-TRUTH cube at EVERY "
                        "band (not just imaged bands). Supervises propagation of "
                        "spatial structure across the whole spectral axis, so "
                        "boundaries stay sharp at bands far from any structure "
                        "image. TRAINING-ONLY (uses the ground truth). Watch "
                        "val/coverage_2sigma: it biases mu, so lower this if "
                        "coverage degrades. 0 = off (default). Try 0.1-0.5.")
    g.add_argument("--lambda_mse", type=float, default=0.0,
                   help="Weight on a target-pixel MSE term added to the NLL loss "
                        "(NLL + lambda_mse * MSE). Forces the mean to commit to peak "
                        "amplitudes. 0 = pure NLL (default). Try 1-5.")
    g.add_argument("--mean_activation", type=str, default="sigmoid",
                   choices=["sigmoid", "scaled_sigmoid", "linear", "none"],
                   help="Output mean activation. sigmoid: bounded (0,1), saturates "
                        "(can't reach rails). scaled_sigmoid: can reach 0/1 at finite "
                        "pre-activation. linear: hard clamp [0,1] (reaches rails, dead "
                        "gradient outside). none: unbounded identity (clamp at inference).")
    g.add_argument("--sig_margin", type=float, default=0.05,
                   help="Margin for scaled_sigmoid: mu=(1+2m)*sigmoid-m. m=0.05 lets "
                        "the mean reach [0,1] within a finite pre-activation range.")
    g.add_argument("--mse_only", action="store_true",
                   help="PURE RECONSTRUCTION MODE: train the mean on full-cube MSE "
                        "only; the sigma head is excluded from the loss (untrained, "
                        "not meaningful). Checkpoint + early-stop monitor "
                        "val/recon_mse instead of val/nll. Use to test whether the "
                        "architecture can reconstruct at all, stripped of the NLL. "
                        "(Equivalent to --train_mode mean_mse.)")
    g.add_argument("--train_mode", type=str, default="joint_nll",
                   choices=["mean_mse", "sigma_nll", "joint_nll"],
                   help="Staged-training regime. mean_mse: Stage 1, train mean "
                        "head only on MSE (monitor val/recon_mse). sigma_nll: "
                        "Stage 2, FREEZE mean pathway and train ONLY the sigma "
                        "head on pure NLL (requires --init_from a Stage-1 ckpt). "
                        "joint_nll: both heads on joint Gaussian NLL (comparison). "
                        "--mse_only overrides this to mean_mse.")
    g.add_argument("--adaptive_context", action="store_true",
                   help="train on PURE uncertainty-selected adaptive context (hex seed "
                        "grown by frozen-acquirer sigma to a log_norm stop count). Every "
                        "sample is converted; no random-context mixing.")
    g.add_argument("--adaptive_points_per_step", type=int, default=8,
                   help="pixels added per acquisition round (larger = faster, coarser)")
    g.add_argument("--adaptive_start_frac_min", type=float, default=0.001,
                   help="min hex-seed start fraction, drawn uniformly per cube (0.1%%)")
    g.add_argument("--adaptive_start_frac_max", type=float, default=0.01,
                   help="max hex-seed start fraction, drawn uniformly per cube (1%%)")
    g.add_argument("--adaptive_acq_reduce", type=str, default="mean",
                   choices=["mean", "max"], help="sigma reduction over bands for acquisition")
    g.add_argument("--adaptive_refresh_epochs", type=int, default=5,
                   help="refresh the frozen acquirer every N epochs (lagged target)")
    g.add_argument("--init_from", type=str, default=None,
                   help="Path to a checkpoint to warm-start from (load matching "
                        "weights). Used by --train_mode sigma_nll to load the "
                        "Stage-1 mean before freezing it.")
    g.add_argument("--resume_from", type=str, default=None,
                   help="Path to a Lightning checkpoint (e.g. .../last.ckpt) to "
                        "RESUME training from: restores weights, optimizer, LR "
                        "schedule, epoch counter and callback state. Differs from "
                        "--init_from, which loads weights only and restarts at "
                        "epoch 0.")
    g.add_argument("--amp",               action="store_true",
                   help="Automatic mixed precision (fp16). NOTE: NLL can be fp16-"
                        "sensitive; if val/nll is unstable, train in fp32.")
    g.add_argument("--devices",           type=int,   default=1,
                   help="Number of accelerator devices (GPUs) for the Trainer. "
                        "1 (default) = single-GPU, identical to the previous behavior. "
                        "For multi-GPU DDP set this to the GPUs-per-node and pass "
                        "--strategy ddp (launch under srun with one task per GPU). "
                        "Effective batch = --batch_size * --devices * --num_nodes.")
    g.add_argument("--num_nodes",         type=int,   default=1,
                   help="Number of nodes for distributed training (default 1).")
    g.add_argument("--strategy",          type=str,   default=None,
                   help="Lightning training strategy. Default None = single-device "
                        "(valid on both Lightning 1.x and 2.x; 1.x rejects the string "
                        "'auto'). Use 'ddp' for multi-GPU/multi-node data-parallel; if "
                        "DDP raises an unused-parameter error, use "
                        "'ddp_find_unused_parameters_true'. The strings 'auto' and "
                        "'none' are also accepted and mapped to None for portability.")
    g.add_argument("--seed",              type=int,   default=42)

    g = p.add_argument_group("Output")
    g.add_argument("--results_dir", type=str, default="./results")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)