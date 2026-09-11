#!/usr/bin/env python3
"""
synth_onthefly.py — On-the-fly synthetic AFM-IR cube generation for CACNP
training.

Instead of training from a fixed library of pre-generated cubes, this Dataset
generates a fresh cube for every __getitem__ call:

  * morphology (phase map, feature sizes, boundaries, noise, ...) is generated
    on the fly via generate_synth_afmir.generate_one_cube,
  * the phase spectra are sampled from a large pre-normalized NIST matrix
    (shape (N, W)) passed in as `spectrum_library`,
  * the cube is then normalized with the SAME normalize_cube scheme used
    everywhere else (cube_minmax / context_minmax / context_zscore),

so the model sees an effectively infinite stream of distinct cubes and cannot
memorize a fixed spectral palette. 

Design notes
------------
* This module imports the generator; it does not duplicate it. The dependency
  flow is training -> this dataset -> generate_synth_afmir.
* Per-worker / per-item seeding: each __getitem__ derives an independent seed
  from (base_seed, epoch-ish index, worker id) so DataLoader workers do NOT
  generate identical cubes. Getting this wrong silently destroys diversity, so
  it is handled explicitly here AND via worker_init_fn (see make_worker_init_fn).
* Returns exactly what GridCNPDataset returns: (cube_tensor (H,Wsp,W) float32,
  obs_mask (H,Wsp) bool), so grid_cnp_collate and the training loop are
  unchanged.
"""

from typing import List, Optional, Sequence, Tuple   # Type hints used below

import numpy as np                                   # Array maths / RNG
import torch                                         # Tensors returned to the loader
import math                                          # log/sqrt for the log-normal fit
from statistics import NormalDist                    # Inverse normal CDF (quantile)
from scipy.signal import find_peaks                  # Peak detection for band selection
from scipy.ndimage import distance_transform_edt, gaussian_filter   # NN-fill + blur


def variance_peak_bands(sampled_spectra, thresh=0.2):    # Rank bands by sample variance
    """Rank bands by between-sample variance.
    variance per band -> min-max normalise -> find_peaks(height=thresh) ->
    return peak band indices sorted by variance DESCENDING. Empty if none.
    sampled_spectra: (n_sampled, W) full spectra at the observed pixels."""
    if sampled_spectra.shape[0] < 2:                 # Need >=2 spectra for a variance
        return np.array([], dtype=int)               # Nothing rankable
    var = sampled_spectra.var(axis=0)                       # (W,)
    rng = float(var.max() - var.min())               # Dynamic range of the variance
    if rng < 1e-12:                                  # Flat variance: no informative band
        return np.array([], dtype=int)               # Nothing rankable
    varn = (var - var.min()) / rng                   # Min-max normalise to [0,1]
    peaks, _ = find_peaks(varn, height=thresh)       # Local maxima above the threshold
    if len(peaks) == 0:                              # No peak cleared the threshold
        return np.array([], dtype=int)               # Nothing rankable
    order = np.argsort(varn[peaks])[::-1]                   # highest variance first
    return peaks[order].astype(int)                  # Band indices, best first
from torch.utils.data import Dataset                 # Base class for the dataset

from generate_synth_afmir import generate_one_cube   # The actual cube generator


try:
    from conv_cnp_afmir_prob import normalize_cube, NORM_MODES   # Shared normalisation
except Exception:  # pragma: no cover - fallback if import order differs
    normalize_cube = None                            # Flagged missing; checked in __init__
    NORM_MODES = ("cube_minmax", "context_minmax", "context_zscore")   # Fallback mode list


# =============================================================================
# Post-assembly artifacts (texture, scratches)
# =============================================================================


def add_texture(cube, phase_map, rng, sigma=(3.0, 15.0), amp=(0.02, 0.06),
                n_scales=2, clip=0.10):              # Within-region intensity texture
    """Spatially correlated within-region intensity variation.

    Multi-scale (n_scales Gaussian fields per phase, each with its own blur sigma)
    and INDEPENDENT PER PHASE, so texture does not bleed across a boundary. The
    combined field is clipped to +/- `clip`, so the spectral values are never
    changed by more than that fraction.
    """
    X, Y, _ = cube.shape                             # Spatial size (bands unused here)
    field = np.zeros((X, Y), dtype=np.float32)       # Accumulated texture field
    for k in np.unique(phase_map):                   # One independent field per phase
        m = (phase_map == k)                         # This phase's pixel mask
        if not m.any():                              # Phase absent from this cube
            continue                                 # Skip it
        fk = np.zeros((X, Y), dtype=np.float32)      # This phase's field
        for _ in range(int(n_scales)):               # Sum several blur scales
            g = rng.standard_normal((X, Y)).astype(np.float32)   # White noise field
            g = gaussian_filter(g, sigma=float(rng.uniform(*sigma)))   # Blur to correlate it
            sd = float(g.std())                      # Its post-blur spread
            if sd > 1e-8:                            # Guard against a degenerate field
                fk += float(rng.uniform(*amp)) * g / sd   # Normalise then scale by amplitude
        field[m] = fk[m]                             # Write only inside this phase
    field = np.clip(field, -clip, clip)              # Cap the fractional change
    return np.clip(cube * (1.0 + field)[:, :, None], 0.0, 1.0).astype(np.float32)   # Apply broadband


def add_scratches(cube, rng, n_scratches=(1, 4), width=(0.8, 2.5),
                  depth=(0.20, 0.60), bright_frac=0.15, curviness=0.25,
                  length=(0.4, 1.2)):                # Linear mechanical defects
    """Linear mechanical defects: BROADBAND multiplicative attenuation.

    The path is a persistent random walk seeded at a random interior point and
    grown in BOTH directions, each arm stopping the moment it leaves the field.
    Points are never clipped back onto the border — clipping would pin an exiting
    scratch to the edge and draw a spurious line down the side of the image.
    """
    X, Y, _ = cube.shape                             # Spatial size
    out = cube.astype(np.float32, copy=True)         # Work on a copy
    total = np.zeros((X, Y), dtype=np.float32)       # Accumulated attenuation field
    diag = float(np.hypot(X, Y))                     # Image diagonal (length budget scale)

    n = int(rng.integers(n_scratches[0], n_scratches[1] + 1))   # How many scratches
    for _ in range(n):                               # Draw each scratch
        r0, c0 = rng.uniform(0, X), rng.uniform(0, Y)   # Random seed point
        theta0 = rng.uniform(0, 2 * np.pi)           # Random initial heading
        budget = diag * float(rng.uniform(*length))  # Total path length to walk
        split = float(rng.uniform(0.2, 0.8))         # Split of that length between arms
        pts = []                                     # Collected path pixels
        for sign, steps in ((+1, budget * split), (-1, budget * (1.0 - split))):   # Both arms
            r, c = r0, c0                            # Restart at the seed point
            theta = theta0 if sign > 0 else theta0 + np.pi   # Opposite heading for arm 2
            for _s in range(int(steps)):             # Walk this arm
                if not (0.0 <= r < X and 0.0 <= c < Y):   # Left the field of view
                    break                            # Stop (do NOT clip to the border)
                pts.append((r, c))                   # Record this path point
                theta += rng.normal(0.0, curviness * 0.15)   # Persistent random turn
                r += np.sin(theta)                   # Step along the heading (row)
                c += np.cos(theta)                   # Step along the heading (col)
        if len(pts) < 3:                             # Degenerate/too-short path
            continue                                 # Skip this scratch

        pts = np.asarray(pts)                        # Path as an array
        seed_mask = np.zeros((X, Y), dtype=bool)     # Rasterised path mask
        seed_mask[pts[:, 0].astype(int), pts[:, 1].astype(int)] = True   # Mark path pixels
        d = distance_transform_edt(~seed_mask)       # Distance from the path
        w = float(rng.uniform(*width))               # Gaussian half-width of this scratch
        prof = np.exp(-(d / max(w, 1e-3)) ** 2).astype(np.float32)   # Gaussian cross-section

        a = float(rng.uniform(*depth))               # Attenuation depth at the centre
        if rng.random() < bright_frac:
            a = -a * 0.6                       # bright defect (debris/better contact)
        total += a * prof                            # Accumulate into the field

    total = np.clip(total, -0.95, 0.95)              # Never fully extinguish or double signal
    out *= (1.0 - total)[:, :, None]           # BROADBAND: same factor every band
    return np.clip(out, 0.0, 1.0).astype(np.float32)   # Clip back to valid range


def add_outlier_pixels(cube, rng, n_range=(1, 20), boost_frac=0.5,
                       peak_boost=(1.5, 3.5), shift_max=1,
                       peak_prominence=0.05, n_peaks_boost=(1, 2)):   # Rare anomalous pixels
    """Add rare ANOMALOUS pixels the model should learn to treat as LOCAL.

    A random number of pixels (uniform in n_range, inclusive) are corrupted. Each
    corrupted pixel is, with probability boost_frac, a TALL-PEAK outlier, otherwise
    a RIGID SPECTRAL SHIFT:

      TALL-PEAK : the pixel's 1-2 tallest peaks are multiplied by a large factor
                  (peak_boost), pushing them above the normal <=1 ceiling -- the
                  out-of-distribution outlier the experimental data shows. Applied
                  AFTER the [0,1] clip by the caller, so the tall peak survives.
      RIGID SHIFT: the whole spectrum is rolled +/- shift_max band indices, with
                  the exposed edge CLAMPED (not wrapped) so a peak cannot teleport
                  across the spectrum -- a per-pixel wavenumber-axis miscalibration.

    Kept SPARSE and ISOLATED on purpose: a handful of pixels per cube, so the model
    learns they are local anomalies rather than real structure. Returns a copy.
    """
    X, Y, W = cube.shape                             # Cube shape
    out = cube.copy()                                # Work on a copy
    n = int(rng.integers(int(n_range[0]), int(n_range[1]) + 1))   # How many anomalous pixels
    if n <= 0:                                       # Feature disabled
        return out                                   # Return unchanged copy
    n = min(n, X * Y)                                # Cannot exceed the pixel count
    flat = rng.choice(X * Y, size=n, replace=False)  # Distinct random pixel indices
    for fi in flat:                                  # Corrupt each chosen pixel
        r, c = divmod(int(fi), Y)                    # Flat index -> (row, col)
        spec = out[r, c, :].astype(np.float32).copy()   # That pixel's spectrum
        if rng.random() < boost_frac:                # Coin flip: tall-peak outlier
            # ---- TALL-PEAK outlier ----
            idx, _ = find_peaks(spec, prominence=peak_prominence)   # Locate its peaks
            if idx.size == 0:                        # No prominent peak found
                idx = np.array([int(np.argmax(spec))])   # Fall back to the global max
            kmax = min(int(n_peaks_boost[1]), idx.size)   # Cap by peaks available
            k = int(rng.integers(int(n_peaks_boost[0]), kmax + 1))   # How many to boost
            tallest = idx[np.argsort(spec[idx])[::-1][:k]]     # boost the tallest peaks
            factor = float(rng.uniform(*peak_boost))   # Multiplicative boost factor
            for pk in tallest:                       # Boost each selected peak
                lo, hi = max(0, pk - 2), min(W, pk + 3)         # boost the peak window
                spec[lo:hi] = spec[lo:hi] * factor   # Scale that window up
            out[r, c, :] = spec                      # Write the outlier spectrum back
        else:                                        # Otherwise: rigid shift outlier
            # ---- RIGID SPECTRAL SHIFT (clamped edges) ----
            sft = int(rng.integers(-int(shift_max), int(shift_max) + 1))   # Shift amount
            if sft == 0:                             # A zero shift is a no-op
                sft = 1                              # Force a real shift
            rolled = np.roll(spec, sft)              # Cyclically shift the spectrum
            if sft > 0:                              # Shifted right
                rolled[:sft] = spec[0]               # Clamp the exposed left edge
            else:                                    # Shifted left
                rolled[sft:] = spec[-1]              # Clamp the exposed right edge
            out[r, c, :] = rolled                    # Write the shifted spectrum back
    return out.astype(np.float32)                    # Return the corrupted cube


# =============================================================================
# Deployment-like context generation (hex grid + adaptive points)
# =============================================================================
def hex_grid_mask(H: int, Wsp: int, n_points: int, rng=None, jitter: float = 0.0):   # Hex lattice mask
    """Hexagonal lattice of ~n_points sites (the real experiment's opening move).
    `jitter` (in px) randomly perturbs each site so the model does not overfit to a
    perfectly periodic lattice."""
    if n_points < 1:                                 # Nothing requested
        return np.zeros((H, Wsp), bool)              # Empty mask
    a = np.sqrt(1.1547005383792517 / (n_points / (H * Wsp)))   # Lattice spacing for this density
    dx, dy = a, 0.86602540378 * a                    # Column and row pitch (hex geometry)
    yc, xc = H / 2.0, Wsp / 2.0                      # Lattice centre
    nc = int(np.ceil((Wsp / 2.0) / dx)) + 2          # Columns needed to cover the width
    nr = int(np.ceil((H / 2.0) / dy)) + 2            # Rows needed to cover the height
    c, r = np.meshgrid(np.arange(-nc, nc + 1), np.arange(-nr, nr + 1))   # Lattice indices
    x = c * dx + (r % 2) * (dx / 2.0) + xc           # Column positions (odd rows offset)
    y = r * dy + yc                                  # Row positions
    pts = np.column_stack([y.ravel(), x.ravel()])    # All lattice sites as (row, col)
    if jitter > 0 and rng is not None:               # Optional random perturbation
        pts = pts + rng.normal(0.0, jitter, pts.shape)   # Jitter each site
    p = np.round(pts).astype(int)                    # Snap to integer pixels
    ok = (p[:, 0] >= 0) & (p[:, 0] < H) & (p[:, 1] >= 0) & (p[:, 1] < Wsp)   # In-bounds sites
    p = np.unique(p[ok], axis=0)                     # Drop duplicates from rounding
    m = np.zeros((H, Wsp), bool)                     # Empty mask
    m[p[:, 0], p[:, 1]] = True                       # Mark the lattice sites
    return m                                         # Return the hex context mask


def apparent_regions(spectra: np.ndarray, bands, tau: float):   # Merge phases the image cannot separate
    """Which phases can the structure image(s) actually TELL APART?

    Two phases are INDISTINGUISHABLE if their endmember intensities agree to within
    `tau` at EVERY imaged band — i.e. the image cannot separate them. Single-linkage
    merge on that relation gives the APPARENT regions: what the image reveals.

    No image segmentation, no K to choose, no failure mode:
      * a FLAT / useless imaged band merges every phase into ONE region, so the
        oracle below degenerates exactly to plain NN-fill — correct, because an
        uninformative image gives the model no help;
      * MORE images split the regions further, so the oracle automatically models
        the resolving power of the image SET.

    tau should track the measurement noise: phases closer than the noise cannot be
    told apart by anyone.
    """
    P = spectra.shape[0]                             # Number of phases
    parent = list(range(P))                          # Union-find parent array

    def find(a):                                     # Union-find root lookup
        while parent[a] != a:                        # Walk up to the root
            parent[a] = parent[parent[a]]            # Path compression
            a = parent[a]                            # Step up
        return a                                     # The root label

    for p in range(P):                               # Every phase pair
        for q in range(p + 1, P):                    # (upper triangle only)
            if all(abs(float(spectra[p, l]) - float(spectra[q, l])) < tau for l in bands):   # Same at every imaged band
                ra, rb = find(p), find(q)            # Their current roots
                if ra != rb:                         # Not already merged
                    parent[ra] = rb                  # Merge the two sets

    roots, lab = {}, np.zeros(P, dtype=int)          # Root -> compact label map
    for p in range(P):                               # Relabel every phase
        r = find(p)                                  # Its root
        if r not in roots:                           # First time we see this root
            roots[r] = len(roots)                    # Assign the next compact id
        lab[p] = roots[r]                            # Store this phase's region id
    return lab                                       # Phase -> apparent-region label


def oracle_error_map(cube_raw, obs_mask, phase_map, spectra, bands, tau):   # Stand-in for model sigma
    """Error of an IMAGE-AWARE interpolator — a stand-in for where the trained
    model's sigma is high, computed from ground truth with ZERO forward passes.

    The oracle has the same information the model has: the sampled spectra AND the
    structure image(s). So it fills each unobserved pixel from the nearest observed
    pixel IN THE SAME APPARENT REGION (the image tells it which pixels share
    chemistry) rather than from the nearest observed pixel of any phase.

    Its error is therefore ~0 at boundaries the image RESOLVES, and large where the
    image is BLIND (phases it cannot separate) or where an apparent region contains
    no sampled spectrum at all — which is exactly where a good acquisition should
    measure, and exactly where the real model's sigma is high.
    """
    H, Wsp, _ = cube_raw.shape                       # Spatial size
    lab_of_phase = apparent_regions(spectra, bands, tau)   # Which phases the image merges
    region = lab_of_phase[phase_map]                     # (H, Wsp) apparent region id

    fill = np.zeros_like(cube_raw)                   # The oracle's reconstruction
    for k in np.unique(region):                      # Fill each apparent region separately
        reg = (region == k)                          # This region's pixels
        obs_in_reg = obs_mask & reg                  # Observed pixels inside it
        if not obs_in_reg.any():
            # apparent region with NO sampled spectrum: the oracle cannot fill it.
            # Leave the fill at 0 -> a large error -> the sampler is driven there.
            continue                                 # Skip (leave this region at zero)
        _, (iy, ix) = distance_transform_edt(~obs_in_reg, return_distances=True,
                                             return_indices=True)   # Nearest in-region observation
        f = cube_raw[iy, ix]                         # Gather that spectrum everywhere
        fill[reg] = f[reg]                           # Write only inside this region

    return np.abs(cube_raw - fill).mean(axis=2)         # (H, Wsp)


def adaptive_context_mask(cube_raw, phase_map, spectra, bands, n_init, n_add, rng,
                          a=1.0, b=1.0, temperature=0.1, batch=8, tau=0.064,
                          hex_jitter=0.5, noise_floor=0.0):   # Deployment-like context builder
    """Build an experiment-like context: hex grid, then points added the way the
    acquisition loop adds them.

        score(x) = a * norm(oracle_error(x)) + b * norm(distance_to_context(x))
        p(x)    ~ exp(score(x) / temperature)      over UNOBSERVED pixels

    Points are added iteratively in batches (the score is recomputed after each),
    because real acquisition is recursive.

    Selection is stochastic (softmax with temperature), not greedy.

    """
    H, Wsp, _ = cube_raw.shape                       # Spatial size
    mask = hex_grid_mask(H, Wsp, n_init, rng=rng, jitter=hex_jitter)   # Hex seed grid

    remaining = int(n_add)                           # Points still to acquire
    while remaining > 0:                             # Acquire in batches until done
        free = ~mask                                 # Currently unobserved pixels
        if not free.any():                           # Everything already measured
            break                                    # Stop
        err = oracle_error_map(cube_raw, mask, phase_map, spectra, bands, tau)   # Where error is high
        if noise_floor > 0:
            # don't chase pixels that are only uncertain because they are NOISY
            err = np.maximum(err - noise_floor, 0.0)   # Subtract the floor, clamp at 0
        d = distance_transform_edt(~mask).astype(np.float64)   # Distance from existing context

        def _n(x):                                   # Min-max normalise over FREE pixels
            v = x[free]                              # Values at unobserved pixels
            lo, hi = float(v.min()), float(v.max())  # Their range
            return np.zeros_like(x) if (hi - lo) < 1e-12 else (x - lo) / (hi - lo)   # Scale to [0,1]

        score = a * _n(err) + b * _n(d)              # Weighted error + distance score
        logits = score / max(temperature, 1e-3)      # Softmax temperature scaling
        logits[~free] = -np.inf                      # Never re-pick observed pixels
        logits -= logits[free].max()                 # Stabilise the exponential
        p = np.exp(logits)                           # Unnormalised probabilities
        tot = p.sum()                                # Normalising constant
        if not np.isfinite(tot) or tot <= 0:         # Degenerate distribution
            break                                    # Stop acquiring
        p = (p / tot).ravel()                        # Normalise and flatten

        k = min(batch, remaining, int(free.sum()))   # Points to add this round
        idx = rng.choice(H * Wsp, size=k, replace=False, p=p)   # Sample without replacement
        mask.ravel()[idx] = True                     # Mark them observed
        remaining -= k                               # Decrement the budget

    return mask                                      # The adaptive context mask


class _LibraryView:                                  # Numpy matrix adapter for the generator
    """
    Thin wrapper presenting a (N, W) numpy matrix the way generate_phase_spectra
    expects an external_library to behave: supports len() and integer indexing
    returning a (W,) row. Avoids copying the whole matrix into a Python list.
    """
    __slots__ = ("mat",)                             # No __dict__; just the matrix

    def __init__(self, mat: np.ndarray):             # Wrap the matrix
        self.mat = mat                               # Store it by reference

    def __len__(self) -> int:                        # Number of spectra
        return self.mat.shape[0]                     # Row count

    def __getitem__(self, idx):                      # Row access by index
        # generate_phase_spectra does .copy() on the returned row, so a view is fine
        return self.mat[idx]                         # Return the (W,) row view

    def __bool__(self) -> bool:                      # Truthiness test
        # generate_phase_spectra checks `if external_library:` — must be truthy
        return self.mat.shape[0] > 0                 # True when non-empty


class OnTheFlySynthDataset(Dataset):                 # Generates a fresh cube per item
    """
    Generates a fresh synthetic cube per item, sampling phase spectra from a
    pre-normalized NIST matrix. Returns (cube, obs_mask) like GridCNPDataset.

    Args:
        spectrum_library : (N, W) float32, ALREADY NORMALIZED. Rows are spectra.
        wavenumbers      : (W,) float32. Length must match spectrum_library W.
        image_size       : (X, Y) spatial size of generated cubes.
        n_tasks_per_epoch: number of items per epoch (defines __len__).
        param_ranges     : dict of generation ranges (see DEFAULT_PARAM_RANGES).
        norm_mode        : input normalization ('context_zscore' etc.).
        n_context / n_context_min / log_uniform_context : context-size sampling,
                            identical semantics to GridCNPDataset.
        seed             : base RNG seed.
        split            : 'train' or 'val'. 'val' uses a FIXED seed offset and
                            FIXED per-index seeds so the validation set is the
                            same cubes every epoch (stable val metric).
        epoch            : current epoch (mixed into the train seed so each epoch
                            sees new cubes; set via set_epoch()).
    """

    def __init__(
        self,
        spectrum_library: np.ndarray,                # (N, W) pre-normalised spectra
        wavenumbers:      np.ndarray,                # (W,) wavenumber grid
        image_size:       Tuple[int, int],           # (X, Y) cube spatial size
        n_tasks_per_epoch: int,                      # Items per epoch
        param_ranges:     dict,                      # Generation ranges
        norm_mode:        str = "context_zscore",    # Cube normalisation scheme
        n_context:        int = 4000,                # Max context pixels
        n_context_min:    int = 4,                   # Min context pixels
        context_dist:     str = 'log_uni',           # Context-size distribution
        mode_frac:        float = 0.05,              # Log-normal mode fraction
        tail_frac:        float = 0.20,              # Log-normal tail fraction
        tail_quantile:    float = 0.99,              # Log-normal tail quantile
        seed:             int = 42,                  # Base RNG seed
        split:            str = "train",             # 'train' or 'val'
        epoch:            int = 0,                   # Current epoch (train only)
        dynamic_image_band: bool = False,            # Pick one imaged band per cube
        image_peak_height:  float = 0.2,             # find_peaks height for that pick
        multi_image:        bool = False,            # Variance-selected multi-band imaging
        texture_prob:       float = 0.0,             # fraction of cubes given texture
        texture_sigma:      Sequence[float] = (3.0, 15.0),   # Texture blur sigma range
        texture_amp:        Sequence[float] = (0.02, 0.06),  # Texture amplitude range
        texture_clip:       float = 0.10,            # Cap on the texture field
        texture_scales:     int = 2,                 # Blur scales summed per phase
        scratch_prob:       float = 0.0,             # fraction of cubes given scratches
        scratch_count:      Sequence[int] = (1, 4),          # Scratches per affected cube
        scratch_width:      Sequence[float] = (0.8, 2.5),    # Scratch half-width range
        scratch_depth:      Sequence[float] = (0.20, 0.60),  # Scratch attenuation range
        outlier_n_range:    Sequence[int] = (0, 0),  # per-cube #anomalous pixels; (0,0)=off
        outlier_boost_frac: float = 0.5,             # P(tall-peak) per anomalous pixel; else shift
        outlier_peak_boost: Sequence[float] = (1.5, 3.5),    # Tall-peak boost factor range
        outlier_shift_max:  int = 1,                 # Max rigid-shift magnitude (bands)
        context_mode:       str = "random",          # random | hex | adaptive | mixture
        context_mix:        Optional[Sequence[float]] = None,  # P(random,hex,adaptive)
        adaptive_tau:       float = 0.064,           # phase indistinguishability threshold
        adaptive_noise_floor: float = 0.0,           # subtract from oracle error
        image_var_thresh:   float = 0.2,             # Variance-peak height threshold
        image_count_weights: Optional[Sequence[float]] = None,   # P(image 1,2,3,4 bands)
        max_images:         int = 4,                 # Cap on imaged bands per cube
    ):
        if normalize_cube is None:                   # The lazy import failed
            raise ImportError("normalize_cube could not be imported from "
                              "conv_cnp_afmir_prob; check import order.")   # Fail loudly
        if norm_mode not in NORM_MODES:              # Reject unknown normalisation
            raise ValueError(f"norm_mode must be one of {NORM_MODES}")   # Fail loudly
        lib = np.asarray(spectrum_library, dtype=np.float32)   # Coerce the library
        if lib.ndim != 2:                            # Must be a 2-D matrix
            raise ValueError(f"spectrum_library must be (N, W); got {lib.shape}")   # Fail loudly
        W_lib = lib.shape[1]                         # Library band count
        W_wv  = int(len(wavenumbers))                # Wavenumber grid length
        if W_lib != W_wv:                            # They must be the same grid
            raise ValueError(f"spectrum_library W ({W_lib}) != wavenumbers W "
                             f"({W_wv}); they must be on the same grid.")   # Fail loudly

        self.library        = _LibraryView(lib)      # Wrap for the generator's API
        self.wavenumbers    = np.asarray(wavenumbers, dtype=np.float32)   # Store the grid
        self.X, self.Y      = int(image_size[0]), int(image_size[1])      # Store cube size
        self.n_tasks        = int(n_tasks_per_epoch) # Store items per epoch
        self.ranges         = param_ranges           # Store generation ranges
        self.norm_mode      = norm_mode              # Store normalisation mode
        self.n_context      = int(n_context)         # Store max context size
        self.n_context_min  = int(n_context_min)     # Store min context size
        self.context_dist   = context_dist           # Store context distribution
        self.mode_frac      = mode_frac              # Store log-normal mode fraction
        self.tail_frac      = tail_frac              # Store log-normal tail fraction
        self.tail_quantile  = tail_quantile          # Store log-normal tail quantile
        self.base_seed      = int(seed)              # Store base seed
        self.split          = split                  # Store split name
        self.epoch          = int(epoch)             # Store current epoch
        self.dynamic_image_band = bool(dynamic_image_band)   # Store dynamic-band flag
        self.image_peak_height  = float(image_peak_height)   # Store peak-height threshold
        self.multi_image        = bool(multi_image)          # Store multi-image flag
        self.texture_prob       = float(texture_prob)        # Store texture probability
        self.texture_sigma      = tuple(texture_sigma)       # Store texture blur range
        self.texture_amp        = tuple(texture_amp)         # Store texture amplitude range
        self.texture_clip       = float(texture_clip)        # Store texture clip
        self.texture_scales     = int(texture_scales)        # Store texture scale count
        self.scratch_prob       = float(scratch_prob)        # Store scratch probability
        self.scratch_count      = tuple(scratch_count)       # Store scratch count range
        self.scratch_width      = tuple(scratch_width)       # Store scratch width range
        self.scratch_depth      = tuple(scratch_depth)       # Store scratch depth range
        self.outlier_n_range    = tuple(outlier_n_range)     # Store outlier count range
        self.outlier_boost_frac = float(outlier_boost_frac)  # Store tall-peak probability
        self.outlier_peak_boost = tuple(outlier_peak_boost)  # Store boost factor range
        self.outlier_shift_max  = int(outlier_shift_max)     # Store max shift magnitude
        self.context_mode       = str(context_mode)          # Store context mode
        self.context_mix        = (list(context_mix) if context_mix is not None
                                   else [0.40, 0.20, 0.40])  # random / hex / adaptive
        self.adaptive_tau       = float(adaptive_tau)        # Store indistinguishability tau
        self.adaptive_noise_floor = float(adaptive_noise_floor)   # Store oracle noise floor
        self.image_var_thresh   = float(image_var_thresh)    # Store variance threshold
        self.max_images         = int(max_images)            # Store image cap
        self.image_count_weights = (list(image_count_weights)
                                    if image_count_weights is not None
                                    else [0.6, 0.25, 0.10, 0.05])   # Default image-count weights
        self._worker_offset = 0  # set by worker_init_fn

    def set_epoch(self, epoch: int):                 # Called by the training callback
        self.epoch = int(epoch)                      # Store the new epoch

    def __len__(self) -> int:                        # Dataset length
        return self.n_tasks                          # Items per epoch

    def _item_seed(self, idx: int) -> int:           # Per-item deterministic seed
        """
        Ensures unique seeds for different workers
        Validation uses epoch=0 always, so val cubes are identical every epoch.
        """
        epoch = 0 if self.split == "val" else self.epoch   # Freeze the val stream
        
        ss = np.random.SeedSequence([
            self.base_seed,                          # Run-level seed
            0 if self.split == "val" else 1,         # Split identifier
            epoch,                                   # Epoch (0 for val)
            self._worker_offset,                     # DataLoader worker id
            idx,                                     # Item index
        ])
        return int(ss.generate_state(1)[0])          # Derive one 32-bit seed

    def _sample_params(self, rng: np.random.Generator) -> dict:   # Draw generation params
        """Draw one generation parameter dict from the configured ranges."""
        r = self.ranges                              # Shorthand for the ranges dict
        n_phases = int(rng.integers(r["n_phases"][0], r["n_phases"][1] + 1))   # Phase count
        n_feat   = n_phases - 1                      # Non-background phase count

        base_feat = float(rng.uniform(*r["feature_size"]))          # Cube-level feature size
        base_curv = float(rng.uniform(*r["boundary_curvature"]))    # Cube-level curvature
        feature_sizes = [
            float(np.clip(base_feat * rng.uniform(0.70, 1.30),
                          r["feature_size"][0], r["feature_size"][1]))
            for _ in range(n_feat)
        ]                                            # Per-phase sizes, jittered about the base
        curvatures = [
            float(np.clip(base_curv * rng.uniform(0.85, 1.15),
                          r["boundary_curvature"][0], r["boundary_curvature"][1]))
            for _ in range(n_feat)
        ]                                            # Per-phase curvatures, jittered
        # phase fractions sum <= 0.85 (>=15% background), matching the generator
        phase_fractions = []                         # Area fraction per non-bg phase
        remaining = 0.85                             # Area budget left to allocate
        for j in range(n_feat):                      # Allocate phase by phase
            hi   = min(r["phase_fraction"][1], remaining / max(1, n_feat - j))   # Upper bound
            lo   = min(r["phase_fraction"][0], hi)   # Lower bound (never above hi)
            frac = float(rng.uniform(lo, hi))        # Draw this phase's fraction
            phase_fractions.append(frac)             # Record it
            remaining -= frac                        # Deduct from the budget

        return dict(
            cube_id               = -1,              # Unused for on-the-fly cubes
            X                     = self.X,          # Cube height
            Y                     = self.Y,          # Cube width
            n_phases              = n_phases,        # Phase count
            feature_sizes         = feature_sizes,   # Per-phase feature sizes
            curvatures            = curvatures,      # Per-phase curvatures
            phase_fractions       = phase_fractions, # Per-phase area fractions
            morphology            = str(rng.choice(r["morphology_types"])),   # dispersed|voronoi
            boundary_width        = float(rng.uniform(*r["boundary_width"])), # Interface width
            mixing_type           = str(rng.choice(r["mixing_types"])),       # Boundary mixing model
            noise_std             = float(rng.uniform(*r["noise_std"])),      # Detector noise level
            intra_phase_variation = float(rng.uniform(*r["intra_phase_variation"])),   # Within-region variation
            spectra_mode          = "external",       # always sample from NIST matrix
            n_peaks_range         = list(r.get("n_peaks", [2, 8])),             # Synthetic-peak count
            peak_width_range      = list(r.get("peak_width", [20.0, 100.0])),   # Synthetic-peak width
            peak_amp_range        = list(r.get("peak_amp_range", [0.3, 1.0])),  # Synthetic-peak amplitude
            spectra_shift_std     = float(r.get("spectra_shift_std", 0.0)),     # Spectral shift jitter
            spectra_width_jitter  = float(r.get("spectra_width_jitter", 0.0)),  # Spectral width jitter
            spectra_amp_jitter    = float(r.get("spectra_amp_jitter", 0.0)),    # Spectral amplitude jitter
            depth_edge_mix        = float(rng.uniform(*r.get("depth_edge_mix", [0.0, 0.0]))),   # Legacy edge mix
            depth_decay_length    = float(rng.uniform(*r.get("depth_decay_length", [6.0, 6.0]))),   # Legacy decay length
            depth_probing_depth_nm= (float(rng.uniform(*r["depth_probing_depth"]))
                                     if r.get("depth_probing_depth") is not None else None),   # Probing depth (nm)
            depth_thickness_range = list(r.get("depth_thickness_range", [200.0, 500.0])),   # Particle thickness (nm)
            depth_thickness_profile = str(r.get("depth_thickness_profile", "sqrt")),        # Thickness profile shape
            depth_background_phase= 0,               # Phase treated as the substrate
            relative_intensity_range = (list(r["relative_intensity"])
                                        if r.get("relative_intensity") is not None else None),   # Per-phase intensity range
            clip_output           = bool(r.get("clip_output", True)),   # Whether to clip to [0,1]
            # generate_one_cube seeds its own rng from this; give it our item seed
            seed                  = int(rng.integers(0, 2 ** 31)),   # Generator seed
        )

    def __getitem__(self, idx: int):                 # Build one training item
        item_seed = self._item_seed(idx)             # Deterministic per-item seed
        rng = np.random.default_rng(item_seed)       # This item's RNG

        # 1. sample generation parameters + generate one cube on the fly
        params = self._sample_params(rng)            # Draw the generation parameters

        noise_std = float(params["noise_std"])       # Remember the requested noise level
        params = dict(params)                        # Copy before mutating
        params["noise_std"] = 0.0                    # Disable noise inside the generator

        out = generate_one_cube(params, self.wavenumbers, self.library)    # Generate the cube
        cube_raw = np.asarray(out["cube"], dtype=np.float32)   # (X, Y, W)

        # Post-assembly artifacts
        if self.texture_prob > 0.0 and rng.random() < self.texture_prob:   # This cube gets texture
            cube_raw = add_texture(
                cube_raw, np.asarray(out["phase_map"]), rng,
                sigma=self.texture_sigma, amp=self.texture_amp,
                n_scales=self.texture_scales, clip=self.texture_clip)      # Apply texture
        if self.scratch_prob > 0.0 and rng.random() < self.scratch_prob:   # This cube gets scratches
            cube_raw = add_scratches(
                cube_raw, rng, n_scratches=self.scratch_count,
                width=self.scratch_width, depth=self.scratch_depth)        # Apply scratches

        # Detector Noise
        if noise_std > 0.0:                                                       # Noise was requested
            cube_raw = cube_raw + rng.normal(0.0, noise_std,
                                             cube_raw.shape).astype(np.float32)   # Add Gaussian noise

        # Add anomalous pixels
        if self.outlier_n_range[1] > 0:              # Outlier injection enabled
            cube_raw = add_outlier_pixels(
                cube_raw, rng, n_range=self.outlier_n_range,
                boost_frac=self.outlier_boost_frac,
                peak_boost=self.outlier_peak_boost,
                shift_max=self.outlier_shift_max)    # Corrupt a few pixels
        H, Wsp, Wv = cube_raw.shape                  # Final cube shape
        n_pix = H * Wsp                              # Pixel count

        # 2. sample context size + build observation mask (same as GridCNPDataset)
        if self.context_dist == 'log_uni':               # Log-uniform context size
            log_lo = np.log(max(self.n_context_min, 1))  # Lower log bound
            log_hi = np.log(self.n_context + 1)          # Upper log bound
            n_ctx = int(np.clip(
                np.round(np.exp(rng.uniform(log_lo, log_hi))),
                self.n_context_min, self.n_context,
            ))                                           # Draw and clip
        elif self.context_dist == 'log_norm':            # Log-normal context size
            mu, sigma, _, _ = lognormal_from_mode_tail(
                n_pix=n_pix,
                mode_frac=self.mode_frac,
                tail_frac=self.tail_frac,
                tail_quantile=self.tail_quantile,
            )                                            # Fit params from mode/tail

            n_ctx = int(
                _rejection_truncate(
                    lambda r, k: np.round(r.lognormal(mean=mu, sigma=sigma, size=k)),
                    self.n_context_min,
                    self.n_context,
                    1,
                    rng,
                )[0]
            )                                            # Draw, rejecting out-of-range
        else:                                            # Uniform context size
            n_ctx = int(rng.integers(self.n_context_min, self.n_context + 1))   # Draw uniformly

        n_ctx = min(n_ctx, n_pix)                        # Cap at the pixel count

        # Build observation mask
        # context_mode controls what kind of context the model is trained on.
        #   random   : uniform random pixels (the original; keeps the model general)
        #   hex      : a hexagonal grid  (the experiment's OPENING MOVE)
        #   adaptive : hex grid + points added the way the acquisition loop adds
        #              them (an image-aware GT oracle stands in for the model's
        #              sigma, at zero forward passes)
        #   mixture  : draw one of the three per cube, per self.context_mix
        #
        mode = self.context_mode                     # Configured context mode
        if mode == "mixture":                        # Draw a mode for this cube
            w = np.asarray(self.context_mix, float); w = w / w.sum()   # Normalise the weights
            mode = str(rng.choice(["random", "hex", "adaptive"], p=w))  # Sample one mode

        if mode == "random":                         # Uniform-random context
            flat_perm = rng.permutation(n_pix)       # Shuffle pixel indices
            obs_flat = np.zeros(n_pix, dtype=bool)   # Start with nothing observed
            obs_flat[flat_perm[:n_ctx]] = True       # Observe the first n_ctx
            obs_mask = obs_flat.reshape(H, Wsp)      # Reshape to 2-D

        elif mode == "hex":                          # Hexagonal-grid context
            obs_mask = hex_grid_mask(H, Wsp, n_ctx, rng=rng,
                                     jitter=float(rng.uniform(0.0, 1.0)))   # Jittered hex grid

        elif mode == "adaptive":                     # Deployment-like adaptive context
            a_w   = float(rng.uniform(0.0, 1.0))     # weight on oracle error
            b_w   = float(rng.uniform(0.0, 1.0))     # weight on distance
            if a_w + b_w < 1e-3:                     # avoid the degenerate 0/0
                a_w, b_w = 1.0, 1.0                  # Fall back to equal weights
            temp  = float(np.exp(rng.uniform(np.log(0.02), np.log(1.0))))  # log-uniform
            tau   = float(self.adaptive_tau * rng.uniform(0.75, 1.5))      # Jittered tau
            # Determine how far long the acquisition the cube is
            frac_init = float(rng.uniform(0.25, 1.0))      # Share of context from the hex core
            n_init = max(4, int(round(n_ctx * frac_init))) # Hex seed size
            n_add  = max(0, n_ctx - n_init)                # Adaptively-acquired remainder

            phase_map = np.asarray(out["phase_map"])     # Ground-truth phase labels
            phase_spec = np.asarray(out["spectra"])      # (n_phases, W) endmembers

            # Imaged bands are selected from initial hexagonal context spectra
            init_mask = hex_grid_mask(H, Wsp, n_init, rng=rng, jitter=0.5)   # The hex core
            sampled0 = cube_raw[init_mask]                                   # Spectra at the hex sites
            ranked0 = variance_peak_bands(sampled0, self.image_var_thresh)   # Rank bands
            if len(ranked0) == 0:                                            # No variance peak found
                ranked0 = np.array([int(sampled0.var(axis=0).argmax())]) \
                          if sampled0.shape[0] >= 2 else np.array([Wv // 2]) # Fallback band
            n_img0 = 1                                                       # Default: one structure image
            if self.multi_image:                                             # Multi-image mode
                w = np.asarray(self.image_count_weights, float); w = w / w.sum()   # Normalise weights
                n_img0 = int(rng.choice(np.arange(1, len(w) + 1), p=w))      # Draw image count
            oracle_bands = [int(b) for b in ranked0[:max(1, min(n_img0, len(ranked0)))]]   # Imaged bands

            obs_mask = adaptive_context_mask(
                cube_raw, phase_map, phase_spec, oracle_bands,
                n_init=n_init, n_add=n_add, rng=rng,
                a=a_w, b=b_w, temperature=temp, batch=8, tau=tau,
                hex_jitter=0.5, noise_floor=self.adaptive_noise_floor)       # Build the context
        else:                                                                # Unrecognised mode
            raise ValueError(f"unknown context_mode {self.context_mode!r}")  # Fail loudly

        if not obs_mask.any():                     # never hand the model an empty context
            obs_mask.ravel()[int(rng.integers(n_pix))] = True   # Observe one random pixel

        # 3. normalize with the selected mode (using the context for context_*)
        cube, _stats = normalize_cube(cube_raw, obs_mask, mode=self.norm_mode)   # Normalise

        # 4. Check the cube does not have any NaNs 
        cube = np.ascontiguousarray(cube, dtype=np.float32)   # Ensure contiguous float32
        if not np.isfinite(cube).all():                       # Any NaN or inf present
            raise FloatingPointError(
                f"non-finite values in generated cube (idx={idx}, seed={item_seed}, "
                f"n_phases={params['n_phases']}, morph={params['morphology']})")   # Fail loudly

        # 5. Dynamic imaged-band selection
        if self.multi_image:                         # Variance-selected multi-image path
            # Variance-based selection on the sampled hexagonal spectra.
            sampled = cube_raw[obs_mask]                                   # (n_obs, W)
            ranked = variance_peak_bands(sampled, self.image_var_thresh)   # Rank bands
            if len(ranked) == 0:                                           # fallback: top-variance band
                ranked = np.array([int(sampled.var(axis=0).argmax())]) if sampled.shape[0] >= 2 \
                         else np.array([int(np.argmax(cube_raw[H // 2, Wsp // 2]))])   # Fallback band
            counts = np.arange(1, len(self.image_count_weights) + 1)       # Possible image counts
            w = np.asarray(self.image_count_weights, float); w = w / w.sum()   # Normalise weights
            N = int(rng.choice(counts, p=w))                               # Draw how many bands to image
            N = max(1, min(N, len(ranked), self.max_images))               # Clamp to what's available
            bands = ranked[:N].astype(int)                                 # The top-N variance peaks
            bands_pad = np.zeros(self.max_images, dtype=np.int64)          # Padded band array
            bands_pad[:N] = bands                                          # Fill the valid slots
            return (torch.from_numpy(cube), torch.from_numpy(obs_mask),
                    torch.from_numpy(bands_pad), torch.tensor(N, dtype=torch.long))   # 4-tuple

        if self.dynamic_image_band:                                        # Single dynamic band path
            pr = int(rng.integers(H)); pc = int(rng.integers(Wsp))         # Random probe position
            spec = cube_raw[pr, pc, :]                                     # That pixel's raw spectrum
            peaks, _ = find_peaks(spec, height=self.image_peak_height)     # Its peaks
            band = int(rng.choice(peaks)) if len(peaks) > 0 else int(np.argmax(spec))   # Pick one
        else:                                                              # No dynamic band
            band = -1                                                      # If no band

        return (torch.from_numpy(cube), torch.from_numpy(obs_mask),
                torch.tensor(band, dtype=torch.long))   # 3-tuple


# Default generation ranges (mirrors the relative-intensity dataset settings).
DEFAULT_PARAM_RANGES = dict(
    n_phases            = [2, 3],                    # Phase count range
    feature_size        = [4.0, 40.0],               # Feature size range (px)
    boundary_curvature  = [0.4, 1.0],                # Curvature range
    phase_fraction      = [0.1, 0.4],                # Area fraction per non-bg phase
    morphology_types    = ["dispersed", "voronoi"],  # Morphologies to sample from
    boundary_width      = [1.5, 2.0],                # Interface mixing width (px)
    mixing_types        = ["linear"],                # Boundary mixing models
    noise_std           = [0.032, 0.032],            # Detector noise range
    intra_phase_variation = [0.0, 0.0],              # Within-region variation range
    spectra_shift_std   = 0.0,                       # Spectral shift jitter (off)
    spectra_width_jitter = 0.0,                      # Spectral width jitter (off)
    spectra_amp_jitter  = 0.0,                       # Spectral amplitude jitter (off)
    depth_edge_mix      = [0.0, 0.0],                # Legacy edge-mix range (off)
    depth_decay_length  = [6.0, 6.0],                # Legacy decay-length range
    depth_probing_depth = None,                      # e.g. [100.0, 300.0] to enable physical depth
    depth_thickness_range = [200.0, 500.0],          # Per-particle thickness range (nm)
    depth_thickness_profile = "sqrt",                # "sqrt" (puddle) | "linear" (wedge)
    relative_intensity  = None,                      # Per-phase intensity range (off)
    clip_output         = True,                      # Clip generated cubes to [0,1]
)


def make_worker_init_fn(base_seed: int):             # Build a DataLoader worker initialiser
    """
    Returns a worker_init_fn that gives each DataLoader worker a distinct seed
    offset, so workers generate independent cubes (not B/num_workers duplicates).
    Also reseeds numpy/torch inside the worker for any incidental RNG use.
    """
    def _init(worker_id: int):                       # Runs inside each worker process
        info = torch.utils.data.get_worker_info()    # Worker metadata
        if info is None:                             # Not in a worker (num_workers=0)
            return                                   # Nothing to do
        ds = info.dataset                            # This worker's dataset copy
        # distinct, stable per-worker offset
        if hasattr(ds, "_worker_offset"):            # Our dataset exposes the hook
            ds._worker_offset = int(worker_id) + 1   # Give it a distinct offset
        s = (int(base_seed) + 100003 * (int(worker_id) + 1)) % (2 ** 31)   # Derive a worker seed
        np.random.seed(s)                            # Reseed numpy's global RNG
        torch.manual_seed(s)                         # Reseed torch's global RNG
    return _init                                     # Hand back the initialiser


def load_spectrum_library(path: str) -> np.ndarray:  # Load the NIST spectrum matrix
    """Load the NIST (N, W) matrix from .npy."""
    mat = np.load(path)                              # Load from disk
    mat = np.asarray(mat, dtype=np.float32)          # Coerce to float32
    if mat.ndim != 2:                                # Must be a 2-D matrix
        raise ValueError(f"spectrum library at {path} must be 2D (N, W); "
                         f"got shape {mat.shape}")   # Fail loudly
    return mat                                       # Return the library

def _rejection_truncate(draw_fn, lo, hi, n_samples, rng, max_iter=200):   # Truncated sampling
    """draw_fn(rng, k) -> (k,) array of (already-rounded) candidates."""
    out = np.empty(0, dtype=float)                   # Accepted samples so far
    for _ in range(max_iter):                        # Bounded retry loop
        need = n_samples - out.size                  # How many are still missing
        if need <= 0:                                # Enough collected
            break                                    # Done
        cand = draw_fn(rng, int(need * 1.4) + 16)    # Over-draw candidates
        cand = cand[(cand >= lo) & (cand <= hi)]     # Keep only in-range ones
        out = np.concatenate([out, cand])            # Accumulate
    if out.size < n_samples:                         # Never filled the quota
        raise RuntimeError("rejection sampling failed to fill; widen bounds/spread")   # Fail loudly
    return out[:n_samples]                           # Return exactly n_samples

def lognormal_from_mode_tail(
    n_pix:         int,                              # Pixels per cube (sets the scale)
    mode_frac:     float = 0.05,                     # coverage at the density PEAK (true mode)
    tail_frac:     float = 0.20,                     # coverage at the upper-tail quantile
    tail_quantile: float = 0.99,                     # which quantile tail_frac represents
):                                                   # Fit log-normal params from mode + tail
    """
    Return (mu, sigma, n_mode, n_tail) for log X ~ N(mu, sigma^2) such that:
        mode of X      = exp(mu - sigma^2) = mode_frac * n_pix
        the q-quantile = exp(mu + sigma*z_q) = tail_frac * n_pix
    Subtracting the two log-equations gives a quadratic in sigma:
        sigma^2 + z_q*sigma - ln(n_tail/n_mode) = 0.
    """
    n_mode = mode_frac * n_pix                       # Context size at the mode
    n_tail = tail_frac * n_pix                       # Context size at the tail quantile
    ratio  = n_tail / n_mode                         # Their ratio (drives sigma)
    if ratio <= 1.0:                                 # Tail must exceed the mode
        raise ValueError("tail_frac must exceed mode_frac")   # Fail loudly
    z = NormalDist().inv_cdf(tail_quantile)          # z-score of the tail quantile
    sigma = (-z + math.sqrt(z * z + 4.0 * math.log(ratio))) / 2.0   # Positive root of the quadratic
    mu    = math.log(n_mode) + sigma * sigma         # Back out mu from the mode equation
    return mu, sigma, n_mode, n_tail                 # Params plus the two anchor counts
