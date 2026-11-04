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
memorize a fixed spectral palette. This is Stage 3 ("generalize to diverse
spectra"): with thousands of NIST spectra sampled at random, a given phase-
spectrum combination essentially never repeats.

Design notes
------------
* This module IMPORTS the generator; it does not duplicate it. The dependency
  flow is training -> this dataset -> generate_synth_afmir.
* The NIST library is assumed ALREADY NORMALIZED (the user normalizes the IR
  data upstream), and is passed as a single (N, W) float32 array. We sample
  ROWS from it; W must equal the model's wavenumber count.
* Per-worker / per-item seeding: each __getitem__ derives an independent seed
  from (base_seed, epoch-ish index, worker id) so DataLoader workers do NOT
  generate identical cubes. Getting this wrong silently destroys diversity, so
  it is handled explicitly here AND via worker_init_fn (see make_worker_init_fn).
* Returns exactly what CACNPDataset returns: (cube_tensor (H,Wsp,W) float32,
  obs_mask (H,Wsp) bool), so cacnp_collate and the training loop are
  unchanged.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import math
from statistics import NormalDist
from scipy.signal import find_peaks
from scipy.ndimage import distance_transform_edt, gaussian_filter


def variance_peak_bands(sampled_spectra, thresh=0.2):
    """Rank bands by between-sample variance (the user's compute_variance).
    variance per band -> min-max normalise -> find_peaks(height=thresh) ->
    return peak band indices sorted by variance DESCENDING. Empty if none.
    sampled_spectra: (n_sampled, W) full spectra at the observed pixels."""
    if sampled_spectra.shape[0] < 2:
        return np.array([], dtype=int)
    var = sampled_spectra.var(axis=0)                       # (W,)
    rng = float(var.max() - var.min())
    if rng < 1e-12:
        return np.array([], dtype=int)
    varn = (var - var.min()) / rng
    peaks, _ = find_peaks(varn, height=thresh)
    if len(peaks) == 0:
        return np.array([], dtype=int)
    order = np.argsort(varn[peaks])[::-1]                   # highest variance first
    return peaks[order].astype(int)
from torch.utils.data import Dataset

from generate_synth_afmir import generate_one_cube

# normalize_cube is defined in the model module; import lazily to avoid a hard
# circular import if this module is imported first.
try:
    from conv_cnp_afmir_prob import normalize_cube, NORM_MODES
except Exception:  # pragma: no cover - fallback if import order differs
    normalize_cube = None
    NORM_MODES = ("cube_minmax", "context_minmax", "context_zscore")


# =============================================================================
# Post-assembly artifacts (texture, scratches)
# =============================================================================
# Both are MULTIPLICATIVE and BROADBAND: they change how much signal a pixel
# returns, not which wavenumbers absorb. That is the physics (contact/thickness
# variation and mechanical damage are transduction effects, not chemistry) and it
# is the discriminating signal the model previously never saw — with perfectly
# uniform regions and no defects, the ONLY source of darkness in training was a
# different phase, so any dark pixel had to be read as a new region.
#
# NOTE both are applied BEFORE the detector noise (see __getitem__): the noise is
# deferred out of generate_one_cube and added last, so a scratch does not also
# attenuate the noise floor.

def add_texture(cube, phase_map, rng, sigma=(3.0, 15.0), amp=(0.02, 0.06),
                n_scales=2, clip=0.10):
    """Spatially correlated within-region intensity variation.

    Multi-scale (n_scales Gaussian fields per phase, each with its own blur sigma)
    and INDEPENDENT PER PHASE, so texture does not bleed across a boundary. The
    combined field is clipped to +/- `clip`, so the spectral values are never
    changed by more than that fraction.
    """
    X, Y, _ = cube.shape
    field = np.zeros((X, Y), dtype=np.float32)
    for k in np.unique(phase_map):
        m = (phase_map == k)
        if not m.any():
            continue
        fk = np.zeros((X, Y), dtype=np.float32)
        for _ in range(int(n_scales)):
            g = rng.standard_normal((X, Y)).astype(np.float32)
            g = gaussian_filter(g, sigma=float(rng.uniform(*sigma)))
            sd = float(g.std())
            if sd > 1e-8:
                fk += float(rng.uniform(*amp)) * g / sd
        field[m] = fk[m]
    field = np.clip(field, -clip, clip)
    return np.clip(cube * (1.0 + field)[:, :, None], 0.0, 1.0).astype(np.float32)


def add_scratches(cube, rng, n_scratches=(1, 4), width=(0.8, 2.5),
                  depth=(0.20, 0.60), bright_frac=0.15, curviness=0.25,
                  length=(0.4, 1.2)):
    """Linear mechanical defects: BROADBAND multiplicative attenuation.

    The path is a persistent random walk seeded at a random interior point and
    grown in BOTH directions, each arm stopping the moment it leaves the field.
    Points are never clipped back onto the border — clipping would pin an exiting
    scratch to the edge and draw a spurious line down the side of the image.
    """
    X, Y, _ = cube.shape
    out = cube.astype(np.float32, copy=True)
    total = np.zeros((X, Y), dtype=np.float32)
    diag = float(np.hypot(X, Y))

    n = int(rng.integers(n_scratches[0], n_scratches[1] + 1))
    for _ in range(n):
        r0, c0 = rng.uniform(0, X), rng.uniform(0, Y)
        theta0 = rng.uniform(0, 2 * np.pi)
        budget = diag * float(rng.uniform(*length))
        split = float(rng.uniform(0.2, 0.8))
        pts = []
        for sign, steps in ((+1, budget * split), (-1, budget * (1.0 - split))):
            r, c = r0, c0
            theta = theta0 if sign > 0 else theta0 + np.pi
            for _s in range(int(steps)):
                if not (0.0 <= r < X and 0.0 <= c < Y):
                    break
                pts.append((r, c))
                theta += rng.normal(0.0, curviness * 0.15)
                r += np.sin(theta)
                c += np.cos(theta)
        if len(pts) < 3:
            continue

        pts = np.asarray(pts)
        seed_mask = np.zeros((X, Y), dtype=bool)
        seed_mask[pts[:, 0].astype(int), pts[:, 1].astype(int)] = True
        d = distance_transform_edt(~seed_mask)
        w = float(rng.uniform(*width))
        prof = np.exp(-(d / max(w, 1e-3)) ** 2).astype(np.float32)

        a = float(rng.uniform(*depth))
        if rng.random() < bright_frac:
            a = -a * 0.6                       # bright defect (debris/better contact)
        total += a * prof

    total = np.clip(total, -0.95, 0.95)
    out *= (1.0 - total)[:, :, None]           # BROADBAND: same factor every band
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Not used in published deployment: optional spectral outliers.
def add_outlier_pixels(cube, rng, n_range=(1, 20), boost_frac=0.5,
                       peak_boost=(1.5, 3.5), shift_max=1,
                       peak_prominence=0.05, n_peaks_boost=(1, 2)):
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
    X, Y, W = cube.shape
    out = cube.copy()
    n = int(rng.integers(int(n_range[0]), int(n_range[1]) + 1))
    if n <= 0:
        return out
    n = min(n, X * Y)
    flat = rng.choice(X * Y, size=n, replace=False)
    for fi in flat:
        r, c = divmod(int(fi), Y)
        spec = out[r, c, :].astype(np.float32).copy()
        if rng.random() < boost_frac:
            # ---- TALL-PEAK outlier ----
            idx, _ = find_peaks(spec, prominence=peak_prominence)
            if idx.size == 0:
                idx = np.array([int(np.argmax(spec))])
            kmax = min(int(n_peaks_boost[1]), idx.size)
            k = int(rng.integers(int(n_peaks_boost[0]), kmax + 1))
            tallest = idx[np.argsort(spec[idx])[::-1][:k]]     # boost the tallest peaks
            factor = float(rng.uniform(*peak_boost))
            for pk in tallest:
                lo, hi = max(0, pk - 2), min(W, pk + 3)         # boost the peak window
                spec[lo:hi] = spec[lo:hi] * factor
            out[r, c, :] = spec
        else:
            # ---- RIGID SPECTRAL SHIFT (clamped edges) ----
            sft = int(rng.integers(-int(shift_max), int(shift_max) + 1))
            if sft == 0:
                sft = 1
            rolled = np.roll(spec, sft)
            if sft > 0:
                rolled[:sft] = spec[0]
            else:
                rolled[sft:] = spec[-1]
            out[r, c, :] = rolled
    return out.astype(np.float32)


# =============================================================================
# Deployment-like context generation (hex grid + adaptive points)
# =============================================================================
def hex_grid_mask(H: int, Wsp: int, n_points: int, rng=None, jitter: float = 0.0):
    """Hexagonal lattice of ~n_points sites (the real experiment's opening move).
    `jitter` (in px) randomly perturbs each site so the model does not overfit to a
    perfectly periodic lattice."""
    if n_points < 1:
        return np.zeros((H, Wsp), bool)
    a = np.sqrt(1.1547005383792517 / (n_points / (H * Wsp)))
    dx, dy = a, 0.86602540378 * a
    yc, xc = H / 2.0, Wsp / 2.0
    nc = int(np.ceil((Wsp / 2.0) / dx)) + 2
    nr = int(np.ceil((H / 2.0) / dy)) + 2
    c, r = np.meshgrid(np.arange(-nc, nc + 1), np.arange(-nr, nr + 1))
    x = c * dx + (r % 2) * (dx / 2.0) + xc
    y = r * dy + yc
    pts = np.column_stack([y.ravel(), x.ravel()])
    if jitter > 0 and rng is not None:
        pts = pts + rng.normal(0.0, jitter, pts.shape)
    p = np.round(pts).astype(int)
    ok = (p[:, 0] >= 0) & (p[:, 0] < H) & (p[:, 1] >= 0) & (p[:, 1] < Wsp)
    p = np.unique(p[ok], axis=0)
    m = np.zeros((H, Wsp), bool)
    m[p[:, 0], p[:, 1]] = True
    return m


def apparent_regions(spectra: np.ndarray, bands, tau: float):
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
    P = spectra.shape[0]
    parent = list(range(P))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for p in range(P):
        for q in range(p + 1, P):
            if all(abs(float(spectra[p, l]) - float(spectra[q, l])) < tau for l in bands):
                ra, rb = find(p), find(q)
                if ra != rb:
                    parent[ra] = rb

    roots, lab = {}, np.zeros(P, dtype=int)
    for p in range(P):
        r = find(p)
        if r not in roots:
            roots[r] = len(roots)
        lab[p] = roots[r]
    return lab


def oracle_error_map(cube_raw, obs_mask, phase_map, spectra, bands, tau):
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
    H, Wsp, _ = cube_raw.shape
    lab_of_phase = apparent_regions(spectra, bands, tau)
    region = lab_of_phase[phase_map]                     # (H, Wsp) apparent region id

    fill = np.zeros_like(cube_raw)
    for k in np.unique(region):
        reg = (region == k)
        obs_in_reg = obs_mask & reg
        if not obs_in_reg.any():
            # apparent region with NO sampled spectrum: the oracle cannot fill it.
            # Leave the fill at 0 -> a large error -> the sampler is driven there.
            continue
        _, (iy, ix) = distance_transform_edt(~obs_in_reg, return_distances=True,
                                             return_indices=True)
        f = cube_raw[iy, ix]
        fill[reg] = f[reg]

    return np.abs(cube_raw - fill).mean(axis=2)         # (H, Wsp)


# Not used in published deployment: optional oracle context selection.
def adaptive_context_mask(cube_raw, phase_map, spectra, bands, n_init, n_add, rng,
                          a=1.0, b=1.0, temperature=0.1, batch=8, tau=0.064,
                          hex_jitter=0.5, noise_floor=0.0):
    """Build a DEPLOYMENT-LIKE context: hex grid, then points added the way the
    acquisition loop adds them.

        score(x) = a * norm(oracle_error(x)) + b * norm(distance_to_context(x))
        p(x)    ~ exp(score(x) / temperature)      over UNOBSERVED pixels

    Points are added ITERATIVELY in batches (the score is recomputed after each),
    because real acquisition is recursive: each new point locally suppresses the
    distance term and the error around it, pushing the next pick elsewhere. A
    one-shot draw from the initial map gives a static clustering that looks nothing
    like a real trajectory.

    Selection is STOCHASTIC (softmax with `temperature`), not greedy: the real
    model's sigma is imperfect, so its picks are noisier and more varied than an
    oracle's. Low temperature -> greedy/clustered, high -> nearly uniform. Sweeping
    the temperature sweeps the whole policy family from aggressive-adaptive to
    random, which is what the model must be robust to.
    """
    H, Wsp, _ = cube_raw.shape
    mask = hex_grid_mask(H, Wsp, n_init, rng=rng, jitter=hex_jitter)

    remaining = int(n_add)
    while remaining > 0:
        free = ~mask
        if not free.any():
            break
        err = oracle_error_map(cube_raw, mask, phase_map, spectra, bands, tau)
        if noise_floor > 0:
            # don't chase pixels that are only uncertain because they are NOISY
            err = np.maximum(err - noise_floor, 0.0)
        d = distance_transform_edt(~mask).astype(np.float64)

        def _n(x):
            v = x[free]
            lo, hi = float(v.min()), float(v.max())
            return np.zeros_like(x) if (hi - lo) < 1e-12 else (x - lo) / (hi - lo)

        score = a * _n(err) + b * _n(d)
        logits = score / max(temperature, 1e-3)
        logits[~free] = -np.inf
        logits -= logits[free].max()
        p = np.exp(logits)
        tot = p.sum()
        if not np.isfinite(tot) or tot <= 0:
            break
        p = (p / tot).ravel()

        k = min(batch, remaining, int(free.sum()))
        idx = rng.choice(H * Wsp, size=k, replace=False, p=p)
        mask.ravel()[idx] = True
        remaining -= k

    return mask


class _LibraryView:
    """
    Thin wrapper presenting a (N, W) numpy matrix the way generate_phase_spectra
    expects an `external_library` to behave: supports len() and integer indexing
    returning a (W,) row. Avoids copying the whole matrix into a Python list.
    """
    __slots__ = ("mat",)

    def __init__(self, mat: np.ndarray):
        self.mat = mat

    def __len__(self) -> int:
        return self.mat.shape[0]

    def __getitem__(self, idx):
        # generate_phase_spectra does .copy() on the returned row, so a view is fine
        return self.mat[idx]

    def __bool__(self) -> bool:
        # generate_phase_spectra checks `if external_library:` — must be truthy
        return self.mat.shape[0] > 0


class OnTheFlySynthDataset(Dataset):
    """
    Generates a fresh synthetic cube per item, sampling phase spectra from a
    pre-normalized NIST matrix. Returns (cube, obs_mask) like CACNPDataset.

    Args:
        spectrum_library : (N, W) float32, ALREADY NORMALIZED. Rows are spectra.
        wavenumbers      : (W,) float32. Length must match spectrum_library W.
        image_size       : (X, Y) spatial size of generated cubes.
        n_tasks_per_epoch: number of items per epoch (defines __len__).
        param_ranges     : dict of generation ranges (see DEFAULT_PARAM_RANGES).
        norm_mode        : input normalization ('context_zscore' etc.).
        n_context / n_context_min / log_uniform_context : context-size sampling,
                            identical semantics to CACNPDataset.
        seed             : base RNG seed.
        split            : 'train' or 'val'. 'val' uses a FIXED seed offset and
                            FIXED per-index seeds so the validation set is the
                            same cubes every epoch (stable val metric).
        epoch            : current epoch (mixed into the train seed so each epoch
                            sees new cubes; set via set_epoch()).
    """

    def __init__(
        self,
        spectrum_library: np.ndarray,
        wavenumbers:      np.ndarray,
        image_size:       Tuple[int, int],
        n_tasks_per_epoch: int,
        param_ranges:     dict,
        norm_mode:        str = "context_zscore",
        n_context:        int = 4000,
        n_context_min:    int = 4,
        context_dist:     str = 'log_norm',
        mode_frac:        float = 0.05,
        tail_frac:        float = 0.20,
        tail_quantile:    float = 0.99,
        seed:             int = 42,
        split:            str = "train",
        epoch:            int = 0,
        dynamic_image_band: bool = False,
        image_peak_height:  float = 0.2,
        multi_image:        bool = True,
        texture_prob:       float = 1.0,        # fraction of cubes given texture
        texture_sigma:      Sequence[float] = (3.0, 15.0),
        texture_amp:        Sequence[float] = (0.02, 0.06),
        texture_clip:       float = 0.10,
        texture_scales:     int = 2,
        scratch_prob:       float = 0.3,        # fraction of cubes given scratches
        scratch_count:      Sequence[int] = (1, 4),
        scratch_width:      Sequence[float] = (0.8, 2.5),
        scratch_depth:      Sequence[float] = (0.20, 0.60),
        outlier_n_range:    Sequence[int] = (0, 0),   # per-cube #anomalous pixels; (0,0)=off
        outlier_boost_frac: float = 0.5,              # P(tall-peak) per anomalous pixel; else shift
        outlier_peak_boost: Sequence[float] = (1.5, 3.5),
        outlier_shift_max:  int = 1,
        context_mode:       str = "random",     # random | hex | adaptive | mixture
        context_mix:        Optional[Sequence[float]] = None,  # P(random,hex,adaptive)
        adaptive_tau:       float = 0.064,      # phase indistinguishability threshold
        adaptive_noise_floor: float = 0.0,      # subtract from oracle error
        image_var_thresh:   float = 0.2,
        image_count_weights: Optional[Sequence[float]] = None,
        max_images:         int = 4,
    ):
        if normalize_cube is None:
            raise ImportError("normalize_cube could not be imported from "
                              "conv_cnp_afmir_prob; check import order.")
        if norm_mode not in NORM_MODES:
            raise ValueError(f"norm_mode must be one of {NORM_MODES}")
        lib = np.asarray(spectrum_library, dtype=np.float32)
        if lib.ndim != 2:
            raise ValueError(f"spectrum_library must be (N, W); got {lib.shape}")
        W_lib = lib.shape[1]
        W_wv  = int(len(wavenumbers))
        if W_lib != W_wv:
            raise ValueError(f"spectrum_library W ({W_lib}) != wavenumbers W "
                             f"({W_wv}); they must be on the same grid.")

        self.library        = _LibraryView(lib)
        self.wavenumbers    = np.asarray(wavenumbers, dtype=np.float32)
        self.X, self.Y      = int(image_size[0]), int(image_size[1])
        self.n_tasks        = int(n_tasks_per_epoch)
        self.ranges         = param_ranges
        self.norm_mode      = norm_mode
        self.n_context      = int(n_context)
        self.n_context_min  = int(n_context_min)
        self.context_dist   = context_dist
        self.mode_frac      = mode_frac
        self.tail_frac      = tail_frac
        self.tail_quantile  = tail_quantile
        self.base_seed      = int(seed)
        self.split          = split
        self.epoch          = int(epoch)
        self.dynamic_image_band = bool(dynamic_image_band)
        self.image_peak_height  = float(image_peak_height)
        self.multi_image        = bool(multi_image)
        self.texture_prob       = float(texture_prob)
        self.texture_sigma      = tuple(texture_sigma)
        self.texture_amp        = tuple(texture_amp)
        self.texture_clip       = float(texture_clip)
        self.texture_scales     = int(texture_scales)
        self.scratch_prob       = float(scratch_prob)
        self.scratch_count      = tuple(scratch_count)
        self.scratch_width      = tuple(scratch_width)
        self.scratch_depth      = tuple(scratch_depth)
        self.outlier_n_range    = tuple(outlier_n_range)
        self.outlier_boost_frac = float(outlier_boost_frac)
        self.outlier_peak_boost = tuple(outlier_peak_boost)
        self.outlier_shift_max  = int(outlier_shift_max)
        self.context_mode       = str(context_mode)
        self.context_mix        = (list(context_mix) if context_mix is not None
                                   else [0.40, 0.20, 0.40])   # random / hex / adaptive
        self.adaptive_tau       = float(adaptive_tau)
        self.adaptive_noise_floor = float(adaptive_noise_floor)
        self.image_var_thresh   = float(image_var_thresh)
        self.max_images         = int(max_images)
        self.image_count_weights = (list(image_count_weights)
                                    if image_count_weights is not None
                                    else [0.4, 0.3, 0.2, 0.1])
        self._worker_offset = 0  # set by worker_init_fn

    # -- epoch control: call before each train epoch so cubes differ per epoch --
    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_tasks

    def _item_seed(self, idx: int) -> int:
        """
        Deterministic, independent seed per (split, epoch, worker, idx).
        Validation uses epoch=0 always, so val cubes are identical every epoch.
        """
        epoch = 0 if self.split == "val" else self.epoch
        # SeedSequence mixes the streams cleanly; entropy list keeps streams
        # independent across workers and epochs.
        ss = np.random.SeedSequence([
            self.base_seed,
            0 if self.split == "val" else 1,
            epoch,
            self._worker_offset,
            idx,
        ])
        return int(ss.generate_state(1)[0])

    def _sample_params(self, rng: np.random.Generator) -> dict:
        """Draw one generation parameter dict from the configured ranges."""
        r = self.ranges
        n_phases = int(rng.integers(r["n_phases"][0], r["n_phases"][1] + 1))
        n_feat   = n_phases - 1

        base_feat = float(rng.uniform(*r["feature_size"]))
        base_curv = float(rng.uniform(*r["boundary_curvature"]))
        feature_sizes = [
            float(np.clip(base_feat * rng.uniform(0.70, 1.30),
                          r["feature_size"][0], r["feature_size"][1]))
            for _ in range(n_feat)
        ]
        curvatures = [
            float(np.clip(base_curv * rng.uniform(0.85, 1.15),
                          r["boundary_curvature"][0], r["boundary_curvature"][1]))
            for _ in range(n_feat)
        ]
        # phase fractions sum <= 0.85 (>=15% background), matching the generator
        phase_fractions = []
        remaining = 0.85
        for j in range(n_feat):
            hi   = min(r["phase_fraction"][1], remaining / max(1, n_feat - j))
            lo   = min(r["phase_fraction"][0], hi)
            frac = float(rng.uniform(lo, hi))
            phase_fractions.append(frac)
            remaining -= frac

        return dict(
            cube_id               = -1,
            X                     = self.X,
            Y                     = self.Y,
            n_phases              = n_phases,
            feature_sizes         = feature_sizes,
            curvatures            = curvatures,
            phase_fractions       = phase_fractions,
            morphology            = str(rng.choice(r["morphology_types"])),
            boundary_width        = float(rng.uniform(*r["boundary_width"])),
            mixing_type           = str(rng.choice(r["mixing_types"])),
            noise_std             = float(rng.uniform(*r["noise_std"])),
            intra_phase_variation = float(rng.uniform(*r["intra_phase_variation"])),
            spectra_mode          = "external",       # always sample from NIST matrix
            # spectral perturbation: 0 by default — the NIST library already
            # provides diversity, and the data is pre-normalized so we don't want
            # _perturb_spectrum re-scaling it. Overridable via ranges.
            n_peaks_range         = list(r.get("n_peaks", [2, 8])),
            peak_width_range      = list(r.get("peak_width", [20.0, 100.0])),
            peak_amp_range        = list(r.get("peak_amp_range", [0.3, 1.0])),
            spectra_shift_std     = float(r.get("spectra_shift_std", 0.0)),
            spectra_width_jitter  = float(r.get("spectra_width_jitter", 0.0)),
            spectra_amp_jitter    = float(r.get("spectra_amp_jitter", 0.0)),
            # depth (subsurface) mixing.
            # PHYSICAL model: set depth_probing_depth to a [lo,hi] range (nm) to
            # enable Beer-Lambert screening (exp(-thickness/probing_depth)) with a
            # per-particle thickness sampled from depth_thickness_range (nm). The
            # legacy edge-distance model is used only when depth_probing_depth is
            # absent AND depth_edge_mix is nonzero.
            depth_edge_mix        = float(rng.uniform(*r.get("depth_edge_mix", [0.0, 0.0]))),
            depth_decay_length    = float(rng.uniform(*r.get("depth_decay_length", [6.0, 6.0]))),
            depth_probing_depth_nm= (float(rng.uniform(*r["depth_probing_depth"]))
                                     if r.get("depth_probing_depth") is not None else None),
            depth_thickness_range = list(r.get("depth_thickness_range", [200.0, 500.0])),
            depth_thickness_profile = str(r.get("depth_thickness_profile", "sqrt")),
            depth_background_phase= 0,
            # per-phase RELATIVE INTENSITY: sample each phase's max peak height
            # uniformly from this range per cube (None -> all 1.0). clip_output
            # False keeps peaks that exceed 1 (only relevant if the range goes >1).
            relative_intensity_range = (list(r["relative_intensity"])
                                        if r.get("relative_intensity") is not None else None),
            clip_output           = bool(r.get("clip_output", True)),
            # generate_one_cube seeds its own rng from this; give it our item seed
            seed                  = int(rng.integers(0, 2 ** 31)),
        )

    def __getitem__(self, idx: int):
        item_seed = self._item_seed(idx)
        rng = np.random.default_rng(item_seed)

        # 1. sample generation parameters + generate one cube on the fly
        params = self._sample_params(rng)

        # DETECTOR NOISE IS DEFERRED. generate_one_cube would otherwise add it
        # inside assemble_cube, i.e. BEFORE the texture/scratch artifacts — which
        # would let a scratch attenuate the noise floor along with the signal.
        # Real detector noise is added last, so we pull it out and apply it after.
        noise_std = float(params["noise_std"])
        params = dict(params)
        params["noise_std"] = 0.0

        out = generate_one_cube(params, self.wavenumbers, self.library)
        cube_raw = np.asarray(out["cube"], dtype=np.float32)   # (X, Y, W)

        # ---- post-assembly artifacts (both broadband + multiplicative) --------
        if self.texture_prob > 0.0 and rng.random() < self.texture_prob:
            cube_raw = add_texture(
                cube_raw, np.asarray(out["phase_map"]), rng,
                sigma=self.texture_sigma, amp=self.texture_amp,
                n_scales=self.texture_scales, clip=self.texture_clip)
        if self.scratch_prob > 0.0 and rng.random() < self.scratch_prob:
            cube_raw = add_scratches(
                cube_raw, rng, n_scratches=self.scratch_count,
                width=self.scratch_width, depth=self.scratch_depth)

        # ---- detector noise LAST ---------------------------------------------
        if noise_std > 0.0:
            cube_raw = cube_raw + rng.normal(0.0, noise_std,
                                             cube_raw.shape).astype(np.float32)
        cube_raw = np.clip(cube_raw, 0.0, 1.0).astype(np.float32)

        # Not used in published deployment (outlier_n_range=(0, 0)).
        # ---- anomalous pixels LAST: applied AFTER the clip so a tall-peak
        #      outlier can exceed 1 (the out-of-distribution case). Sparse. ----
        if self.outlier_n_range[1] > 0:
            cube_raw = add_outlier_pixels(
                cube_raw, rng, n_range=self.outlier_n_range,
                boost_frac=self.outlier_boost_frac,
                peak_boost=self.outlier_peak_boost,
                shift_max=self.outlier_shift_max)
        H, Wsp, Wv = cube_raw.shape
        n_pix = H * Wsp

        # 2. sample context size + build observation mask (same as CACNPDataset)
        if self.context_dist == 'log_uni':
            log_lo = np.log(max(self.n_context_min, 1))
            log_hi = np.log(self.n_context + 1)
            n_ctx = int(np.clip(
                np.round(np.exp(rng.uniform(log_lo, log_hi))),
                self.n_context_min, self.n_context,
            ))
        elif self.context_dist == 'log_norm':
            mu, sigma, _, _ = lognormal_from_mode_tail(
                n_pix=n_pix,
                mode_frac=self.mode_frac,
                tail_frac=self.tail_frac,
                tail_quantile=self.tail_quantile,
            )

            n_ctx = int(
                _rejection_truncate(
                    lambda r, k: np.round(r.lognormal(mean=mu, sigma=sigma, size=k)),
                    self.n_context_min,
                    self.n_context,
                    1,
                    rng,
                )[0]
            )
        else:
            n_ctx = int(rng.integers(self.n_context_min, self.n_context + 1))

        n_ctx = min(n_ctx, n_pix)

        # ---- build the observation mask -------------------------------------
        # context_mode controls WHAT KIND of context the model is trained on.
        #   random   : uniform random pixels (the original; keeps the model general)
        #   hex      : a hexagonal grid  (the experiment's OPENING MOVE)
        #   adaptive : hex grid + points added the way the acquisition loop adds
        #              them (an image-aware GT oracle stands in for the model's
        #              sigma, at zero forward passes)
        #   mixture  : draw one of the three per cube, per self.context_mix
        #
        # WHY: the deployed model never sees a uniform-random context. It starts on
        # a hex grid and then receives increasingly CLUSTERED contexts as the
        # acquisition loop adds points. A model trained only on uniform-random masks
        # is out-of-distribution for its own acquisition policy — which degrades
        # both the mean and (worse) sigma, the very signal the loop consumes.
        mode = self.context_mode
        if mode == "mixture":
            w = np.asarray(self.context_mix, float); w = w / w.sum()
            mode = str(rng.choice(["random", "hex", "adaptive"], p=w))

        if mode == "random":
            flat_perm = rng.permutation(n_pix)
            obs_flat = np.zeros(n_pix, dtype=bool)
            obs_flat[flat_perm[:n_ctx]] = True
            obs_mask = obs_flat.reshape(H, Wsp)

        elif mode == "hex":
            obs_mask = hex_grid_mask(H, Wsp, n_ctx, rng=rng,
                                     jitter=float(rng.uniform(0.0, 1.0)))

        elif mode == "adaptive":
            # RANDOMIZE THE POLICY FAMILY. Training on contexts from ONE acquisition
            # rule would overfit the model to that rule — fatal when several rules
            # are to be compared at deployment. So per cube we draw the error/
            # distance weights, the greediness, the split between the hex core and
            # the adaptive points, and the grid jitter.
            a_w   = float(rng.uniform(0.0, 1.0))          # weight on oracle error
            b_w   = float(rng.uniform(0.0, 1.0))          # weight on distance
            if a_w + b_w < 1e-3:                          # avoid the degenerate 0/0
                a_w, b_w = 1.0, 1.0
            temp  = float(np.exp(rng.uniform(np.log(0.02), np.log(1.0))))  # log-uniform
            tau   = float(self.adaptive_tau * rng.uniform(0.75, 1.5))
            # how far along the acquisition TRAJECTORY is this cube? Uniform, so the
            # model is calibrated at EVERY step the loop passes through — not just
            # at the endpoint.
            frac_init = float(rng.uniform(0.25, 1.0))
            n_init = max(4, int(round(n_ctx * frac_init)))
            n_add  = max(0, n_ctx - n_init)

            phase_map = np.asarray(out["phase_map"])
            phase_spec = np.asarray(out["spectra"])       # (n_phases, W) endmembers

            # the imaged band(s) are chosen from the HEX spectra — exactly as at
            # deployment (image first, then acquire), so the oracle knows what the
            # model will know.
            init_mask = hex_grid_mask(H, Wsp, n_init, rng=rng, jitter=0.5)
            sampled0 = cube_raw[init_mask]
            ranked0 = variance_peak_bands(sampled0, self.image_var_thresh)
            if len(ranked0) == 0:
                ranked0 = np.array([int(sampled0.var(axis=0).argmax())]) \
                          if sampled0.shape[0] >= 2 else np.array([Wv // 2])
            n_img0 = 1
            if self.multi_image:
                w = np.asarray(self.image_count_weights, float); w = w / w.sum()
                n_img0 = int(rng.choice(np.arange(1, len(w) + 1), p=w))
            oracle_bands = [int(b) for b in ranked0[:max(1, min(n_img0, len(ranked0)))]]

            obs_mask = adaptive_context_mask(
                cube_raw, phase_map, phase_spec, oracle_bands,
                n_init=n_init, n_add=n_add, rng=rng,
                a=a_w, b=b_w, temperature=temp, batch=8, tau=tau,
                hex_jitter=0.5, noise_floor=self.adaptive_noise_floor)
        else:
            raise ValueError(f"unknown context_mode {self.context_mode!r}")

        if not obs_mask.any():                     # never hand the model an empty context
            obs_mask.ravel()[int(rng.integers(n_pix))] = True

        # 3. normalize with the selected mode (using the context for context_*)
        cube, _stats = normalize_cube(cube_raw, obs_mask, mode=self.norm_mode)

        # 4. safety guard: on-the-fly cubes are never inspected by a human, so a
        #    malformed cube must fail loudly rather than inject NaNs into training
        cube = np.ascontiguousarray(cube, dtype=np.float32)
        if not np.isfinite(cube).all():
            raise FloatingPointError(
                f"non-finite values in generated cube (idx={idx}, seed={item_seed}, "
                f"n_phases={params['n_phases']}, morph={params['morphology']})")

        # 5. dynamic imaged-band selection (mimics the experimental workflow:
        #    place the probe at a random position, capture a full RAW spectrum,
        #    find its peaks, and image one peak at random). Peak-finding is on the
        #    raw [0,1] absorbance so the height threshold is meaningful. band = -1
        #    means "no dynamic band" (fixed-band / no-image runs handle it later).
        if self.multi_image:
            # variance-based selection on the SAMPLED (observed) spectra, mimicking
            # the experimental hexagonal-grid workflow: analyse between-point
            # variance, pick the highest-variance peaks as structure-image bands.
            # Number of images ~ weighted(1..max_images); the top-N variance peaks
            # are imaged. Padded to max_images with a validity count.
            sampled = cube_raw[obs_mask]                          # (n_obs, W)
            ranked = variance_peak_bands(sampled, self.image_var_thresh)
            if len(ranked) == 0:                                 # fallback: top-variance band
                ranked = np.array([int(sampled.var(axis=0).argmax())]) if sampled.shape[0] >= 2 \
                         else np.array([int(np.argmax(cube_raw[H // 2, Wsp // 2]))])
            counts = np.arange(1, len(self.image_count_weights) + 1)
            w = np.asarray(self.image_count_weights, float); w = w / w.sum()
            N = int(rng.choice(counts, p=w))
            N = max(1, min(N, len(ranked), self.max_images))
            bands = ranked[:N].astype(int)
            bands_pad = np.zeros(self.max_images, dtype=np.int64)
            bands_pad[:N] = bands
            return (torch.from_numpy(cube), torch.from_numpy(obs_mask),
                    torch.from_numpy(bands_pad), torch.tensor(N, dtype=torch.long))

        if self.dynamic_image_band:
            pr = int(rng.integers(H)); pc = int(rng.integers(Wsp))
            spec = cube_raw[pr, pc, :]
            peaks, _ = find_peaks(spec, height=self.image_peak_height)
            band = int(rng.choice(peaks)) if len(peaks) > 0 else int(np.argmax(spec))
        else:
            band = -1

        return (torch.from_numpy(cube), torch.from_numpy(obs_mask),
                torch.tensor(band, dtype=torch.long))


# Default generation ranges (mirrors the relative-intensity dataset settings).
DEFAULT_PARAM_RANGES = dict(
    n_phases            = [2, 4],
    feature_size        = [1.5, 40.0],
    boundary_curvature  = [0.4, 1.0],
    phase_fraction      = [0.002, 0.4],
    morphology_types    = ["dispersed", "voronoi"],
    boundary_width      = [0.0, 0.0],
    mixing_types        = ["linear"],
    noise_std           = [0.01, 0.08],
    intra_phase_variation = [0.0, 0.0],
    # spectral perturbation off: NIST library supplies diversity, data is
    # already normalized (don't let _perturb_spectrum rescale it).
    spectra_shift_std   = 0.0,
    spectra_width_jitter = 0.0,
    spectra_amp_jitter  = 0.0,
    # depth (subsurface) mixing.
    #   PHYSICAL model: Not used in published deployment. Optionally set depth_probing_depth=[lo,hi] (nm) to
    #   enable it; each cube samples a probing depth, each particle a thickness
    #   from depth_thickness_range (nm). depth_probing_depth=None disables it.
    #   LEGACY edge model: depth_edge_mix/depth_decay_length (used only when
    #   depth_probing_depth is None and depth_edge_mix is nonzero).
    depth_edge_mix      = [0.0, 1.0],
    depth_decay_length  = [0.0, 10.0],
    depth_probing_depth = None,            # e.g. [100.0, 300.0] to enable physical depth
    depth_thickness_range = [200.0, 500.0],
    depth_thickness_profile = "sqrt",      # "sqrt" (puddle) | "linear" (wedge)
    # per-phase relative intensity: None -> all 1.0; e.g. [0.5, 1.0] to vary
    relative_intensity  = None,
    clip_output         = True,
)


def _initialize_worker(worker_id: int, base_seed: int):
    """Top-level worker callback; preserves cluster seeds and supports spawn."""
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    if hasattr(ds, "_worker_offset"):
        ds._worker_offset = int(worker_id) + 1
    seed = (int(base_seed) + 100003 * (int(worker_id) + 1)) % (2 ** 31)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_worker_init_fn(base_seed: int):
    """Picklable worker initializer (Linux, macOS and Windows)."""
    from functools import partial
    return partial(_initialize_worker, base_seed=base_seed)


def load_spectrum_library(path: str) -> np.ndarray:
    """Load the pre-normalized NIST (N, W) matrix from .npy."""
    mat = np.load(path)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"spectrum library at {path} must be 2D (N, W); "
                         f"got shape {mat.shape}")
    return mat

def _rejection_truncate(draw_fn, lo, hi, n_samples, rng, max_iter=200):
    """draw_fn(rng, k) -> (k,) array of (already-rounded) candidates."""
    out = np.empty(0, dtype=float)
    for _ in range(max_iter):
        need = n_samples - out.size
        if need <= 0:
            break
        cand = draw_fn(rng, int(need * 1.4) + 16)
        cand = cand[(cand >= lo) & (cand <= hi)]
        out = np.concatenate([out, cand])
    if out.size < n_samples:
        raise RuntimeError("rejection sampling failed to fill; widen bounds/spread")
    return out[:n_samples]

def lognormal_from_mode_tail(
    n_pix:         int,
    mode_frac:     float = 0.05,   # coverage at the density PEAK (true mode)
    tail_frac:     float = 0.20,   # coverage at the upper-tail quantile
    tail_quantile: float = 0.99,   # which quantile tail_frac represents
):
    """
    Return (mu, sigma, n_mode, n_tail) for log X ~ N(mu, sigma^2) such that:
        mode of X      = exp(mu - sigma^2) = mode_frac * n_pix
        the q-quantile = exp(mu + sigma*z_q) = tail_frac * n_pix
    Subtracting the two log-equations gives a quadratic in sigma:
        sigma^2 + z_q*sigma - ln(n_tail/n_mode) = 0.
    """
    n_mode = mode_frac * n_pix
    n_tail = tail_frac * n_pix
    ratio  = n_tail / n_mode
    if ratio <= 1.0:
        raise ValueError("tail_frac must exceed mode_frac")
    z = NormalDist().inv_cdf(tail_quantile)
    sigma = (-z + math.sqrt(z * z + 4.0 * math.log(ratio))) / 2.0
    mu    = math.log(n_mode) + sigma * sigma
    return mu, sigma, n_mode, n_tail
