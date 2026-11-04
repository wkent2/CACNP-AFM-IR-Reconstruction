#!/usr/bin/env python3
"""
generate_synth_afmir.py
=======================
Base morphology/spectral generator used by CACNP on-the-fly training.

The standalone CLI below is an optional demonstration, NOT the published
training pipeline. Published defaults are in conv_cnp_afmir_prob.py and
synth_onthefly.py; training calls generate_one_cube with explicit parameters.
Standalone Gaussian spectra and CSV-directory sampling: Not used in published deployment.

Each cube is a (X, Y, W) float32 array normalised to [0, 1].  The spatial
structure is produced by Voronoi-guided ellipse placement (dispersed inclusions
in a matrix) or Voronoi grain tessellation.  Each phase is assigned a spectrum
(synthetic Gaussian peaks or loaded from external CSV/TXT files), and boundary
pixels receive a user-controlled spectral mix.

Two spectrum modes
------------------
  synthetic  : Gaussian peaks placed at random positions within [wv_min, wv_max].
               Each phase gets an independent random spectrum.

  external   : Load 2-column (wavenumber, intensity) CSV/TXT files from a
               directory.  Files are interpolated and optionally Gaussian-
               smoothed onto the target wavenumber grid.  Assign randomly
               (or cyclically) to phases.

Output layout
-------------
  <output_dir>/synth_afmir_YYYYMMDD_HHMMSS/
    dataset_params.txt           global CLI parameters
    cube_0000/
      hyperspectral_cube.npy     (X, Y, W) float32
      phase_map.npy              (X, Y)   int32
      spectra.npy                (n_phases, W) float32
      wavenumbers.npy            (W,) float32  [cm-1]
      params.txt                 per-cube parameters
    cube_0001/ ...

Usage examples
--------------
  # 50 synthetic cubes, 32×32 pixels, 104 wavenumbers, 2–3 phases:
  python generate_synth_afmir.py --n_cubes 50

  # External spectra from a directory:
  python generate_synth_afmir.py --n_cubes 50 \
      --spectra_mode external --spectra_dir ./my_spectra

  # Override image/spectral grid:
  python generate_synth_afmir.py --n_cubes 100 \
      --image_size 64 64 --wv_min 900 --wv_max 3200 --n_wavenumbers 150

See generate_synth_afmir.md for full documentation.
"""

# =============================================================================
# Imports
# =============================================================================

import os
import sys
import math
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage, interpolate, spatial

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn("matplotlib not found — phase-map PNGs will not be saved.")


# =============================================================================
# Lightweight geometry (adapted from microgeneration.py)
# =============================================================================

class Circle:
    def __init__(self, center, r):
        self.center = np.asarray(center, dtype=float)
        self.r = float(r)

    def within(self, pts: np.ndarray) -> np.ndarray:
        d = pts - self.center
        return (d[:, 0] ** 2 + d[:, 1] ** 2) <= self.r ** 2


class Ellipse:
    """Rotated ellipse.  angle is CCW from +x axis in degrees."""
    def __init__(self, center, a, b, angle=0.0):
        self.center = np.asarray(center, dtype=float)
        self.a = float(a)
        self.b = float(b)
        self.angle = float(angle)

    def within(self, pts: np.ndarray) -> np.ndarray:
        theta = math.radians(self.angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dx = pts[:, 0] - self.center[0]
        dy = pts[:, 1] - self.center[1]
        xr =  cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        return (xr / self.a) ** 2 + (yr / self.b) ** 2 <= 1.0


# =============================================================================
# Phase map generation
# =============================================================================

def _voronoi_inscribed_radius(vor, idx: int, fallback: float = 20.0) -> float:
    """Minimum distance from seed idx to any of its Voronoi ridge vertices."""
    seed = vor.points[idx]
    verts = []
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        if idx in (p1, p2):
            if v1 >= 0:
                verts.append(vor.vertices[v1])
            if v2 >= 0:
                verts.append(vor.vertices[v2])
    if not verts:
        return fallback
    return float(min(np.linalg.norm(seed - v) for v in verts))


def _dispersed_phase_map(
    X: int, Y: int,
    n_phases: int,
    feature_sizes: List[float],
    curvatures: List[float],
    phase_fractions: List[float],
    rng: np.random.Generator,
    all_pts: np.ndarray,
) -> np.ndarray:
    """
    Background (phase 0) fills the canvas; each non-background phase is placed
    as a collection of Voronoi-spaced ellipses.  Later phases overwrite earlier
    ones, creating natural overlaps at phase boundaries.
    """
    label_flat = np.zeros(X * Y, dtype=np.int32)
    # `beneath_flat[p]` = the label directly UNDER the current top label at pixel p.
    # Updated whenever a DIFFERENT-phase ellipse paints over an existing label, so
    # it records what the top particle is sitting on (an earlier particle, or the
    # background). Stays background where nothing was overwritten.
    beneath_flat = np.zeros(X * Y, dtype=np.int32)

    def _paint(hit, phase_idx):
        # record what we are covering, only where we actually CHANGE the label to a
        # different phase (same-phase overlaps carry the same spectrum -> no mix)
        prev = label_flat[hit]
        changed = hit.copy()
        changed[hit] = (prev != phase_idx)
        beneath_flat[changed] = label_flat[changed]     # what was here before
        label_flat[hit] = phase_idx

    for phase_idx in range(1, n_phases):
        feat = feature_sizes[phase_idx - 1]
        curv = curvatures[phase_idx - 1]
        frac = phase_fractions[phase_idx - 1]

        # Estimate number of features from target area fraction
        area_per = math.pi * feat ** 2 * curv
        n_feat = max(1, round(frac * X * Y / max(area_per, 1.0)))

        margin = max(2.0, feat * 0.25)
        seeds = rng.uniform(
            low=[margin, margin],
            high=[max(margin + 1, X - margin), max(margin + 1, Y - margin)],
            size=(n_feat, 2),
        )

        if n_feat >= 4:
            mirrors = np.array([
                [-X, Y / 2], [2 * X, Y / 2],
                [X / 2, -Y], [X / 2, 2 * Y],
            ])
            vor = spatial.Voronoi(np.vstack([seeds, mirrors]))
            for i in range(n_feat):
                r_ins = _voronoi_inscribed_radius(vor, i, fallback=feat)
                a = float(np.clip(feat * rng.uniform(0.70, 1.30), 1.5, r_ins * 0.90))
                b = float(max(1.0, a * curv * rng.uniform(0.85, 1.15)))
                angle = float(rng.uniform(0, 180))
                _paint(Ellipse(seeds[i], a, b, angle).within(all_pts), phase_idx)
        else:
            for i in range(n_feat):
                a = float(max(1.5, feat * rng.uniform(0.70, 1.30)))
                b = float(max(1.0, a * curv))
                angle = float(rng.uniform(0, 180))
                _paint(Ellipse(seeds[i], a, b, angle).within(all_pts), phase_idx)

    return label_flat.reshape(X, Y), beneath_flat.reshape(X, Y)


def _voronoi_grain_map(
    X: int, Y: int,
    n_phases: int,
    feature_sizes: List[float],
    rng: np.random.Generator,
    all_pts: np.ndarray,
) -> np.ndarray:
    """
    Voronoi tessellation where each grain cell is randomly assigned to a phase.
    Mean feature size controls grain density.
    """
    mean_feat = float(np.mean(feature_sizes)) if feature_sizes else max(X, Y) / 4
    n_seeds = max(4, int(X * Y / max(1.0, math.pi * mean_feat ** 2)))

    seeds = rng.uniform([0, 0], [X, Y], size=(n_seeds, 2))
    seed_phases = rng.integers(0, n_phases, size=n_seeds)

    tree = spatial.cKDTree(seeds)
    _, nearest = tree.query(all_pts, workers=1)
    nearest = np.clip(nearest, 0, n_seeds - 1)
    return seed_phases[nearest].astype(np.int32).reshape(X, Y)


def generate_phase_map(
    X: int,
    Y: int,
    n_phases: int,
    feature_sizes: List[float],
    curvatures: List[float],
    phase_fractions: List[float],
    morphology: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate an (X, Y) integer phase label map.

    Phase 0 = background (matrix).  Phases 1..n_phases-1 = dispersed or grain phases.

    Args:
        feature_sizes       : mean feature radius in pixels for each non-background phase
        curvatures          : ellipse aspect ratio b/a ∈ (0,1] per non-background phase
                              (1.0 = circle, lower = more elongated)
        phase_fractions     : target area fraction for each non-background phase
        morphology          : 'dispersed' or 'voronoi'
    """
    yy, xx = np.mgrid[0:X, 0:Y]
    all_pts = np.column_stack([xx.ravel(), yy.ravel()])

    if morphology == "voronoi":
        pm = _voronoi_grain_map(X, Y, n_phases, feature_sizes, rng, all_pts)
        beneath = np.zeros_like(pm)          # grains do not stack -> nothing beneath
        return pm, beneath
    else:
        return _dispersed_phase_map(
            X, Y, n_phases, feature_sizes, curvatures, phase_fractions, rng, all_pts
        )


# =============================================================================
# Spectrum generation
# =============================================================================

def synthetic_gaussian_spectrum(
    wv: np.ndarray,
    n_peaks: int,
    peak_width_range: Tuple[float, float],
    rng: np.random.Generator,
    peak_amp_range: Tuple[float, float] = (0.2, 1.0),
) -> np.ndarray:
    """
    Build a synthetic spectrum as a sum of Gaussian peaks.

    Peaks are placed uniformly at random within wv.  Width (FWHM) is drawn
    from peak_width_range in cm⁻¹.  Amplitude is drawn uniformly from
    peak_amp_range; lowering the minimum allows short minor peaks relative
    to the dominant peak.  Output is normalised to [0, 1].
    """
    wv_min, wv_max = float(wv.min()), float(wv.max())
    spec = np.zeros(len(wv), dtype=np.float64)
    for _ in range(n_peaks):
        center = rng.uniform(wv_min, wv_max)
        fwhm = rng.uniform(*peak_width_range)
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        amp = rng.uniform(*peak_amp_range)
        spec += amp * np.exp(-0.5 * ((wv - center) / sigma) ** 2)
    if spec.max() > 0:
        spec /= spec.max()
    return spec.astype(np.float32)


def _try_load_two_column(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a 2-column (wavenumber, intensity) file.  Handles:
      - CSV with optional single header row
      - Whitespace-separated TXT
      - Lines beginning with '#' ignored
    Raises ValueError if format cannot be parsed.
    """
    path = Path(filepath)
    for skip in (0, 1):
        try:
            data = np.loadtxt(filepath, comments="#", skiprows=skip)
            if data.ndim == 2 and data.shape[1] >= 2:
                wv = data[:, 0].astype(np.float64)
                intensity = data[:, 1].astype(np.float64)
                if len(wv) > 5 and not np.all(wv == wv[0]):
                    return wv, intensity
        except Exception:
            continue
    raise ValueError(f"Cannot parse spectrum file: {filepath}")


def load_external_spectrum(
    filepath: str,
    wv_target: np.ndarray,
    sigma_smooth: float = 0.0,
) -> np.ndarray:
    """
    Load a spectrum from a 2-column CSV/TXT file and interpolate to wv_target.

    Args:
        filepath      : path to the spectrum file
        wv_target     : target wavenumber array (cm⁻¹)
        sigma_smooth  : Gaussian smoothing width in cm⁻¹ before interpolation
                        (≈ instrument resolution difference).  0 = no smoothing.

    Returns:
        spectrum : (len(wv_target),) float32 — RAW interpolated intensity
                   (NOT per-spectrum normalized). Relative scale across the
                   loaded library is preserved; a single SHARED normalization is
                   applied once in load_external_library so inter-phase intensity
                   differences survive into the generated cubes.
    """
    wv_file, spec_file = _try_load_two_column(filepath)

    # Sort ascending in wavenumber
    sort_idx = np.argsort(wv_file)
    wv_file, spec_file = wv_file[sort_idx], spec_file[sort_idx]

    # Optional smoothing on the original grid
    if sigma_smooth > 0:
        dw = float(wv_file[1] - wv_file[0]) if len(wv_file) > 1 else 1.0
        sigma_px = sigma_smooth / max(abs(dw), 1e-6)
        spec_file = ndimage.gaussian_filter1d(spec_file, sigma=sigma_px)

    # Interpolate onto target grid; edge-fill outside range
    interp_fn = interpolate.interp1d(
        wv_file, spec_file,
        kind="linear",
        bounds_error=False,
        fill_value=(float(spec_file[0]), float(spec_file[-1])),
    )
    spec = interp_fn(wv_target).astype(np.float64)

    # NO per-spectrum normalization here. Returning raw interpolated intensity
    # preserves the relative heights BETWEEN experimental spectra. (Previously
    # each spectrum was independently scaled to [0,1], forcing every phase to
    # peak at 1 and erasing inter-phase intensity contrast.)
    return spec.astype(np.float32)


def _perturb_spectrum(
    spectrum: np.ndarray,
    wv: np.ndarray,
    shift_std: float,
    width_jitter: float,
    amp_jitter: float,
    rng: np.random.Generator,
    clip: bool = True,
) -> np.ndarray:
    """
    Slightly perturb a spectrum: wavenumber shift + amplitude noise.

    shift_std   : std of wavenumber shift in cm⁻¹
    width_jitter: fractional change in peak widths (Gaussian blur / sharpen)
    amp_jitter  : std of multiplicative amplitude noise
    """
    spec = spectrum.copy().astype(np.float64)

    # Rigid wavenumber shift (roll in index space)
    if shift_std > 0 and len(wv) > 1:
        dw = float(wv[1] - wv[0])
        shift_px = int(round(rng.normal(0, shift_std) / max(abs(dw), 1e-6)))
        spec = np.roll(spec, shift_px)

    # Width jitter via Gaussian blur/sharpen
    if width_jitter > 0:
        sigma_px = abs(rng.normal(0, width_jitter))
        if sigma_px > 0.1:
            spec = ndimage.gaussian_filter1d(spec, sigma=sigma_px)

    # Amplitude jitter
    if amp_jitter > 0:
        spec *= max(0.1, 1.0 + rng.normal(0, amp_jitter))

    # Do NOT re-normalize to [0,1] here — that would erase the shared-scale
    # relative intensities set in load_external_library and force every phase
    # back to peak=1. Only clip to the valid range (jitter, if enabled, can push
    # slightly out of [0,1]; with jitter disabled this is a no-op).
    if clip:
        spec = spec.clip(0.0, 1.0)
    return spec.astype(np.float32)


def random_poly_background(
        wv,
        max_amplitude=0.3,
        rng=None,
        coeffs=None,
):
    """
    Generate a random 3rd-order polynomial background.

    Parameters
    ----------
    wv : ndarray
        Wavenumber axis.
    max_amplitude : float
        Maximum peak-to-peak amplitude.
    rng : np.random.Generator or None
    coeffs : ndarray
        polynomial coefficients 

    Returns
    -------
    bg : ndarray
        Background signal.
    coeffs : ndarray
        Polynomial coefficients [a,b,c,d,...].
    """

    if (rng is None) and (coeffs is None):
        rng = np.random.default_rng()

    # Normalize x to [-1,1] for stable coefficient generation
    x = (wv - wv.mean()) / (wv.max() - wv.min()) * 2
    
    if coeffs is None:
        # Random cubic coefficients
        # coeffs = rng.normal(size=np.random.randint(0,10))
        coeffs = rng.normal(size=10)

    bg = np.polyval(coeffs, x)

    bg = (bg-bg.min())/(bg.max()-bg.min())

    scale = rng.uniform(0, max_amplitude)

    bg *= scale

    return bg, coeffs


def generate_phase_spectra(
    n_phases: int,
    wv: np.ndarray,
    params: dict,
    external_library: Optional[List[np.ndarray]],
    rng: np.random.Generator,
    phase_intensities: Optional[np.ndarray] = None,
    clip: bool = True,
) -> np.ndarray:
    """
    Generate one base spectrum per phase.

    Returns:
        spectra : (n_phases, W) float32, each row normalised to [0, 1]
    """
    W = len(wv)
    spectra = np.zeros((n_phases, W), dtype=np.float32)

    for k in range(n_phases):
        if params["spectra_mode"] == "external" and external_library:
            idx = rng.integers(0, len(external_library))
            base = external_library[idx].copy()
        else:
            n_peaks = int(rng.integers(params["n_peaks_range"][0],
                                       params["n_peaks_range"][1] + 1))
            base = synthetic_gaussian_spectrum(
                wv, n_peaks, params["peak_width_range"], rng,
                peak_amp_range=params["peak_amp_range"],
            )
            background, _ = random_poly_background(wv,max_amplitude=0.3,rng=rng)
            base += background

        # Perturb each phase slightly so they are not identical
        pert = _perturb_spectrum(
            base, wv,
            shift_std  = params.get("spectra_shift_std", 5.0),
            width_jitter = params.get("spectra_width_jitter", 0.5),
            amp_jitter   = params.get("spectra_amp_jitter", 0.1),
            rng          = rng,
            clip         = clip,
        )
        # RELATIVE INTENSITY: scale this phase's whole spectrum by its factor
        scale = 1.0 if phase_intensities is None else float(phase_intensities[k])
        spectra[k] = (pert * scale).astype(np.float32)

    return spectra


# =============================================================================
# Boundary mixing
# =============================================================================

def compute_mixing_weights(
    phase_map: np.ndarray,
    n_phases: int,
    mixing_type: str,
    boundary_width: float,
) -> np.ndarray:
    """
    Compute per-pixel spectral mixing weights at phase boundaries.

    For pixels far from boundaries the weight is 1.0 for their own phase.
    Within `boundary_width` pixels of a phase transition the weight smoothly
    shifts toward the nearest neighbouring phase.

    Args:
        mixing_type : 'sharp' | 'linear' | 'gaussian' | 'sigmoid'
        boundary_width : transition half-width in pixels

    Returns:
        weights : (n_phases, X, Y) float32, sums to 1 along axis 0
    """
    X, Y = phase_map.shape

    # One-hot initialisation
    weights = np.zeros((n_phases, X, Y), dtype=np.float32)
    for k in range(n_phases):
        weights[k] = (phase_map == k).astype(np.float32)

    if mixing_type == "sharp" or boundary_width < 0.5:
        return weights

    # Distance from every pixel to nearest pixel of every phase
    dists = np.full((n_phases, X, Y), fill_value=1e8, dtype=np.float32)
    for k in range(n_phases):
        mask = phase_map == k
        if mask.any():
            dists[k] = ndimage.distance_transform_edt(~mask).astype(np.float32)

    # Distance to nearest OTHER phase (exclude own)
    INF = 1e8
    dists_other = dists.copy()
    for k in range(n_phases):
        dists_other[k][phase_map == k] = INF

    dist_to_boundary = dists_other.min(axis=0)           # (X, Y)
    nearest_other    = dists_other.argmin(axis=0)         # (X, Y)

    in_bz   = dist_to_boundary < boundary_width
    d_norm  = np.clip(dist_to_boundary / max(boundary_width, 1e-6), 0.0, 1.0)

    # g = fraction of spectrum given to nearest OTHER phase
    # g(0) = 0.5 (at boundary), g(1) = 0 (at boundary_width distance)
    if mixing_type == "linear":
        g = np.where(in_bz, 0.5 * (1.0 - d_norm), 0.0)
    elif mixing_type == "gaussian":
        # bell centred at boundary (d=0), falls to ~0 at boundary_width (3-sigma)
        g = np.where(in_bz, 0.5 * np.exp(-4.5 * d_norm ** 2), 0.0)
    elif mixing_type == "sigmoid":
        # 0.25 * (1 - tanh(4*(d-0.5))): g(0)≈0.49, g(0.5)=0.25, g(1)≈0.009
        # The original 0.5*(1-tanh(3*(d-0.5))) gave g(0)≈0.95 — far too much mixing
        # at the boundary, causing both sides to swap spectra and produce an
        # intensity spike rather than a smooth transition.
        g = np.where(in_bz, 0.25 * (1.0 - np.tanh(4.0 * (d_norm - 0.5))), 0.0)
    else:
        g = np.zeros_like(dist_to_boundary)

    # Subtract g from own phase, add g to nearest other
    for k in range(n_phases):
        is_k          = phase_map == k
        receives_from = (nearest_other == k) & ~is_k
        weights[k]    = np.where(is_k,          weights[k] - g, weights[k])
        weights[k]    = np.where(receives_from, weights[k] + g, weights[k])

    weights = np.clip(weights, 0.0, 1.0)
    w_sum   = weights.sum(axis=0, keepdims=True)
    weights /= np.where(w_sum > 0, w_sum, 1.0)

    return weights


# =============================================================================
# Intra-phase spectral variation
# =============================================================================

def add_intra_phase_variation(
    cube: np.ndarray,
    phase_map: np.ndarray,
    spectra: np.ndarray,
    variation_amplitude: float,
    rng: np.random.Generator,
    clip: bool = True,
) -> np.ndarray:
    """
    Add smooth within-phase spectral variation using a Gaussian-blurred random
    spatial field.  Each phase independently receives a smooth amplitude
    modulation of its base spectrum.

    variation_amplitude : max fractional perturbation (e.g. 0.05 = ±5%)
    """
    if variation_amplitude <= 0:
        return cube

    X, Y, W = cube.shape
    sigma_blur = max(1.0, min(X, Y) / 6.0)

    for k in range(len(spectra)):
        mask = phase_map == k
        if not mask.any():
            continue

        # Smooth random field in [−1, +1]
        field = rng.standard_normal((X, Y)).astype(np.float32)
        field = ndimage.gaussian_filter(field, sigma=sigma_blur)
        lo, hi = field.min(), field.max()
        if hi > lo:
            field = 2.0 * (field - lo) / (hi - lo) - 1.0  # [-1, 1]

        # Spatially modulate the base spectrum amplitude
        # perturbation at (x,y) = field[x,y] * amplitude * spectra[k]
        perturb = (
            variation_amplitude * field[:, :, np.newaxis]  # (X, Y, 1)
            * spectra[k][np.newaxis, np.newaxis, :]         # (1, 1, W)
        )
        cube = np.where(
            mask[:, :, np.newaxis],
            cube + perturb.astype(np.float32),
            cube,
        )

    return np.clip(cube, 0.0, 1.0) if clip else cube


# =============================================================================
# Depth (subsurface) spectral mixing
# =============================================================================

# Physical thickness-based mixing: Not used in published deployment.
# Published training uses the legacy edge-distance branch of this function.
def apply_depth_mixing(cube, phase_map, spectra, background_phase=0,
                       edge_mix=0.4, decay_length=6.0, beneath_map=None, rng=None,
                       probing_depth_nm=None, thickness_range=(200.0, 500.0),
                       thickness_profile="sqrt", thickness_map_in=None, clip=True):
    """Emulate the finite probing depth of AFM-IR: a spectrum measured on a
    particle carries a subsurface contribution from whatever lies beneath it, and
    a thicker top particle SCREENS that subsurface signal more.

    Two models are available.

    PHYSICAL (set probing_depth_nm): Beer-Lambert screening, following Dazzi et al.,
    Anal. Chem. 2024, 96, 17931 (their eq 5). The signal fraction coming from
    BENEATH the top particle is

        alpha(x,y) = exp( -t(x,y) / probing_depth_nm )

    where t(x,y) is the local VERTICAL THICKNESS of the top particle (nm) and
    probing_depth_nm is the depth at which the signal falls to 1/e. Thin edges
    (t -> 0) are transparent, so alpha -> 1 and the subsurface shows through; thick
    centres (t >> p) screen it, alpha -> 0. This is the physically correct driver:
    what matters is thickness of overlying material, not lateral edge distance.
    The relevant knob is the RATIO t/p: when the top particle is much thicker than
    the probing depth it is opaque (little bleed); when it is thinner than or
    comparable to the probing depth the subsurface bleeds through strongly (this is
    the low-repetition-rate / thin-domain regime the paper describes).

    Per-particle thickness: each connected particle (one connected component of a
    non-background phase) is given a peak thickness T_max sampled uniformly from
    thickness_range, and a puddle-like profile that rises from 0 at the rim to
    T_max at the centre: t = T_max * f(d_in / d_in_max), with f = sqrt (default,
    shallow-cap / puddle) or linear. d_in is the inward distance transform of that
    particle. Because thickness is per-particle, overlapping PAIRS screen each
    other correctly: at an overlap the TOP particle's thickness attenuates the
    particle beneath it (from beneath_map), so a thin overlap edge shows the lower
    particle and a thick overlap hides it.

    LEGACY (probing_depth_nm is None): the original ad-hoc form
    alpha = edge_mix * exp(-d_in / decay_length), kept for backward compatibility.

    In both models the subsurface spectrum is spectra[beneath_map] per pixel (the
    background where a particle sits on background, or the underlying particle at an
    overlap), and NO mixing is applied where the thing beneath is the SAME phase
    (identical spectra -> no change).

    Args:
        cube             : (X, Y, W) assembled cube
        phase_map        : (X, Y) int phase labels; background_phase is the matrix
        spectra          : (n_phases, W) base spectra
        background_phase : label treated as the background/matrix (default 0)
        edge_mix, decay_length : LEGACY model parameters (used only when
                           probing_depth_nm is None)
        beneath_map      : (X, Y) label physically under the top particle per pixel
        rng              : np.random.Generator (needed to sample per-particle T_max)
        probing_depth_nm : probing depth p (nm). Set this to use the PHYSICAL model.
        thickness_range  : (lo, hi) per-particle peak thickness T_max (nm)
        thickness_profile: "sqrt" (puddle cap) or "linear" (wedge)
        thickness_map_in : optional (X, Y) explicit per-pixel thickness (nm); if
                           given, it is used directly instead of being sampled

    Returns:
        (mixed_cube, alpha_map, thickness_map)
          alpha_map     : (X, Y) per-pixel subsurface fraction actually applied
          thickness_map : (X, Y) per-pixel top-particle thickness in nm (0 in the
                          background). Returned so you can visualize the depth map.
    """
    X, Y, W = cube.shape
    zero = np.zeros((X, Y), dtype=np.float32)

    # subsurface target spectrum, per pixel: whatever is beneath the top particle
    if beneath_map is None:
        beneath = np.full(phase_map.shape, background_phase, dtype=np.int32)
    else:
        beneath = beneath_map.astype(np.int32)
    sub_spec = spectra[beneath]                              # (X, Y, W) fancy-index

    labels = [k for k in np.unique(phase_map) if k != background_phase]
    alpha_map     = np.zeros((X, Y), dtype=np.float32)
    thickness_map = np.zeros((X, Y), dtype=np.float32)       # nm

    use_physical = probing_depth_nm is not None and probing_depth_nm > 0.0

    if not use_physical:
        # ---- LEGACY: exponential in pixel edge-distance ----
        if edge_mix <= 0.0 or decay_length <= 0.0:
            return cube, zero, zero
        for k in labels:
            mask = phase_map == k
            if not mask.any():
                continue
            d_in = ndimage.distance_transform_edt(mask).astype(np.float32)
            a = edge_mix * np.exp(-d_in / float(decay_length))
            same = mask & (beneath == k)
            a = np.where(mask & ~same, a, 0.0).astype(np.float32)
            alpha_map = np.where(mask, a, alpha_map)
        a3 = alpha_map[:, :, np.newaxis]
        mixed = (1.0 - a3) * cube + a3 * sub_spec
        return ((np.clip(mixed, 0.0, 1.0) if clip else mixed).astype(np.float32),
                alpha_map, thickness_map)

    # ---- PHYSICAL: layered Beer-Lambert (top + lower particle + substrate) ----
    # At a pixel the measured signal is a depth-weighted sum of the layers, each
    # attenuated by the material above it (paper eq 5). For a stack top / lower /
    # substrate with top thickness t_t and lower thickness t_l:
    #   w_top   = 1 - exp(-t_t / p)                     (top particle)
    #   w_lower = exp(-t_t / p) - exp(-(t_t + t_l) / p) (particle beneath, if any)
    #   w_sub   = exp(-(t_t + t_l) / p)                 (substrate below everything)
    # These sum to 1. With no lower particle (t_l = 0) this reduces exactly to the
    # two-layer top+substrate mix; with an opaque lower particle (t_l >> p) the
    # substrate term vanishes and it reduces to the top+lower mix. So overlapping
    # PAIRS now also let the substrate bleed through when both are thin.
    rng = rng or np.random.default_rng()

    # Per-phase thickness over each particle's FULL extent. A phase-k particle
    # covers where it is the TOP (phase_map==k) OR where it lies immediately
    # BENEATH a later particle (beneath==k, an overlap it was painted under). For
    # pairs this recovers the whole footprint, so the lower particle's thickness
    # is available at the overlap.
    thick_by_phase = {}
    for k in labels:
        Mk = (phase_map == k) | (beneath == k)
        if not Mk.any():
            continue
        tk = np.zeros((X, Y), dtype=np.float32)
        comp, ncomp = ndimage.label(Mk)
        for c in range(1, ncomp + 1):
            cm = comp == c
            d = ndimage.distance_transform_edt(cm).astype(np.float32)
            dmax = float(d.max())
            if dmax <= 0.0:
                continue
            r = d / dmax                                    # 0 at rim -> 1 at centre
            prof = np.sqrt(r) if thickness_profile == "sqrt" else r
            T_max = float(rng.uniform(*thickness_range))
            tk[cm] = (T_max * prof)[cm]
        thick_by_phase[k] = tk

    # per-pixel top thickness (visible particle) and lower thickness (particle
    # immediately beneath; 0 where the layer beneath is the substrate)
    t_top   = np.zeros((X, Y), dtype=np.float32)
    t_lower = np.zeros((X, Y), dtype=np.float32)
    for k in labels:
        if k not in thick_by_phase:
            continue
        m = phase_map == k
        t_top[m] = thick_by_phase[k][m]
        bm = beneath == k
        t_lower[bm] = thick_by_phase[k][bm]

    if thickness_map_in is not None:                        # optional explicit t_top override
        t_top = np.asarray(thickness_map_in, dtype=np.float32)

    p = float(probing_depth_nm)
    e_top  = np.exp(-t_top / p).astype(np.float32)
    e_both = np.exp(-(t_top + t_lower) / p).astype(np.float32)
    w_top   = (1.0 - e_top)
    w_lower = (e_top - e_both)
    w_sub   = e_both

    part       = (phase_map != background_phase)
    lower_spec = spectra[beneath]                           # (X,Y,W); = substrate where beneath is bg
    bg_spec    = spectra[background_phase]                  # (W,)

    new_cube = (w_top[:, :, None] * cube
                + w_lower[:, :, None] * lower_spec
                + w_sub[:, :, None] * bg_spec[None, None, :]).astype(np.float32)
    mixed = np.where(part[:, :, None], new_cube, cube).astype(np.float32)

    # report: alpha = total subsurface fraction (1 - top); thickness = t_top
    alpha_map     = np.where(part, 1.0 - w_top, 0.0).astype(np.float32)
    thickness_map = t_top
    return ((np.clip(mixed, 0.0, 1.0) if clip else mixed).astype(np.float32),
            alpha_map, thickness_map)


# =============================================================================
# Cube assembly
# =============================================================================

def assemble_cube(
    phase_map: np.ndarray,
    spectra: np.ndarray,
    mixing_weights: np.ndarray,
    noise_std: float,
    intra_phase_variation: float,
    rng: np.random.Generator,
    clip: bool = True,
) -> np.ndarray:
    """
    Assemble the hyperspectral cube.

    cube[x, y, w] = Σ_k  weights[k,x,y] * spectra[k,w]  +  noise

    Args:
        mixing_weights        : (n_phases, X, Y)
        spectra               : (n_phases, W)
        noise_std             : Gaussian noise standard deviation
        intra_phase_variation : fractional within-phase spectral amplitude variation

    Returns:
        cube : (X, Y, W) float32 in [0, 1]
    """
    # einsum: weights (k,x,y) × spectra (k,w) → cube (x,y,w)
    cube = np.einsum("kij,kw->ijw", mixing_weights, spectra).astype(np.float32)

    # Within-phase variation
    cube = add_intra_phase_variation(cube, phase_map, spectra, intra_phase_variation, rng, clip=clip)

    # Additive Gaussian noise
    if noise_std > 0:
        cube += rng.normal(0.0, noise_std, size=cube.shape).astype(np.float32)

    return np.clip(cube, 0.0, 1.0) if clip else cube


# =============================================================================
# Per-cube generator
# =============================================================================

def generate_one_cube(
    params: dict,
    wv: np.ndarray,
    external_library: Optional[List[np.ndarray]],
) -> dict:
    """
    Generate one complete synthetic hyperspectral cube.

    Returns dict with keys: cube, phase_map, spectra, wavenumbers, params
    """
    rng = np.random.default_rng(params["seed"])
    X, Y   = params["X"], params["Y"]
    n_ph   = params["n_phases"]

    # ---- per-phase RELATIVE INTENSITY + clipping policy ----
    clip_output = bool(params.get("clip_output", True))
    if params.get("phase_intensities") is not None:
        phase_intensities = np.asarray(params["phase_intensities"], dtype=np.float32)
        if phase_intensities.shape[0] != n_ph:
            raise ValueError(f"phase_intensities has {phase_intensities.shape[0]} "
                             f"entries but n_phases={n_ph}")
    elif params.get("relative_intensity_range") is not None:
        lo, hi = params["relative_intensity_range"]
        phase_intensities = rng.uniform(float(lo), float(hi), size=n_ph).astype(np.float32)
    else:
        phase_intensities = np.ones(n_ph, dtype=np.float32)

    # 1. Phase map (+ beneath map: what lies under each top particle at overlaps)
    phase_map, beneath_map = generate_phase_map(
        X=X, Y=Y,
        n_phases       = n_ph,
        feature_sizes  = params["feature_sizes"],
        curvatures     = params["curvatures"],
        phase_fractions= params["phase_fractions"],
        morphology     = params["morphology"],
        rng            = rng,
    )

    # 2. Spectra
    spectra = generate_phase_spectra(n_ph, wv, params, external_library, rng,
                                     phase_intensities=phase_intensities,
                                     clip=clip_output)

    # 3. Mixing weights
    mixing_weights = compute_mixing_weights(
        phase_map      = phase_map,
        n_phases       = n_ph,
        mixing_type    = params["mixing_type"],
        boundary_width = params["boundary_width"],
    )

    # 4. Assemble
    cube = assemble_cube(
        phase_map             = phase_map,
        spectra               = spectra,
        mixing_weights        = mixing_weights,
        noise_std             = params["noise_std"],
        intra_phase_variation = params["intra_phase_variation"],
        rng                   = rng,
        clip                  = clip_output,
    )

    # 5. Depth (subsurface) mixing -- emulate AFM-IR finite probing depth.
    #    Off by default (edge_mix=0); set params to enable.
    alpha_map = None
    thickness_map = None
    probing_depth_nm = params.get("depth_probing_depth_nm", None)   # set -> PHYSICAL model
    legacy_on = (params.get("depth_edge_mix", 0.0) > 0.0
                 and params.get("depth_decay_length", 0.0) > 0.0)
    if (probing_depth_nm is not None and probing_depth_nm > 0.0) or legacy_on:
        cube, alpha_map, thickness_map = apply_depth_mixing(
            cube, phase_map, spectra,
            background_phase = params.get("depth_background_phase", 0),
            edge_mix         = params.get("depth_edge_mix", 0.4),
            decay_length     = params.get("depth_decay_length", 6.0),
            beneath_map      = beneath_map,
            rng              = rng,
            probing_depth_nm = probing_depth_nm,
            thickness_range  = tuple(params.get("depth_thickness_range", (200.0, 500.0))),
            thickness_profile= params.get("depth_thickness_profile", "sqrt"),
            clip             = clip_output,
        )

    return dict(cube=cube, phase_map=phase_map, beneath_map=beneath_map,
                spectra=spectra, phase_intensities=phase_intensities,
                wavenumbers=wv, params=params,
                depth_alpha=alpha_map, depth_thickness=thickness_map)


# =============================================================================
# Parameter sampling  (uniform across user-specified ranges)
# =============================================================================

def sample_parameter_sets(args, n_cubes: int, rng: np.random.Generator) -> List[dict]:
    """
    Draw n_cubes parameter dictionaries, each value sampled uniformly from
    the CLI-specified range.  Discrete parameters (n_phases) are drawn as
    integers; continuous parameters are drawn as floats.
    """
    mixing_choices    = args.mixing_types
    morphology_choices = args.morphology_types
    param_sets = []

    for i in range(n_cubes):
        n_phases = int(rng.integers(args.n_phases[0], args.n_phases[1] + 1))
        n_feat   = n_phases - 1  # number of non-background phases

        # Base values drawn from the overall range; per-phase variation added
        base_feat = rng.uniform(*args.feature_size)
        base_curv = rng.uniform(*args.boundary_curvature)

        feature_sizes = [
            float(np.clip(base_feat * rng.uniform(0.70, 1.30),
                          args.feature_size[0], args.feature_size[1]))
            for _ in range(n_feat)
        ]
        curvatures = [
            float(np.clip(base_curv * rng.uniform(0.85, 1.15),
                          args.boundary_curvature[0], args.boundary_curvature[1]))
            for _ in range(n_feat)
        ]

        # Phase fractions: drawn uniformly, forced to sum ≤ 0.85 (leaves ≥15% background)
        phase_fractions = []
        remaining = 0.85
        for j in range(n_feat):
            hi     = min(args.phase_fraction[1], remaining / max(1, n_feat - j))
            lo     = min(args.phase_fraction[0], hi)
            frac   = float(rng.uniform(lo, hi))
            phase_fractions.append(frac)
            remaining -= frac

        params = dict(
            cube_id                = i,
            X                      = args.image_size[0],
            Y                      = args.image_size[1],
            n_phases               = n_phases,
            feature_sizes          = feature_sizes,
            curvatures             = curvatures,
            phase_fractions        = phase_fractions,
            morphology             = str(rng.choice(morphology_choices)),
            boundary_width         = float(rng.uniform(*args.boundary_width)),
            mixing_type            = str(rng.choice(mixing_choices)),
            noise_std              = float(rng.uniform(*args.noise_std)),
            intra_phase_variation  = float(rng.uniform(*args.intra_phase_variation)),
            spectra_mode           = args.spectra_mode,
            n_peaks_range          = list(args.n_peaks),
            peak_width_range       = list(args.peak_width),
            peak_amp_range         = list(args.peak_amp_range),
            spectra_shift_std      = args.spectra_shift_std,
            spectra_width_jitter   = args.spectra_width_jitter,
            spectra_amp_jitter     = args.spectra_amp_jitter,
            seed                   = int(rng.integers(0, 2 ** 31)),
        )
        param_sets.append(params)

    return param_sets


# =============================================================================
# I/O helpers
# =============================================================================

def _params_to_str(params: dict) -> str:
    lines = []
    for k, v in params.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def save_result(
    output_dir: Path,
    result: dict,
    save_png: bool,
) -> None:
    """Save cube, phase map, spectra, wavenumbers, and params for one cube."""
    params    = result["params"]
    cube_dir  = output_dir / f"cube_{params['cube_id']:04d}"
    cube_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(cube_dir / "hyperspectral_cube.npy"), result["cube"])
    np.save(str(cube_dir / "phase_map.npy"),          result["phase_map"])
    np.save(str(cube_dir / "spectra.npy"),            result["spectra"])
    np.save(str(cube_dir / "wavenumbers.npy"),        result["wavenumbers"])

    with open(cube_dir / "params.txt", "w") as f:
        f.write(_params_to_str(params))

    if save_png and HAS_MPL:
        _save_phase_png(cube_dir, result["phase_map"], result["spectra"],
                        result["wavenumbers"], params)


def _save_phase_png(cube_dir, phase_map, spectra, wv, params):
    n_ph   = params["n_phases"]
    cmap   = plt.cm.get_cmap("tab10", n_ph)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Phase map
    ax = axes[0]
    im = ax.imshow(phase_map, origin="lower", cmap=cmap,
                   vmin=-0.5, vmax=n_ph - 0.5, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_ph))
    cbar.ax.set_yticklabels([f"Phase {k}" for k in range(n_ph)], fontsize=7)
    ax.set_title(f"Phase map  ({params['morphology']})", fontsize=9)
    ax.set_xlabel("x  [px]"); ax.set_ylabel("y  [px]")

    # Base spectra per phase
    ax = axes[1]
    for k in range(n_ph):
        lbl = "background" if k == 0 else f"Phase {k}"
        ax.plot(wv, spectra[k], color=cmap(k), lw=1.4, label=lbl)
    ax.set_xlabel("Wavenumber  [cm⁻¹]" if wv[0] > 10 else "Wavenumber  [norm.]")
    ax.set_ylabel("Intensity  [norm.]")
    ax.set_title("Base spectra per phase", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(cube_dir / "overview.png"), dpi=120, bbox_inches="tight")
    plt.close()


def load_external_library(spectra_dir: str, wv: np.ndarray,
                           sigma_smooth: float) -> List[np.ndarray]:
    """
    Load all 2-column spectrum files (*.csv, *.txt, *.dat) from spectra_dir.

    Returns a list of (W,) float32 arrays normalised to [0, 1].
    """
    spectra_dir = Path(spectra_dir)
    patterns    = ["*.csv", "*.txt", "*.dat", "*.CSV", "*.TXT"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(spectra_dir.glob(pat))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            f"No spectrum files (*.csv / *.txt / *.dat) found in {spectra_dir}"
        )

    library = []
    for fp in files:
        try:
            spec = load_external_spectrum(str(fp), wv, sigma_smooth)
            library.append(spec)
        except Exception as e:
            warnings.warn(f"Skipping {fp.name}: {e}")

    if not library:
        raise RuntimeError(f"Could not load any spectra from {spectra_dir}")

    # SHARED-scale normalization: divide ALL spectra by a single global value so
    # that (a) inter-phase relative intensities are preserved (the tallest peak
    # in the whole library maps to 1, every other phase keeps its true relative
    # height) and (b) everything stays in [0,1], which the model's scaled_sigmoid
    # output and the downstream cube clip both require. Shift by the global min
    # first so a non-zero baseline doesn't bias the scale.
    g_min = min(float(s.min()) for s in library)
    g_max = max(float(s.max()) for s in library)
    denom = (g_max - g_min) if (g_max > g_min) else 1.0
    library = [((s.astype(np.float64) - g_min) / denom).astype(np.float32)
               for s in library]
    print(f"  Shared-scale norm: global min={g_min:.4g}, max={g_max:.4g} "
          f"-> all spectra in [0,1], relative heights preserved")

    print(f"  Loaded {len(library)} external spectra from {spectra_dir}")
    return library


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic AFM-IR hyperspectral cubes for ANP training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    p.add_argument("--n_cubes", type=int, default=50,
                   help="Number of hyperspectral cubes to generate")
    p.add_argument("--output_dir", type=str, default="./synth_afmir_data",
                   help="Root directory for output.  A timestamped subdirectory is "
                        "created inside.")
    p.add_argument("--seed", type=int, default=42,
                   help="Master random seed for reproducibility")
    p.add_argument("--save_png", action="store_true",
                   help="Also save a phase-map + spectra overview PNG per cube "
                        "(requires matplotlib)")

    # ── Spatial grid ───────────────────────────────────────────────────────────
    p.add_argument("--image_size", type=int, nargs=2, default=[32, 32],
                   metavar=("X", "Y"),
                   help="Spatial dimensions of each cube in pixels")

    # ── Spectral grid ──────────────────────────────────────────────────────────
    p.add_argument("--wv_min", type=float, default=800.0,
                   help="Minimum wavenumber [cm⁻¹]")
    p.add_argument("--wv_max", type=float, default=3500.0,
                   help="Maximum wavenumber [cm⁻¹]")
    p.add_argument("--n_wavenumbers", type=int, default=104,
                   help="Number of wavenumber steps (spectral channels W)")

    # ── Phases ─────────────────────────────────────────────────────────────────
    p.add_argument("--n_phases", type=int, nargs=2, default=[2, 3],
                   metavar=("MIN", "MAX"),
                   help="Range for number of phases including background (phase 0)")
    p.add_argument("--feature_size", type=float, nargs=2, default=[4.0, 40.0],
                   metavar=("MIN_PX", "MAX_PX"),
                   help="Range for mean feature radius [pixels] of non-background phases")
    p.add_argument("--boundary_curvature", type=float, nargs=2, default=[0.40, 1.00],
                   metavar=("MIN", "MAX"),
                   help="Range for ellipse aspect ratio b/a ∈ (0,1].  "
                        "1.0 = circular, lower = more elongated")
    p.add_argument("--phase_fraction", type=float, nargs=2, default=[0.10, 0.40],
                   metavar=("MIN", "MAX"),
                   help="Range for target area fraction of each non-background phase")
    p.add_argument("--morphology_types", type=str, nargs="+",
                   default=["dispersed", "voronoi"],
                   choices=["dispersed", "voronoi"],
                   help="Morphology types to include (one chosen randomly per cube)")

    # ── Boundary mixing ────────────────────────────────────────────────────────
    p.add_argument("--boundary_width", type=float, nargs=2, default=[1.0, 6.0],
                   metavar=("MIN_PX", "MAX_PX"),
                   help="Range for boundary mixing half-width [pixels]")
    p.add_argument("--mixing_types", type=str, nargs="+",
                   default=["sharp", "linear", "gaussian", "sigmoid"],
                   choices=["sharp", "linear", "gaussian", "sigmoid"],
                   help="Spectral mixing functions at phase boundaries "
                        "(one chosen randomly per cube)")

    # ── Noise / variation ──────────────────────────────────────────────────────
    p.add_argument("--noise_std", type=float, nargs=2, default=[0.002, 0.020],
                   metavar=("MIN", "MAX"),
                   help="Range for Gaussian noise standard deviation (in [0,1] units)")
    p.add_argument("--intra_phase_variation", type=float, nargs=2, default=[0.0, 0.05],
                   metavar=("MIN", "MAX"),
                   help="Range for within-phase spectral amplitude variation "
                        "(fractional, e.g. 0.05 = ±5%% of base spectrum)")

    # ── Spectra mode ───────────────────────────────────────────────────────────
    p.add_argument("--spectra_mode", type=str, default="synthetic",
                   choices=["synthetic", "external"],
                   help="'synthetic': generate Gaussian-peak spectra; "
                        "'external': load from files in --spectra_dir")
    p.add_argument("--spectra_dir", type=str, default=None,
                   help="Directory containing 2-column spectrum files "
                        "(required when --spectra_mode external)")
    p.add_argument("--spectra_smooth", type=float, default=0.0,
                   help="Gaussian smoothing width [cm⁻¹] applied when loading "
                        "external spectra, to account for resolution differences")
    p.add_argument("--n_peaks", type=int, nargs=2, default=[2, 8],
                   metavar=("MIN", "MAX"),
                   help="Range for number of Gaussian peaks per spectrum "
                        "(synthetic mode only)")
    p.add_argument("--peak_width", type=float, nargs=2, default=[20.0, 100.0],
                   metavar=("MIN_CM1", "MAX_CM1"),
                   help="Range for peak FWHM [cm⁻¹] (synthetic mode only)")
    p.add_argument("--peak_amp_range", type=float, nargs=2, default=[0.05, 1.0],
                   metavar=("MIN", "MAX"),
                   help="Range for individual peak amplitude drawn uniformly before "
                        "spectrum normalisation (synthetic mode only).  MIN controls "
                        "how short minor peaks can be relative to the dominant peak. "
                        "E.g. 0.05 1.0 allows peaks as small as 5%% of the tallest; "
                        "0.8 1.0 forces all peaks to similar height.")
    p.add_argument("--spectra_shift_std", type=float, default=5.0,
                   help="Std of wavenumber shift applied per-phase spectrum [cm⁻¹]")
    p.add_argument("--spectra_width_jitter", type=float, default=0.5,
                   help="Std of Gaussian blur applied to width-jitter each spectrum [px]")
    p.add_argument("--spectra_amp_jitter", type=float, default=0.10,
                   help="Std of multiplicative amplitude noise per-phase spectrum")

    args = p.parse_args()

    # Validation
    if args.spectra_mode == "external" and args.spectra_dir is None:
        p.error("--spectra_dir is required when --spectra_mode external")
    if args.n_phases[0] < 2:
        p.error("--n_phases MIN must be ≥ 2 (at least background + 1 dispersed phase)")
    if args.image_size[0] < 4 or args.image_size[1] < 4:
        p.error("--image_size must be at least 4×4")

    return args


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    master_rng = np.random.default_rng(args.seed)

    # ── Wavenumber grid (fixed for all cubes in this dataset) ─────────────────
    wv = np.linspace(args.wv_min, args.wv_max, args.n_wavenumbers,
                     dtype=np.float32)

    # ── External spectral library (loaded once, reused) ────────────────────────
    external_library: Optional[List[np.ndarray]] = None
    if args.spectra_mode == "external":
        external_library = load_external_library(
            args.spectra_dir, wv, args.spectra_smooth
        )

    # ── Output directory ───────────────────────────────────────────────────────
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir    = Path(args.output_dir) / f"synth_afmir_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {run_dir}")

    # ── Sample parameter sets ─────────────────────────────────────────────────
    param_sets = sample_parameter_sets(args, args.n_cubes, master_rng)

    # ── Save dataset-level parameters ─────────────────────────────────────────
    dataset_meta = dict(
        generated_at    = ts,
        n_cubes         = args.n_cubes,
        image_size_XY   = args.image_size,
        wv_min          = args.wv_min,
        wv_max          = args.wv_max,
        n_wavenumbers   = args.n_wavenumbers,
        n_phases_range  = args.n_phases,
        feature_size_range  = args.feature_size,
        boundary_curvature_range = args.boundary_curvature,
        phase_fraction_range = args.phase_fraction,
        morphology_types = args.morphology_types,
        boundary_width_range = args.boundary_width,
        mixing_types    = args.mixing_types,
        noise_std_range = args.noise_std,
        intra_phase_variation_range = args.intra_phase_variation,
        spectra_mode    = args.spectra_mode,
        spectra_dir     = args.spectra_dir,
        spectra_smooth  = args.spectra_smooth,
        n_peaks_range   = args.n_peaks,
        peak_width_range = args.peak_width,
        spectra_shift_std = args.spectra_shift_std,
        spectra_width_jitter = args.spectra_width_jitter,
        spectra_amp_jitter = args.spectra_amp_jitter,
        master_seed     = args.seed,
    )
    with open(run_dir / "dataset_params.txt", "w") as f:
        f.write("# Dataset-level generation parameters\n")
        f.write(_params_to_str(dataset_meta))
    np.save(str(run_dir / "wavenumbers.npy"), wv)

    # ── Generate cubes ────────────────────────────────────────────────────────
    t0     = time.time()
    errors = []
    print(f"Generating {args.n_cubes} cubes "
          f"({args.image_size[0]}×{args.image_size[1]}×{args.n_wavenumbers})...\n")

    for i, params in enumerate(param_sets):
        try:
            result = generate_one_cube(params, wv, external_library)
            save_result(run_dir, result, save_png=args.save_png)
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (args.n_cubes - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1:4d}/{args.n_cubes}]  "
                  f"phases={params['n_phases']}  "
                  f"morph={params['morphology']:9s}  "
                  f"mix={params['mixing_type']:8s}  "
                  f"bw={params['boundary_width']:.1f}px  "
                  f"eta {eta:.0f}s",
                  end="\r")
        except Exception as e:
            errors.append((i, str(e)))
            warnings.warn(f"Cube {i} failed: {e}")

    elapsed = time.time() - t0
    print(f"\n\nDone.  {args.n_cubes - len(errors)}/{args.n_cubes} cubes generated "
          f"in {elapsed:.1f}s  ({elapsed/max(args.n_cubes,1):.2f}s/cube)")
    if errors:
        print(f"  {len(errors)} failed: {errors}")
    print(f"\nDataset saved to:\n  {run_dir}\n")

    # Optional fixed-library training; not the published live-generation run.
    print("For fixed-library training, use conv_cnp_afmir_prob.py")
    print(f"with --no-onthefly --data_dir {run_dir}")



if __name__ == "__main__":
    main()
