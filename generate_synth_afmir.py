#!/usr/bin/env python3
"""
generate_synth_afmir.py
=======================
Generate synthetic AFM-IR hyperspectral cubes for training the CACNP model.

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

import os                                            # Path/env utilities
import sys                                           # Interpreter utilities
import math                                          # pi, sqrt, log, radians
import time                                          # Timing / ETA in main()
import argparse                                      # CLI parsing
import warnings                                      # Non-fatal warnings
from datetime import datetime                        # Timestamped output dir
from pathlib import Path                             # Filesystem paths
from typing import Dict, List, Optional, Tuple       # Type hints

import numpy as np                                   # Array maths / RNG
from scipy import ndimage, interpolate, spatial      # Distance transforms, interp, Voronoi

try:
    import matplotlib                                # Optional: overview PNGs
    matplotlib.use("Agg")                            # Headless backend (no display needed)
    import matplotlib.pyplot as plt                  # Plotting API
    import matplotlib.colors as mcolors              # Colour helpers
    HAS_MPL = True                                   # Plotting is available
except ImportError:
    HAS_MPL = False                                  # Plotting unavailable
    warnings.warn("matplotlib not found — phase-map PNGs will not be saved.")   # Tell the user


# =============================================================================
# Lightweight geometry 
# =============================================================================

class Circle:                                        # Simple circular region test
    def __init__(self, center, r):                   # Centre point and radius
        self.center = np.asarray(center, dtype=float)   # Store centre as an array
        self.r = float(r)                            # Store radius

    def within(self, pts: np.ndarray) -> np.ndarray: # Which points lie inside?
        d = pts - self.center                        # Offsets from the centre
        return (d[:, 0] ** 2 + d[:, 1] ** 2) <= self.r ** 2   # Inside-circle test


class Ellipse:                                        # Rotated elliptical region test
    """Rotated ellipse.  angle is CCW from +x axis in degrees."""
    def __init__(self, center, a, b, angle=0.0):      # Centre, semi-axes, rotation
        self.center = np.asarray(center, dtype=float) # Store centre as an array
        self.a = float(a)                             # Semi-major axis
        self.b = float(b)                             # Semi-minor axis
        self.angle = float(angle)                     # Rotation in degrees

    def within(self, pts: np.ndarray) -> np.ndarray: # Which points lie inside?
        theta = math.radians(self.angle)             # Rotation in radians
        cos_t, sin_t = math.cos(theta), math.sin(theta)   # Rotation matrix entries
        dx = pts[:, 0] - self.center[0]              # x offsets from the centre
        dy = pts[:, 1] - self.center[1]              # y offsets from the centre
        xr =  cos_t * dx + sin_t * dy                # Rotate into the ellipse frame (x)
        yr = -sin_t * dx + cos_t * dy                # Rotate into the ellipse frame (y)
        return (xr / self.a) ** 2 + (yr / self.b) ** 2 <= 1.0   # Inside-ellipse test


# =============================================================================
# Phase map generation
# =============================================================================

def _voronoi_inscribed_radius(vor, idx: int, fallback: float = 20.0) -> float:   # Largest non-overlapping radius
    """Minimum distance from seed idx to any of its Voronoi ridge vertices."""
    seed = vor.points[idx]                           # This cell's seed point
    verts = []                                       # Collected ridge vertices
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):   # Every Voronoi ridge
        if idx in (p1, p2):                          # Ridge belongs to this seed
            if v1 >= 0:                              # -1 marks a vertex at infinity
                verts.append(vor.vertices[v1])       # Keep the finite vertex
            if v2 >= 0:                              # -1 marks a vertex at infinity
                verts.append(vor.vertices[v2])       # Keep the finite vertex
    if not verts:                                    # Unbounded cell with no finite vertices
        return fallback                              # Use the caller's default
    return float(min(np.linalg.norm(seed - v) for v in verts))   # Closest ridge vertex


def _dispersed_phase_map(
    X: int, Y: int,                                  # Spatial dimensions
    n_phases: int,                                   # Phase count (including background)
    feature_sizes: List[float],                      # Mean radius per non-bg phase
    curvatures: List[float],                         # Aspect ratio b/a per non-bg phase
    phase_fractions: List[float],                    # Target area fraction per non-bg phase
    rng: np.random.Generator,                        # RNG for placement
    all_pts: np.ndarray,                             # (X*Y, 2) pixel coordinates
) -> np.ndarray:
    """
    Background (phase 0) fills the canvas; each non-background phase is placed
    as a collection of Voronoi-spaced ellipses.  Later phases overwrite earlier
    ones, creating natural overlaps at phase boundaries.
    """
    label_flat = np.zeros(X * Y, dtype=np.int32)     # Top-visible phase per pixel
    # Stores depth information
    beneath_flat = np.zeros(X * Y, dtype=np.int32)   # What lies under the top label

    def _paint(hit, phase_idx):                      # Paint one ellipse onto the map
        # Record what we is being covered, only where we actually CHANGE the label to a
        # different phase (same-phase overlaps carry the same spectrum -> no mix)
        prev = label_flat[hit]                       # Labels currently under this ellipse
        changed = hit.copy()                         # Start from the ellipse footprint
        changed[hit] = (prev != phase_idx)           # Keep only genuinely-changed pixels
        beneath_flat[changed] = label_flat[changed]  # what was here before
        label_flat[hit] = phase_idx                  # This phase is now on top

    for phase_idx in range(1, n_phases):             # Every non-background phase
        feat = feature_sizes[phase_idx - 1]          # This phase's mean feature radius
        curv = curvatures[phase_idx - 1]             # This phase's aspect ratio
        frac = phase_fractions[phase_idx - 1]        # This phase's target area fraction

        # Estimate number of features from target area fraction
        area_per = math.pi * feat ** 2 * curv        # Mean area of one ellipse
        n_feat = max(1, round(frac * X * Y / max(area_per, 1.0)))   # Ellipses needed

        margin = max(2.0, feat * 0.25)               # Keep seeds away from the border
        seeds = rng.uniform(
            low=[margin, margin],
            high=[max(margin + 1, X - margin), max(margin + 1, Y - margin)],
            size=(n_feat, 2),
        )                                            # Random seed centres

        if n_feat >= 4:                              # Enough seeds for a Voronoi diagram
            mirrors = np.array([
                [-X, Y / 2], [2 * X, Y / 2],
                [X / 2, -Y], [X / 2, 2 * Y],
            ])                                       # Far mirror points to bound the cells
            vor = spatial.Voronoi(np.vstack([seeds, mirrors]))   # Voronoi over seeds+mirrors
            for i in range(n_feat):                  # Place one ellipse per seed
                r_ins = _voronoi_inscribed_radius(vor, i, fallback=feat)   # Non-overlap radius
                a = float(np.clip(feat * rng.uniform(0.70, 1.30), 1.5, r_ins * 0.90))   # Semi-major
                b = float(max(1.0, a * curv * rng.uniform(0.85, 1.15)))   # Semi-minor
                angle = float(rng.uniform(0, 180))   # Random orientation
                _paint(Ellipse(seeds[i], a, b, angle).within(all_pts), phase_idx)   # Paint it
        else:                                        # Too few seeds for Voronoi spacing
            for i in range(n_feat):                  # Place ellipses without spacing control
                a = float(max(1.5, feat * rng.uniform(0.70, 1.30)))   # Semi-major
                b = float(max(1.0, a * curv))        # Semi-minor
                angle = float(rng.uniform(0, 180))   # Random orientation
                _paint(Ellipse(seeds[i], a, b, angle).within(all_pts), phase_idx)   # Paint it

    return label_flat.reshape(X, Y), beneath_flat.reshape(X, Y)   # Top map + beneath map


def _voronoi_grain_map(
    X: int, Y: int,                                  # Spatial dimensions
    n_phases: int,                                   # Phase count
    feature_sizes: List[float],                      # Sets the grain density
    rng: np.random.Generator,                        # RNG for seeds/assignment
    all_pts: np.ndarray,                             # (X*Y, 2) pixel coordinates
) -> np.ndarray:
    """
    Voronoi tessellation where each grain cell is randomly assigned to a phase.
    Mean feature size controls grain density.
    """
    mean_feat = float(np.mean(feature_sizes)) if feature_sizes else max(X, Y) / 4   # Mean grain radius
    n_seeds = max(4, int(X * Y / max(1.0, math.pi * mean_feat ** 2)))   # Grains needed for that size

    seeds = rng.uniform([0, 0], [X, Y], size=(n_seeds, 2))   # Random grain centres
    seed_phases = rng.integers(0, n_phases, size=n_seeds)    # Random phase per grain

    tree = spatial.cKDTree(seeds)                    # Spatial index over the seeds
    _, nearest = tree.query(all_pts, workers=1)      # Nearest seed for every pixel
    nearest = np.clip(nearest, 0, n_seeds - 1)       # Guard against out-of-range indices
    return seed_phases[nearest].astype(np.int32).reshape(X, Y)   # Grain phase map


def generate_phase_map(
    X: int,                                          # Spatial height
    Y: int,                                          # Spatial width
    n_phases: int,                                   # Phase count (including background)
    feature_sizes: List[float],                      # Mean radius per non-bg phase
    curvatures: List[float],                         # Aspect ratio per non-bg phase
    phase_fractions: List[float],                    # Target area fraction per non-bg phase
    morphology: str,                                 # 'dispersed' or 'voronoi'
    rng: np.random.Generator,                        # RNG
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
    yy, xx = np.mgrid[0:X, 0:Y]                      # Row/column index grids
    all_pts = np.column_stack([xx.ravel(), yy.ravel()])   # (X*Y, 2) pixel coordinates

    if morphology == "voronoi":                      # Grain-tessellation morphology
        pm = _voronoi_grain_map(X, Y, n_phases, feature_sizes, rng, all_pts)   # Build grain map
        beneath = np.zeros_like(pm)                  # grains do not stack -> nothing beneath
        return pm, beneath                           # No subsurface structure to report
    else:                                            # Dispersed-particle morphology
        return _dispersed_phase_map(
            X, Y, n_phases, feature_sizes, curvatures, phase_fractions, rng, all_pts
        )                                            # Returns (phase_map, beneath_map)


# =============================================================================
# Spectrum generation
# =============================================================================

def synthetic_gaussian_spectrum(
    wv: np.ndarray,                                  # Wavenumber grid
    n_peaks: int,                                    # How many Gaussian peaks to sum
    peak_width_range: Tuple[float, float],           # FWHM range (cm^-1)
    rng: np.random.Generator,                        # RNG
    peak_amp_range: Tuple[float, float] = (0.2, 1.0),# Per-peak amplitude range
) -> np.ndarray:
    """
    Build a synthetic spectrum as a sum of Gaussian peaks.

    Peaks are placed uniformly at random within wv.  Width (FWHM) is drawn
    from peak_width_range in cm⁻¹.  Amplitude is drawn uniformly from
    peak_amp_range; lowering the minimum allows short minor peaks relative
    to the dominant peak.  Output is normalised to [0, 1].
    """
    wv_min, wv_max = float(wv.min()), float(wv.max())   # Spectral range
    spec = np.zeros(len(wv), dtype=np.float64)          # Accumulated spectrum
    for _ in range(n_peaks):                            # Add one Gaussian per peak
        center = rng.uniform(wv_min, wv_max)            # Random peak position
        fwhm = rng.uniform(*peak_width_range)           # Random full width at half max
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))   # FWHM -> Gaussian sigma
        amp = rng.uniform(*peak_amp_range)              # Random peak height
        spec += amp * np.exp(-0.5 * ((wv - center) / sigma) ** 2)   # Add the Gaussian
    if spec.max() > 0:                                  # Avoid dividing by zero
        spec /= spec.max()                              # Normalise the tallest peak to 1
    return spec.astype(np.float32)                      # Return the spectrum
   

def _try_load_two_column(filepath: str) -> Tuple[np.ndarray, np.ndarray]:   # Robust 2-column loader
    """
    Load a 2-column (wavenumber, intensity) file.  Handles:
      - CSV with optional single header row
      - Whitespace-separated TXT
      - Lines beginning with '#' ignored
    Raises ValueError if format cannot be parsed.
    """
    path = Path(filepath)                            # Path object (kept for clarity)
    for skip in (0, 1):                              # Try with and without a header row
        try:
            data = np.loadtxt(filepath, comments="#", skiprows=skip)   # Attempt the load
            if data.ndim == 2 and data.shape[1] >= 2:       # Needs at least two columns
                wv = data[:, 0].astype(np.float64)          # Column 0: wavenumbers
                intensity = data[:, 1].astype(np.float64)   # Column 1: intensities
                if len(wv) > 5 and not np.all(wv == wv[0]): # Sanity: a real varying axis
                    return wv, intensity                    # Success
        except Exception:                                   # Parse failed with this skip count
            continue                                        # Try the next one
    raise ValueError(f"Cannot parse spectrum file: {filepath}")   # Both attempts failed


def load_external_spectrum(
    filepath: str,                                   # Spectrum file to load
    wv_target: np.ndarray,                           # Wavenumber grid to interpolate onto
    sigma_smooth: float = 0.0,                       # Optional pre-smoothing (cm^-1)
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
    wv_file, spec_file = _try_load_two_column(filepath)   # Parse the file

    # Sort ascending in wavenumber
    sort_idx = np.argsort(wv_file)                   # Order that sorts the axis
    wv_file, spec_file = wv_file[sort_idx], spec_file[sort_idx]   # Apply it to both

    # Optional smoothing on the original grid
    if sigma_smooth > 0:                             # Smoothing requested
        dw = float(wv_file[1] - wv_file[0]) if len(wv_file) > 1 else 1.0   # Grid spacing
        sigma_px = sigma_smooth / max(abs(dw), 1e-6)   # Convert cm^-1 to pixels
        spec_file = ndimage.gaussian_filter1d(spec_file, sigma=sigma_px)   # Blur it

    # Interpolate onto target grid; edge-fill outside range
    interp_fn = interpolate.interp1d(
        wv_file, spec_file,
        kind="linear",                               # Linear interpolation
        bounds_error=False,                          # Allow out-of-range queries
        fill_value=(float(spec_file[0]), float(spec_file[-1])),   # Clamp to the end values
    )
    spec = interp_fn(wv_target).astype(np.float64)   # Resample onto the target grid

    return spec.astype(np.float32)                   # Raw interpolated intensity


def _perturb_spectrum(
    spectrum: np.ndarray,                            # Spectrum to perturb
    wv: np.ndarray,                                  # Its wavenumber grid
    shift_std: float,                                # Std of the rigid shift (cm^-1)
    width_jitter: float,                             # Std of the blur/sharpen jitter
    amp_jitter: float,                               # Std of the amplitude jitter
    rng: np.random.Generator,                        # RNG
    clip: bool = True,                               # Clip the result to [0,1]?
) -> np.ndarray:
    """
    Slightly perturb a spectrum: wavenumber shift + amplitude noise.

    shift_std   : std of wavenumber shift in cm⁻¹
    width_jitter: fractional change in peak widths (Gaussian blur / sharpen)
    amp_jitter  : std of multiplicative amplitude noise
    """
    spec = spectrum.copy().astype(np.float64)        # Work on a float64 copy

    # Rigid wavenumber shift (roll in index space)
    if shift_std > 0 and len(wv) > 1:                # Shift jitter enabled
        dw = float(wv[1] - wv[0])                    # Grid spacing
        shift_px = int(round(rng.normal(0, shift_std) / max(abs(dw), 1e-6)))   # cm^-1 -> bands
        spec = np.roll(spec, shift_px)               # Apply the rigid shift

    # Width jitter via Gaussian blur/sharpen
    if width_jitter > 0:                             # Width jitter enabled
        sigma_px = abs(rng.normal(0, width_jitter))  # Draw a blur width
        if sigma_px > 0.1:                           # Ignore negligible blurs
            spec = ndimage.gaussian_filter1d(spec, sigma=sigma_px)   # Broaden the peaks

    # Amplitude jitter
    if amp_jitter > 0:                               # Amplitude jitter enabled
        spec *= max(0.1, 1.0 + rng.normal(0, amp_jitter))   # Scale (never below 0.1x)

    # Do NOT re-normalize to [0,1] here — that would erase the shared-scale
    # relative intensities set in load_external_library and force every phase
    # back to peak=1. Only clip to the valid range (jitter, if enabled, can push
    # slightly out of [0,1]; with jitter disabled this is a no-op).
    if clip:                                         # Clipping requested
        spec = spec.clip(0.0, 1.0)                   # Bound to the valid range
    return spec.astype(np.float32)                   # Return the perturbed spectrum


def random_poly_background(
        wv,                                          # Wavenumber axis
        max_amplitude=0.3,                           # Max peak-to-peak amplitude
        rng=None,                                    # RNG (created if omitted)
        coeffs=None,                                 # Optional fixed coefficients
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

    if (rng is None) and (coeffs is None):           # Neither RNG nor coefficients given
        rng = np.random.default_rng()                # Make a default RNG

    # Normalize x to [-1,1] for stable coefficient generation
    x = (wv - wv.mean()) / (wv.max() - wv.min()) * 2   # Rescale the axis to ~[-1,1]
    
    if coeffs is None:                               # No fixed coefficients supplied
        # Random cubic coefficients
        # coeffs = rng.normal(size=np.random.randint(0,10))
        coeffs = rng.normal(size=10)                 # Draw 10 random coefficients

    bg = np.polyval(coeffs, x)                       # Evaluate the polynomial

    bg = (bg-bg.min())/(bg.max()-bg.min())           # Normalise the shape to [0,1]

    scale = rng.uniform(0, max_amplitude)            # Draw the background strength

    bg *= scale                                      # Scale it down

    return bg, coeffs                                # Background plus its coefficients


def generate_phase_spectra(
    n_phases: int,                                   # How many spectra to produce
    wv: np.ndarray,                                  # Wavenumber grid
    params: dict,                                    # Generation parameters
    external_library: Optional[List[np.ndarray]],    # Library to sample from (external mode)
    rng: np.random.Generator,                        # RNG
    phase_intensities: Optional[np.ndarray] = None,  # Per-phase relative intensity
    clip: bool = True,                               # Clip spectra to [0,1]?
) -> np.ndarray:
    """
    Generate one base spectrum per phase.

    Returns:
        spectra : (n_phases, W) float32, each row normalised to [0, 1]
    """
    W = len(wv)                                      # Band count
    spectra = np.zeros((n_phases, W), dtype=np.float32)   # Output array

    for k in range(n_phases):                        # One spectrum per phase
        if params["spectra_mode"] == "external" and external_library:   # Sample from the library
            idx = rng.integers(0, len(external_library))   # Random library row
            base = external_library[idx].copy()      # Copy it (do not mutate the library)
        else:                                        # Build a synthetic spectrum
            n_peaks = int(rng.integers(params["n_peaks_range"][0],
                                       params["n_peaks_range"][1] + 1))   # Peak count
            base = synthetic_gaussian_spectrum(
                wv, n_peaks, params["peak_width_range"], rng,
                peak_amp_range=params["peak_amp_range"],
            )                                        # Sum of random Gaussians
            background, _ = random_poly_background(wv,max_amplitude=0.3,rng=rng)   # Baseline drift
            base += background                       # Add the polynomial background

        # Perturb each phase slightly so they are not identical
        pert = _perturb_spectrum(
            base, wv,
            shift_std  = params.get("spectra_shift_std", 5.0),      # Rigid shift jitter
            width_jitter = params.get("spectra_width_jitter", 0.5), # Width jitter
            amp_jitter   = params.get("spectra_amp_jitter", 0.1),   # Amplitude jitter
            rng          = rng,                      # RNG
            clip         = clip,                     # Clip policy
        )
        # RELATIVE INTENSITY: scale this phase's whole spectrum by its factor
        scale = 1.0 if phase_intensities is None else float(phase_intensities[k])   # This phase's factor
        spectra[k] = (pert * scale).astype(np.float32)   # Store the scaled spectrum

    return spectra                                   # (n_phases, W) endmembers


# =============================================================================
# Boundary mixing
# =============================================================================

def compute_mixing_weights(
    phase_map: np.ndarray,                           # (X, Y) phase labels
    n_phases: int,                                   # Phase count
    mixing_type: str,                                # Transition profile name
    boundary_width: float,                           # Transition half-width (px)
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
    X, Y = phase_map.shape                           # Spatial size

    # One-hot initialisation
    weights = np.zeros((n_phases, X, Y), dtype=np.float32)   # Weight per phase per pixel
    for k in range(n_phases):                        # Each phase
        weights[k] = (phase_map == k).astype(np.float32)   # 1 inside that phase, else 0

    if mixing_type == "sharp" or boundary_width < 0.5:   # No blending requested
        return weights                               # Return the one-hot weights

    # Distance from every pixel to nearest pixel of every phase
    dists = np.full((n_phases, X, Y), fill_value=1e8, dtype=np.float32)   # Init to "far"
    for k in range(n_phases):                        # Each phase
        mask = phase_map == k                        # That phase's pixels
        if mask.any():                               # Phase is present
            dists[k] = ndimage.distance_transform_edt(~mask).astype(np.float32)   # Distance to it

    # Distance to nearest OTHER phase (exclude own)
    INF = 1e8                                        # Sentinel for "not applicable"
    dists_other = dists.copy()                       # Copy to mask out own-phase distances
    for k in range(n_phases):                        # Each phase
        dists_other[k][phase_map == k] = INF         # A pixel is not "other" to itself

    dist_to_boundary = dists_other.min(axis=0)           # (X, Y)
    nearest_other    = dists_other.argmin(axis=0)         # (X, Y)

    in_bz   = dist_to_boundary < boundary_width      # Inside the blending zone
    d_norm  = np.clip(dist_to_boundary / max(boundary_width, 1e-6), 0.0, 1.0)   # Normalised depth

    # g = fraction of spectrum given to nearest OTHER phase
    # g(0) = 0.5 (at boundary), g(1) = 0 (at boundary_width distance)
    if mixing_type == "linear":                      # Linear ramp
        g = np.where(in_bz, 0.5 * (1.0 - d_norm), 0.0)   # Falls linearly to 0
    elif mixing_type == "gaussian":                  # Gaussian falloff
        # bell centred at boundary (d=0), falls to ~0 at boundary_width (3-sigma)
        g = np.where(in_bz, 0.5 * np.exp(-4.5 * d_norm ** 2), 0.0)   # Bell profile
    elif mixing_type == "sigmoid":                   # Sigmoidal falloff
        # 0.25 * (1 - tanh(4*(d-0.5))): g(0)≈0.49, g(0.5)=0.25, g(1)≈0.009
        # The original 0.5*(1-tanh(3*(d-0.5))) gave g(0)≈0.95 — far too much mixing
        # at the boundary, causing both sides to swap spectra and produce an
        # intensity spike rather than a smooth transition.
        g = np.where(in_bz, 0.25 * (1.0 - np.tanh(4.0 * (d_norm - 0.5))), 0.0)   # Tanh profile
    else:                                            # Unknown mixing type
        g = np.zeros_like(dist_to_boundary)          # No blending

    # Subtract g from own phase, add g to nearest other
    for k in range(n_phases):                        # Each phase
        is_k          = phase_map == k               # Pixels belonging to phase k
        receives_from = (nearest_other == k) & ~is_k   # Pixels whose nearest other is k
        weights[k]    = np.where(is_k,          weights[k] - g, weights[k])   # Give away g
        weights[k]    = np.where(receives_from, weights[k] + g, weights[k])   # Receive g

    weights = np.clip(weights, 0.0, 1.0)             # Keep weights valid
    w_sum   = weights.sum(axis=0, keepdims=True)     # Total weight per pixel
    weights /= np.where(w_sum > 0, w_sum, 1.0)       # Renormalise to sum to 1

    return weights                                   # (n_phases, X, Y)


# =============================================================================
# Intra-phase spectral variation
# =============================================================================

def add_intra_phase_variation(
    cube: np.ndarray,                                # (X, Y, W) assembled cube
    phase_map: np.ndarray,                           # (X, Y) phase labels
    spectra: np.ndarray,                             # (n_phases, W) endmembers
    variation_amplitude: float,                      # Max fractional perturbation
    rng: np.random.Generator,                        # RNG
    clip: bool = True,                               # Clip the result to [0,1]?
) -> np.ndarray:
    """
    Add smooth within-phase spectral variation using a Gaussian-blurred random
    spatial field.  Each phase independently receives a smooth amplitude
    modulation of its base spectrum.

    variation_amplitude : max fractional perturbation (e.g. 0.05 = ±5%)
    """
    if variation_amplitude <= 0:                     # Feature disabled
        return cube                                  # Return unchanged

    X, Y, W = cube.shape                             # Cube shape
    sigma_blur = max(1.0, min(X, Y) / 6.0)           # Blur scale of the variation field

    for k in range(len(spectra)):                    # One field per phase
        mask = phase_map == k                        # That phase's pixels
        if not mask.any():                           # Phase absent
            continue                                 # Skip it

        # Smooth random field in [−1, +1]
        field = rng.standard_normal((X, Y)).astype(np.float32)   # White noise
        field = ndimage.gaussian_filter(field, sigma=sigma_blur)   # Blur to correlate it
        lo, hi = field.min(), field.max()            # Its range
        if hi > lo:                                  # Guard against a constant field
            field = 2.0 * (field - lo) / (hi - lo) - 1.0  # [-1, 1]

        # Spatially modulate the base spectrum amplitude
        # perturbation at (x,y) = field[x,y] * amplitude * spectra[k]
        perturb = (
            variation_amplitude * field[:, :, np.newaxis]  # (X, Y, 1)
            * spectra[k][np.newaxis, np.newaxis, :]         # (1, 1, W)
        )                                            # Broadcast to (X, Y, W)
        cube = np.where(
            mask[:, :, np.newaxis],                  # Only inside this phase
            cube + perturb.astype(np.float32),       # Apply the perturbation
            cube,                                    # Leave other phases untouched
        )

    return np.clip(cube, 0.0, 1.0) if clip else cube   # Optionally clip to valid range


# =============================================================================
# Depth (subsurface) spectral mixing
# =============================================================================

def apply_depth_mixing(cube, phase_map, spectra, background_phase=0,
                       edge_mix=0.4, decay_length=6.0, beneath_map=None, rng=None,
                       probing_depth_nm=None, thickness_range=(200.0, 500.0),
                       thickness_profile="sqrt", thickness_map_in=None, clip=True):
    """Emulate the finite probing depth of AFM-IR: a spectrum measured on a
    particle carries a subsurface contribution from whatever lies beneath it, and
    a thicker top particle screens that subsurface signal more.

    Two models are available.

    PHYSICAL (set probing_depth_nm): Beer-Lambert screening, following Dazzi et al.,
    Anal. Chem. 2024, 96, 17931 (their eq 5). The signal fraction coming from
    BENEATH the top particle is

        alpha(x,y) = exp( -t(x,y) / probing_depth_nm )

    where t(x,y) is the local verticle thickness of the top particle (nm) and
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
    X, Y, W = cube.shape                             # Cube shape
    zero = np.zeros((X, Y), dtype=np.float32)        # Reusable all-zero map

    # subsurface target spectrum, per pixel: whatever is beneath the top particle
    if beneath_map is None:                          # Caller supplied no beneath map
        beneath = np.full(phase_map.shape, background_phase, dtype=np.int32)   # Assume substrate
    else:                                            # Beneath map available
        beneath = beneath_map.astype(np.int32)       # Use it
    sub_spec = spectra[beneath]                              # (X, Y, W) fancy-index

    labels = [k for k in np.unique(phase_map) if k != background_phase]   # Non-background phases
    alpha_map     = np.zeros((X, Y), dtype=np.float32)   # Subsurface fraction per pixel
    thickness_map = np.zeros((X, Y), dtype=np.float32)       # nm

    use_physical = probing_depth_nm is not None and probing_depth_nm > 0.0   # Which model to run

    if not use_physical:
        # ---- LEGACY: exponential in pixel edge-distance ----
        if edge_mix <= 0.0 or decay_length <= 0.0:   # Legacy model disabled too
            return cube, zero, zero                  # Nothing to do
        for k in labels:                             # Each non-background phase
            mask = phase_map == k                    # That phase's pixels
            if not mask.any():                       # Phase absent
                continue                             # Skip
            d_in = ndimage.distance_transform_edt(mask).astype(np.float32)   # Inward distance
            a = edge_mix * np.exp(-d_in / float(decay_length))   # Exponential in edge distance
            same = mask & (beneath == k)             # Sitting on the SAME phase
            a = np.where(mask & ~same, a, 0.0).astype(np.float32)   # No mix for same-phase
            alpha_map = np.where(mask, a, alpha_map)   # Record this phase's alpha
        a3 = alpha_map[:, :, np.newaxis]             # Broadcast over bands
        mixed = (1.0 - a3) * cube + a3 * sub_spec    # Two-layer blend
        return ((np.clip(mixed, 0.0, 1.0) if clip else mixed).astype(np.float32),
                alpha_map, thickness_map)            # Cube + diagnostics

    rng = rng or np.random.default_rng()             # Need an RNG to sample thicknesses

    # Per-phase thickness over each particle's full extent
    thick_by_phase = {}                              # phase -> (X, Y) thickness map
    for k in labels:                                 # Each non-background phase
        Mk = (phase_map == k) | (beneath == k)       # Its full footprint (top or buried)
        if not Mk.any():                             # Phase absent
            continue                                 # Skip
        tk = np.zeros((X, Y), dtype=np.float32)      # This phase's thickness map
        comp, ncomp = ndimage.label(Mk)              # Split into individual particles
        for c in range(1, ncomp + 1):                # Each connected particle
            cm = comp == c                           # That particle's pixels
            d = ndimage.distance_transform_edt(cm).astype(np.float32)   # Inward distance
            dmax = float(d.max())                    # Distance at the particle centre
            if dmax <= 0.0:                          # Degenerate particle
                continue                             # Skip it
            r = d / dmax                                    # 0 at rim -> 1 at centre
            prof = np.sqrt(r) if thickness_profile == "sqrt" else r   # Puddle cap or wedge
            T_max = float(rng.uniform(*thickness_range))   # This particle's peak thickness
            tk[cm] = (T_max * prof)[cm]              # Write its thickness profile
        thick_by_phase[k] = tk                       # Store it

    # Per-pixel top thickness (visible particle) and lower thickness (particle
    # immediately beneath; 0 where the layer beneath is the substrate)
    t_top   = np.zeros((X, Y), dtype=np.float32)     # Thickness of the visible particle
    t_lower = np.zeros((X, Y), dtype=np.float32)     # Thickness of the one beneath
    for k in labels:                                 # Each non-background phase
        if k not in thick_by_phase:                  # No thickness map for it
            continue                                 # Skip
        m = phase_map == k                           # Where it is on top
        t_top[m] = thick_by_phase[k][m]              # Record it as the top thickness
        bm = beneath == k                            # Where it lies beneath something
        t_lower[bm] = thick_by_phase[k][bm]          # Record it as the lower thickness

    if thickness_map_in is not None:                        # optional explicit t_top override
        t_top = np.asarray(thickness_map_in, dtype=np.float32)   # Use the supplied map

    p = float(probing_depth_nm)                      # Probing depth (nm)
    e_top  = np.exp(-t_top / p).astype(np.float32)   # Transmission through the top particle
    e_both = np.exp(-(t_top + t_lower) / p).astype(np.float32)   # Through top + lower
    w_top   = (1.0 - e_top)                          # Weight of the top particle
    w_lower = (e_top - e_both)                       # Weight of the particle beneath
    w_sub   = e_both                                 # Weight of the substrate

    part       = (phase_map != background_phase)     # Pixels that are on a particle
    lower_spec = spectra[beneath]                           # (X,Y,W); = substrate where beneath is bg
    bg_spec    = spectra[background_phase]                  # (W,)

    new_cube = (w_top[:, :, None] * cube
                + w_lower[:, :, None] * lower_spec
                + w_sub[:, :, None] * bg_spec[None, None, :]).astype(np.float32)   # Three-layer blend
    mixed = np.where(part[:, :, None], new_cube, cube).astype(np.float32)   # Only on particles

    # report: alpha = total subsurface fraction (1 - top); thickness = t_top
    alpha_map     = np.where(part, 1.0 - w_top, 0.0).astype(np.float32)   # Subsurface fraction
    thickness_map = t_top                            # Report the top thickness
    return ((np.clip(mixed, 0.0, 1.0) if clip else mixed).astype(np.float32),
            alpha_map, thickness_map)                # Cube + diagnostics


# =============================================================================
# Cube assembly
# =============================================================================

def assemble_cube(
    phase_map: np.ndarray,                           # (X, Y) phase labels
    spectra: np.ndarray,                             # (n_phases, W) endmembers
    mixing_weights: np.ndarray,                      # (n_phases, X, Y) blend weights
    noise_std: float,                                # Gaussian noise level
    intra_phase_variation: float,                    # Within-phase variation amplitude
    rng: np.random.Generator,                        # RNG
    clip: bool = True,                               # Clip the result to [0,1]?
) -> np.ndarray:
    """
    Assemble the hyperspectral cube.

    cube[x, y, w] = sum_k  weights[k,x,y] * spectra[k,w]  +  noise

    Args:
        mixing_weights        : (n_phases, X, Y)
        spectra               : (n_phases, W)
        noise_std             : Gaussian noise standard deviation
        intra_phase_variation : fractional within-phase spectral amplitude variation

    Returns:
        cube : (X, Y, W) float32 in [0, 1]
    """
    # einsum: weights (k,x,y) × spectra (k,w) → cube (x,y,w)
    cube = np.einsum("kij,kw->ijw", mixing_weights, spectra).astype(np.float32)   # Weighted sum

    # Within-phase variation
    cube = add_intra_phase_variation(cube, phase_map, spectra, intra_phase_variation, rng, clip=clip)   # Modulate

    # Additive Gaussian noise
    if noise_std > 0:                                # Noise requested
        cube += rng.normal(0.0, noise_std, size=cube.shape).astype(np.float32)   # Add it

    return np.clip(cube, 0.0, 1.0) if clip else cube   # Optionally clip to valid range


# =============================================================================
# Per-cube generator
# =============================================================================

def generate_one_cube(
    params: dict,                                    # Generation parameters
    wv: np.ndarray,                                  # Wavenumber grid
    external_library: Optional[List[np.ndarray]],    # Spectrum library (external mode)
) -> dict:
    """
    Generate one complete synthetic hyperspectral cube.

    Returns dict with keys: cube, phase_map, spectra, wavenumbers, params
    """
    rng = np.random.default_rng(params["seed"])      # Deterministic per-cube RNG
    X, Y   = params["X"], params["Y"]                # Spatial size
    n_ph   = params["n_phases"]                      # Phase count

    clip_output = bool(params.get("clip_output", True))   # Whether to clip to [0,1]
    if params.get("phase_intensities") is not None:  # Explicit per-phase intensities given
        phase_intensities = np.asarray(params["phase_intensities"], dtype=np.float32)   # Use them
        if phase_intensities.shape[0] != n_ph:       # Must have one per phase
            raise ValueError(f"phase_intensities has {phase_intensities.shape[0]} "
                             f"entries but n_phases={n_ph}")   # Fail loudly
    elif params.get("relative_intensity_range") is not None:   # A range was given instead
        lo, hi = params["relative_intensity_range"]  # Unpack the range
        phase_intensities = rng.uniform(float(lo), float(hi), size=n_ph).astype(np.float32)   # Sample
    else:                                            # Neither given
        phase_intensities = np.ones(n_ph, dtype=np.float32)   # All phases peak at 1

    # 1. Phase map (+ beneath map: what lies under each top particle at overlaps)
    phase_map, beneath_map = generate_phase_map(
        X=X, Y=Y,                                    # Spatial size
        n_phases       = n_ph,                       # Phase count
        feature_sizes  = params["feature_sizes"],    # Per-phase feature sizes
        curvatures     = params["curvatures"],       # Per-phase curvatures
        phase_fractions= params["phase_fractions"],  # Per-phase area fractions
        morphology     = params["morphology"],       # dispersed | voronoi
        rng            = rng,                        # RNG
    )

    # 2. Spectra
    spectra = generate_phase_spectra(n_ph, wv, params, external_library, rng,
                                     phase_intensities=phase_intensities,
                                     clip=clip_output)   # One endmember per phase

    # 3. Mixing weights
    mixing_weights = compute_mixing_weights(
        phase_map      = phase_map,                  # Phase labels
        n_phases       = n_ph,                       # Phase count
        mixing_type    = params["mixing_type"],      # Transition profile
        boundary_width = params["boundary_width"],   # Transition half-width
    )

    # 4. Assemble
    cube = assemble_cube(
        phase_map             = phase_map,           # Phase labels
        spectra               = spectra,             # Endmembers
        mixing_weights        = mixing_weights,      # Blend weights
        noise_std             = params["noise_std"], # Detector noise
        intra_phase_variation = params["intra_phase_variation"],   # Within-phase variation
        rng                   = rng,                 # RNG
        clip                  = clip_output,         # Clip policy
    )

    # 5. Depth (subsurface) mixing 
    alpha_map = None                                 # Subsurface-fraction diagnostic
    thickness_map = None                             # Thickness diagnostic
    probing_depth_nm = params.get("depth_probing_depth_nm", None)   # set -> PHYSICAL model
    legacy_on = (params.get("depth_edge_mix", 0.0) > 0.0
                 and params.get("depth_decay_length", 0.0) > 0.0)   # Legacy model enabled?
    if (probing_depth_nm is not None and probing_depth_nm > 0.0) or legacy_on:   # Either model on
        cube, alpha_map, thickness_map = apply_depth_mixing(
            cube, phase_map, spectra,                # Cube and its structure
            background_phase = params.get("depth_background_phase", 0),   # Substrate label
            edge_mix         = params.get("depth_edge_mix", 0.4),         # Legacy edge mix
            decay_length     = params.get("depth_decay_length", 6.0),     # Legacy decay length
            beneath_map      = beneath_map,          # What lies under each particle
            rng              = rng,                  # RNG (per-particle thickness)
            probing_depth_nm = probing_depth_nm,     # Physical probing depth
            thickness_range  = tuple(params.get("depth_thickness_range", (200.0, 500.0))),   # Thickness range
            thickness_profile= params.get("depth_thickness_profile", "sqrt"),   # Profile shape
            clip             = clip_output,          # Clip policy
        )

    return dict(cube=cube, phase_map=phase_map, beneath_map=beneath_map,
                spectra=spectra, phase_intensities=phase_intensities,
                wavenumbers=wv, params=params,
                depth_alpha=alpha_map, depth_thickness=thickness_map)   # Everything the caller needs


# =============================================================================
# Parameter sampling 
# =============================================================================

def sample_parameter_sets(args, n_cubes: int, rng: np.random.Generator) -> List[dict]:   # Draw CLI-range params
    """
    Draw n_cubes parameter dictionaries, each value sampled uniformly from
    the CLI-specified range.  Discrete parameters (n_phases) are drawn as
    integers; continuous parameters are drawn as floats.
    """
    mixing_choices    = args.mixing_types            # Allowed boundary-mixing models
    morphology_choices = args.morphology_types       # Allowed morphologies
    param_sets = []                                  # Output list

    for i in range(n_cubes):                         # One parameter dict per cube
        n_phases = int(rng.integers(args.n_phases[0], args.n_phases[1] + 1))   # Phase count
        n_feat   = n_phases - 1  # number of non-background phases

        # Base values drawn from the overall range; per-phase variation added
        base_feat = rng.uniform(*args.feature_size)      # Cube-level feature size
        base_curv = rng.uniform(*args.boundary_curvature)   # Cube-level curvature

        feature_sizes = [
            float(np.clip(base_feat * rng.uniform(0.70, 1.30),
                          args.feature_size[0], args.feature_size[1]))
            for _ in range(n_feat)
        ]                                            # Per-phase sizes, jittered about the base
        curvatures = [
            float(np.clip(base_curv * rng.uniform(0.85, 1.15),
                          args.boundary_curvature[0], args.boundary_curvature[1]))
            for _ in range(n_feat)
        ]                                            # Per-phase curvatures, jittered

        # Phase fractions: drawn uniformly, forced to sum <= 0.85 (leaves >=15% background)
        phase_fractions = []                         # Area fraction per non-bg phase
        remaining = 0.85                             # Area budget left to allocate
        for j in range(n_feat):                      # Allocate phase by phase
            hi     = min(args.phase_fraction[1], remaining / max(1, n_feat - j))   # Upper bound
            lo     = min(args.phase_fraction[0], hi)   # Lower bound (never above hi)
            frac   = float(rng.uniform(lo, hi))      # Draw this phase's fraction
            phase_fractions.append(frac)             # Record it
            remaining -= frac                        # Deduct from the budget

        params = dict(
            cube_id                = i,              # Index of this cube
            X                      = args.image_size[0],   # Cube height
            Y                      = args.image_size[1],   # Cube width
            n_phases               = n_phases,       # Phase count
            feature_sizes          = feature_sizes,  # Per-phase feature sizes
            curvatures             = curvatures,     # Per-phase curvatures
            phase_fractions        = phase_fractions,   # Per-phase area fractions
            morphology             = str(rng.choice(morphology_choices)),   # Chosen morphology
            boundary_width         = float(rng.uniform(*args.boundary_width)),   # Transition width
            mixing_type            = str(rng.choice(mixing_choices)),   # Chosen mixing model
            noise_std              = float(rng.uniform(*args.noise_std)),   # Detector noise
            intra_phase_variation  = float(rng.uniform(*args.intra_phase_variation)),   # Within-phase variation
            spectra_mode           = args.spectra_mode,   # synthetic | external
            n_peaks_range          = list(args.n_peaks),  # Synthetic peak count range
            peak_width_range       = list(args.peak_width),   # Synthetic peak width range
            peak_amp_range         = list(args.peak_amp_range),   # Synthetic peak amplitude range
            spectra_shift_std      = args.spectra_shift_std,      # Shift jitter
            spectra_width_jitter   = args.spectra_width_jitter,   # Width jitter
            spectra_amp_jitter     = args.spectra_amp_jitter,     # Amplitude jitter
            seed                   = int(rng.integers(0, 2 ** 31)),   # Per-cube seed
        )
        param_sets.append(params)                    # Collect it

    return param_sets                                # All parameter dicts


# =============================================================================
# I/O helpers
# =============================================================================

def _params_to_str(params: dict) -> str:             # Render a dict as "key: value" lines
    lines = []                                       # Output lines
    for k, v in params.items():                      # Each parameter
        lines.append(f"{k}: {v}")                    # Format it
    return "\n".join(lines)                          # Join into one string


def save_result(
    output_dir: Path,                                # Run directory
    result: dict,                                    # Output of generate_one_cube
    save_png: bool,                                  # Also write an overview PNG?
) -> None:
    """Save cube, phase map, spectra, wavenumbers, and params for one cube."""
    params    = result["params"]                     # This cube's parameters
    cube_dir  = output_dir / f"cube_{params['cube_id']:04d}"   # Per-cube subdirectory
    cube_dir.mkdir(parents=True, exist_ok=True)      # Create it

    np.save(str(cube_dir / "hyperspectral_cube.npy"), result["cube"])   # The cube
    np.save(str(cube_dir / "phase_map.npy"),          result["phase_map"])   # Phase labels
    np.save(str(cube_dir / "spectra.npy"),            result["spectra"])   # Endmembers
    np.save(str(cube_dir / "wavenumbers.npy"),        result["wavenumbers"])   # Wavenumber grid

    with open(cube_dir / "params.txt", "w") as f:    # Human-readable parameter dump
        f.write(_params_to_str(params))              # Write it

    if save_png and HAS_MPL:                         # Overview figure requested and possible
        _save_phase_png(cube_dir, result["phase_map"], result["spectra"],
                        result["wavenumbers"], params)   # Draw and save it


def _save_phase_png(cube_dir, phase_map, spectra, wv, params):   # Overview figure per cube
    n_ph   = params["n_phases"]                      # Phase count
    cmap   = plt.cm.get_cmap("tab10", n_ph)          # Discrete colour map
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Two panels

    # Phase map
    ax = axes[0]                                     # Left panel
    im = ax.imshow(phase_map, origin="lower", cmap=cmap,
                   vmin=-0.5, vmax=n_ph - 0.5, interpolation="nearest")   # Draw the labels
    cbar = fig.colorbar(im, ax=ax, ticks=range(n_ph))   # One tick per phase
    cbar.ax.set_yticklabels([f"Phase {k}" for k in range(n_ph)], fontsize=7)   # Label them
    ax.set_title(f"Phase map  ({params['morphology']})", fontsize=9)   # Panel title
    ax.set_xlabel("x  [px]"); ax.set_ylabel("y  [px]")   # Axis labels

    # Base spectra per phase
    ax = axes[1]                                     # Right panel
    for k in range(n_ph):                            # One curve per phase
        lbl = "background" if k == 0 else f"Phase {k}"   # Legend label
        ax.plot(wv, spectra[k], color=cmap(k), lw=1.4, label=lbl)   # Plot the endmember
    ax.set_xlabel("Wavenumber  [cm⁻¹]" if wv[0] > 10 else "Wavenumber  [norm.]")   # x label
    ax.set_ylabel("Intensity  [norm.]")              # y label
    ax.set_title("Base spectra per phase", fontsize=9)   # Panel title
    ax.legend(fontsize=7, loc="upper right")         # Legend
    ax.grid(True, alpha=0.3)                         # Light grid

    plt.tight_layout()                               # Tidy the layout
    plt.savefig(str(cube_dir / "overview.png"), dpi=120, bbox_inches="tight")   # Write the PNG
    plt.close()                                      # Free the figure


def load_external_library(spectra_dir: str, wv: np.ndarray,
                           sigma_smooth: float) -> List[np.ndarray]:   # Load a spectra folder
    """
    Load all 2-column spectrum files (*.csv, *.txt, *.dat) from spectra_dir.

    Returns a list of (W,) float32 arrays normalised to [0, 1].
    """
    spectra_dir = Path(spectra_dir)                  # As a Path
    patterns    = ["*.csv", "*.txt", "*.dat", "*.CSV", "*.TXT"]   # Accepted extensions
    files: List[Path] = []                           # Matching files
    for pat in patterns:                             # Each pattern
        files.extend(spectra_dir.glob(pat))          # Collect matches
    files = sorted(set(files))                       # Deduplicate and order

    if not files:                                    # Nothing found
        raise FileNotFoundError(
            f"No spectrum files (*.csv / *.txt / *.dat) found in {spectra_dir}"
        )                                            # Fail loudly

    library = []                                     # Loaded spectra
    for fp in files:                                 # Each file
        try:
            spec = load_external_spectrum(str(fp), wv, sigma_smooth)   # Load + interpolate
            library.append(spec)                     # Keep it
        except Exception as e:                       # Unparseable file
            warnings.warn(f"Skipping {fp.name}: {e}")   # Warn and continue

    if not library:                                  # Every file failed
        raise RuntimeError(f"Could not load any spectra from {spectra_dir}")   # Fail loudly

    # Normalize
    g_min = min(float(s.min()) for s in library)     # Global minimum across the library
    g_max = max(float(s.max()) for s in library)     # Global maximum across the library
    denom = (g_max - g_min) if (g_max > g_min) else 1.0   # Guard against a flat library
    library = [((s.astype(np.float64) - g_min) / denom).astype(np.float32)
               for s in library]                     # Apply the shared scaling
    print(f"  Shared-scale norm: global min={g_min:.4g}, max={g_max:.4g} "
          f"-> all spectra in [0,1], relative heights preserved")   # Report the scaling

    print(f"  Loaded {len(library)} external spectra from {spectra_dir}")   # Report the count
    return library                                   # The normalised library


# =============================================================================
# CLI
# =============================================================================

def parse_args():                                    # Build and parse the CLI
    p = argparse.ArgumentParser(
        description="Generate synthetic AFM-IR hyperspectral cubes for ANP training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,   # Show defaults in --help
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

    args = p.parse_args()                            # Parse the command line

    # Validation
    if args.spectra_mode == "external" and args.spectra_dir is None:   # External needs a folder
        p.error("--spectra_dir is required when --spectra_mode external")   # Fail loudly
    if args.n_phases[0] < 2:                         # Need background + at least one phase
        p.error("--n_phases MIN must be ≥ 2 (at least background + 1 dispersed phase)")   # Fail loudly
    if args.image_size[0] < 4 or args.image_size[1] < 4:   # Too small to be meaningful
        p.error("--image_size must be at least 4×4")   # Fail loudly

    return args                                      # Validated arguments


# =============================================================================
# Main
# =============================================================================

def main():                                          # Dataset-generation entry point
    args = parse_args()                              # Parse and validate the CLI
    master_rng = np.random.default_rng(args.seed)    # Master RNG for parameter sampling

    # Wavenumbers
    wv = np.linspace(args.wv_min, args.wv_max, args.n_wavenumbers,
                     dtype=np.float32)               # Shared spectral axis

    # Load external library
    external_library: Optional[List[np.ndarray]] = None   # None in synthetic mode
    if args.spectra_mode == "external":              # External spectra requested
        external_library = load_external_library(
            args.spectra_dir, wv, args.spectra_smooth
        )                                            # Load them once

    # Establish output directory
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")   # Timestamp for this run
    run_dir    = Path(args.output_dir) / f"synth_afmir_{ts}"   # Run directory path
    run_dir.mkdir(parents=True, exist_ok=True)       # Create it
    print(f"\nOutput directory: {run_dir}")          # Tell the user where it went

    # Get generation parameters
    param_sets = sample_parameter_sets(args, args.n_cubes, master_rng)   # One dict per cube

    # Save metadata
    dataset_meta = dict(
        generated_at    = ts,                        # When this dataset was made
        n_cubes         = args.n_cubes,              # How many cubes
        image_size_XY   = args.image_size,           # Spatial size
        wv_min          = args.wv_min,               # Spectral range start
        wv_max          = args.wv_max,               # Spectral range end
        n_wavenumbers   = args.n_wavenumbers,        # Band count
        n_phases_range  = args.n_phases,             # Phase-count range
        feature_size_range  = args.feature_size,     # Feature-size range
        boundary_curvature_range = args.boundary_curvature,   # Curvature range
        phase_fraction_range = args.phase_fraction,  # Area-fraction range
        morphology_types = args.morphology_types,    # Allowed morphologies
        boundary_width_range = args.boundary_width,  # Transition-width range
        mixing_types    = args.mixing_types,         # Allowed mixing models
        noise_std_range = args.noise_std,            # Noise range
        intra_phase_variation_range = args.intra_phase_variation,   # Variation range
        spectra_mode    = args.spectra_mode,         # synthetic | external
        spectra_dir     = args.spectra_dir,          # External spectra folder
        spectra_smooth  = args.spectra_smooth,       # External smoothing width
        n_peaks_range   = args.n_peaks,              # Synthetic peak-count range
        peak_width_range = args.peak_width,          # Synthetic peak-width range
        spectra_shift_std = args.spectra_shift_std,  # Shift jitter
        spectra_width_jitter = args.spectra_width_jitter,   # Width jitter
        spectra_amp_jitter = args.spectra_amp_jitter,       # Amplitude jitter
        master_seed     = args.seed,                 # Master seed
    )
    with open(run_dir / "dataset_params.txt", "w") as f:   # Dataset-level metadata file
        f.write("# Dataset-level generation parameters\n")   # Header line
        f.write(_params_to_str(dataset_meta))        # The parameters
    np.save(str(run_dir / "wavenumbers.npy"), wv)    # Shared wavenumber grid

    # Generate cubes
    t0     = time.time()                             # Start of the timing window
    errors = []                                      # Failed cubes (index, message)
    print(f"Generating {args.n_cubes} cubes "
          f"({args.image_size[0]}×{args.image_size[1]}×{args.n_wavenumbers})...\n")   # Banner

    for i, params in enumerate(param_sets):          # Each cube
        try:
            result = generate_one_cube(params, wv, external_library)   # Generate it
            save_result(run_dir, result, save_png=args.save_png)       # Save it
            elapsed = time.time() - t0               # Time so far
            rate    = (i + 1) / elapsed              # Cubes per second
            eta     = (args.n_cubes - i - 1) / max(rate, 1e-6)   # Estimated time remaining
            print(f"  [{i+1:4d}/{args.n_cubes}]  "
                  f"phases={params['n_phases']}  "
                  f"morph={params['morphology']:9s}  "
                  f"mix={params['mixing_type']:8s}  "
                  f"bw={params['boundary_width']:.1f}px  "
                  f"eta {eta:.0f}s",
                  end="\r")                          # Overwrite the same line
        except Exception as e:                       # This cube failed
            errors.append((i, str(e)))               # Record the failure
            warnings.warn(f"Cube {i} failed: {e}")   # Warn and continue

    elapsed = time.time() - t0                       # Total generation time
    print(f"\n\nDone.  {args.n_cubes - len(errors)}/{args.n_cubes} cubes generated "
          f"in {elapsed:.1f}s  ({elapsed/max(args.n_cubes,1):.2f}s/cube)")   # Summary
    if errors:                                       # Some cubes failed
        print(f"  {len(errors)} failed: {errors}")   # List them
    print(f"\nDataset saved to:\n  {run_dir}\n")     # Where the output went



if __name__ == "__main__":                           # Run as a script
    main()                                           # Generate the dataset
