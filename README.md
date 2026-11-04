# CACNP for sparse AFM-IR hyperspectral reconstruction

The convolutional-attentive conditional neural process (CACNP) reconstructs AFM-IR hyperspectral cubes from spectra measured at a subset of spatial locations and full-field images acquired at selected wavenumbers (structure images). The pretrained model predicts spectral intensities and voxelwise uncertainty without sample-specific retraining.

This repository provides model training and synthetic-data generation code, a trained checkpoint, two experimental datasets, training-spectrum metadata, and an example notebook for reconstruction and uncertainty calibration. The notebook illustrates use of the method; this release does not include the scripts for generating every manuscript figure or comparison.

## Repository contents

```text
.
├── README.md
├── example_reconstruction_multi_particle.ipynb
├── code/                         # Three Python modules
├── models/                       # Pretrained checkpoint
├── data/
│   ├── DATA_README.txt
│   ├── experimental/
│   │   ├── single_particle/
│   │   └── multi_particle/
│   └── metadata/                 # Training-spectrum metadata CSV
├── environment/                  # Dependencies and Linux environment export
└── outputs/                      # Generated locally; ignored by Git
```

Run terminal commands from the repository root. The Python scripts stay together in `code/`, so their existing imports work when invoked directly. The notebook finds the repository and adds `code/` to its import path automatically. No package installation or environment-variable setup is needed beyond installing dependencies.

| File or directory | Contents |
| --- | --- |
| `code/conv_cnp_afmir_prob.py` | CACNP architecture, Lightning training wrapper, training CLI, normalization, cube loading, and basic metrics. |
| `code/synth_onthefly.py` | Live synthetic generation, measurement artifacts, and context/structure-image selection. |
| `code/generate_synth_afmir.py` | Underlying morphology generation, spectral assembly, and mixing routines. |
| `models/model_weights.ckpt` | Pretrained checkpoint used by the example. |
| `example_reconstruction_multi_particle.ipynb` | Sparse reconstruction, held-out-context uncertainty calibration, image comparisons, and spectral comparisons. |
| `data/experimental/` | Single- and multi-particle AFM-IR cubes, spectral axes, and associated scanning-probe images. |
| `data/DATA_README.txt` | Experimental-data description and acquisition information. |
| `data/metadata/spectral_corpus_metadata.csv` | Compound identifiers, spectrum sources, and processing metadata associated with the training spectral corpus. |
| `environment/requirements.txt` | Python dependency list (not an exact training-environment lockfile). |
| `environment/CACNP_env.txt` | Linux environment export supplied as a record of package versions. |

The three Python modules above contain the project code needed for training and inference; separate `anp_afmir.py` and `cnp_afmir.py` modules are not required. The main classes are `CACNP` and `CACNPLightningModule`. Compatibility aliases for older ConvCNP names remain available.

## Installation and example

Use Python 3.10 or a compatible environment. Install a PyTorch build appropriate for your platform and, for GPU use, your CUDA installation. The Python dependencies for the example are NumPy, SciPy, Matplotlib, PyTorch, PyTorch Lightning, and JupyterLab. For example, after installing PyTorch:

```bash
python -m pip install -r environment/requirements.txt
```

HDF5 input and TensorBoard logging additionally require `h5py` and `tensorboard`, respectively. `environment/CACNP_env.txt` records Linux-specific packages and build identifiers, including pip-installed packages; it is a reference environment export rather than a portable installation specification for every operating system. Compatibility of a newly installed environment should be checked by running the example.

From the repository root, start JupyterLab:

```bash
jupyter lab example_reconstruction_multi_particle.ipynb
```

Ensure the checkpoint is available at `models/model_weights.ckpt`, then run the notebook cells in order. The notebook uses CUDA when available and otherwise runs on CPU. CPU inference can be slower. The training spectral corpus is **not required for inference**.

The example uses:

- The multi-particle cube, with shape `(128, 128, 277)`.
- Nested random context sets containing 0.1%, 0.5%, 1%, 5%, and 10% of spatial pixels, with seed 307.
- Three structure-image channels, `[107, 72, 242]`, corresponding to 1189, 1049, and 1729 cm⁻¹. These are zero-based indices and can be changed in the configuration cell. Set `STRUCTURE_IMAGE_BANDS = None` to use automatic selection and set `N_STRUCTURE_IMAGES` to the desired count.
- Channel 97 (1149 cm⁻¹) for the image comparison.

Context fractions count spatially acquired spectra; full-field structure images are additional inputs. The example selects measurements retrospectively from a fully acquired cube and does not control an instrument.

Configuration variables also control plot limits, random seeds, calibration splitting, and figure saving. Saved notebook figures go into `outputs/`, which is created automatically. The final cell compares ground-truth and reconstructed spectra at four configurable locations for every context fraction, with calibrated ±2σ bands and residual plots. Panel titles identify whether each location is part of the measured context.

Keep `%matplotlib inline` **after** importing the CACNP module if Matplotlib reports that `FigureCanvasAgg` cannot display figures.

## Normalization and uncertainty calibration

For each prediction, the notebook computes one intensity mean and standard deviation using only the context spectra supplied to that pass. Predicted means are converted back to physical units by restoring both offset and scale. Predicted standard deviations are multiplied by the scale only.

Calibration uses two model passes per context fraction:

1. Randomly hold out approximately 30% of measured context positions and condition on the remaining 70%. Normalization uses only the retained context.
2. Compute the global multiplier `s = sqrt(mean(((prediction - measurement) / sigma)**2))` over all channels of the held-out spectra.
3. Predict with the complete context and multiply its predicted standard deviations by `s`.

The same structure images are available in both passes. The notebook retains `sigmas_raw`, `sigmas_calibrated`, `calibration_scales`, and the context/calibration masks. Both uncertainty arrays are in mV. Image panels display calibrated σ; spectral bands display calibrated ±2σ.

The reported coverage is the fraction satisfying `abs(measurement - prediction) <= 2*sigma` at spatial positions outside the full context, over all spectral channels. Both calibration and coverage include the supplied structure-image channels. Coverage is evaluated for demonstration and is not used to fit the multiplier.

This empirical variance scaling does not guarantee 95.4% coverage. It transfers a multiplier estimated with reduced context to the full-context prediction, and its effectiveness depends on the available measurements. The main model module returns uncalibrated sigma; the notebook implements the post-hoc calibration.

## Experimental data

The datasets contain polystyrene and poly(methyl methacrylate) particles embedded in epoxy, acquired using a Bruker Dimension IconIR system.

| Dataset | Cube filename | Shape | Field of view |
| --- | --- | --- | --- |
| Single particle | `data/experimental/single_particle/Single_PSPMMAEXPOXY_hcube.npy` | `(64, 64, 277)` | 1.5 × 1.5 µm |
| Multi particle | `data/experimental/multi_particle/Multi_PSPMMAEXPOXY_hcube.npy` | `(128, 128, 277)` | 20 × 20 µm |

Cube axes are `(row, column, spectral channel)` and spectral intensities are in mV. Each dataset includes its wavenumber array, topography, phase, and amplitude-error images. The spectral grid is 761–1865 cm⁻¹ in 4 cm⁻¹ increments. The multi-particle wavenumber filename retains the spelling `Mulit_PSPMMAEXPOXY_wavenum.npy`; the notebook uses that exact filename.

The checkpoint expects the published spectral grid and channel ordering. To use another compatible cube, change the cube and wavenumber paths, structure-image indices, and spectral comparison locations as appropriate for that dataset.

## Training spectral corpus and metadata

The experimental infrared spectra used to generate synthetic training cubes are **not redistributed because redistribution is restricted**. Instead, `data/metadata/spectral_corpus_metadata.csv` provides identifying and provenance information associated with the corpus. It contains metadata, not spectral intensity arrays, and cannot itself be supplied as the training spectrum library.

The CSV fields include:

- `compound_id`, `compound_name`, and `cas_rn`: compound identifiers and names, where available.
- `n_spectra`, `chosen_index`, and `nist_index`: available-spectrum counts and source selection indices.
- `source`, `source_url`, `spectrum_title`, and `spectrum_origin`: retrieval/source information, source links, and original spectrum descriptions.
- `chosen_phase` and `chosen_yaxis_type`: recorded sample phase and spectral ordinate type.
- `coverage`, `axis_plausible`, `converted`, `is_percent`, `x_units_in`, `x_converted`, and `x_recognized`: coverage, unit, and conversion metadata recorded during corpus preparation.

The CSV contains one record for each of the **15,829 spectra included in the final training corpus matrix**, matching the corpus size reported in the manuscript. The `source` field identifies 15,813 records as `nistchempy` and 16 as `direct`. The first, unnamed column is an exported row index; the source selection indices identify spectra within their source records, not necessarily rows of the numerical training matrix.

Users wishing to retrain must obtain the source spectra separately under the applicable access and reuse terms, or supply their own permitted spectral library. The metadata supports tracing source spectra but does not replace the numerical corpus or establish permission to redistribute it.

## Training

To train with an independently obtained spectral library:

```bash
python code/conv_cnp_afmir_prob.py \
  --spectrum_library /path/to/corpus_matrix.npy \
  --wavenumbers /path/to/wavenumbers.npy \
  --results_dir ./outputs/training
```

The spectrum library must be a NumPy array of shape `(N, 277)` for the published configuration, with **each spectrum already min–max normalized**. Loading does not perform this corpus normalization automatically. Supply a matching `(277,)` wavenumber array. Training code supports other matching channel counts, but a newly trained model is required for a different spectral grid; the supplied checkpoint is specific to the published grid.

The CLI defaults select the published training configuration. Inspect available options with:

```bash
python code/conv_cnp_afmir_prob.py --help
```

Key settings are:

- 128 × 128 synthetic cubes with 2–4 phases and dispersed/Voronoi morphologies.
- Edge-distance depth mixing with edge fraction 0–1 and decay length 0–10 pixels; zero lateral boundary width.
- Spatial texture, scratches, and Gaussian detector noise with standard deviation 0.01–0.08 in normalized units.
- Uniformly sampled context positions, with a truncated log-normal context count of 4–4000; 1–4 variance-selected structure images.
- Context z-score normalization, nearest-neighbor filling, a U-net, and cross-attention to measured spectra.
- Gaussian negative log-likelihood with context uncertainty, target MSE, structure-image gradient, whole-cube gradient, and total-variation penalties. Their additional weights are 0.1, 1.0, 0.5, 0.2, and 0.02, respectively.
- AdamW with learning rate 0.0003, weight decay 0.00001, five warmup epochs, and cosine decay; batch size 4, 2000 tasks per epoch, up to 800 epochs, and early-stopping patience 40.

The published run used an NVIDIA A100, seven data workers, 32-bit precision, and seed 42, with validation seeded separately. Worker count contributes to the synthetic-data sequence. Exact replication of the original training run additionally depends on the original numerical corpus and software/hardware environment. Loading the supplied checkpoint uses its saved model hyperparameters.

Additional functionality remains in the code, including physical thickness-based mixing, anomalous spectral pixels, phase-intensity scaling, and alternative architecture and training options. Options outside the published configuration are marked `# Not used in published deployment` where applicable and are disabled in the published defaults.


## Standalone synthetic generator

To generate one demonstration cube with artificial spectra and an overview image, without the restricted corpus:

```bash
python code/generate_synth_afmir.py --n_cubes 1 --save_png --output_dir ./outputs/synthetic_demo
```

This standalone generator uses different defaults from the published live training pipeline and does not apply its complete artifact/context-selection procedure. Use the training command above for live synthetic training.
