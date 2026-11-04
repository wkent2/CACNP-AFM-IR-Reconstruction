Description of Nano-IR (AFM-IR) Hyperspectral Data

Data: 
 - Two hyperspectral cube reconstructions (Single- and Multi-particle) of different regions of the same PS-PMMA-Epoxy standard sample (Bruker).
 - The hyperspectral cube, wavenumbers, topography image, amplitude error, phase image, for each dataset have been saved as .npy arrays (NumPy arrays that can be loaded in Python using np.load(...)). 
 - The single- and multi-particle datasets are 1.5 x 1.5 µm and 20.0 x 20.0 µm, respectively. 
 - Both datasets were acquired on a Bruker Dimension IconIR System. 
 - The single- and multi-particle datasets took approximately 5 and 20 hours to reconstruct, respectively. 


                                      Single-Particle Cube Data
|----------------------------------------------------------------------------------------------|
| Filename                           | Contents           | Shape                     |  Units |
|----------------------------------------------------------------------------------------------|
| Single_PSPMMAEXPOXY_amp_err.npy    | Hyperspectral cube | (X,Y,W) -> (64,64,277)    |  mV    |
| Single_PSPMMAEXPOXY_topography.npy | Topography image   | (X,Y)   -> (64,64)        |  nm    |
| Single_PSPMMAEXPOXY_amp_err.npy    | Aplitude error img | (X,Y)   -> (64,64)        |  nm    |
| Single_PSPMMAEXPOXY_phase.npy      | Phase image        | (X,Y)   -> (64,64)        |  mV    |
| Single_PSPMMAEXPOXY_wavenum.npy    | Wavenumbers        | (W)     -> (277)          |  cm^-1 |
|----------------------------------------------------------------------------------------------|


                                      Multi-Particle Cube Data
|----------------------------------------------------------------------------------------------|
| Filename                           | Contents           | Shape                     |  Units |
|----------------------------------------------------------------------------------------------|
| Multi_PSPMMAEXPOXY_amp_err.npy     | Hyperspectral cube | (X,Y,W) -> (128,128,277)  |  mV    |
| Multi_PSPMMAEXPOXY_topography.npy  | Topography image   | (X,Y)   -> (128,128)      |  nm    |
| Multi_PSPMMAEXPOXY_amp_err.npy     | Aplitude error img | (X,Y)   -> (128,128)      |  nm    |
| Multi_PSPMMAEXPOXY_phase.npy       | Phase image        | (X,Y)   -> (128,128)      |  mV    |
| Mulit_PSPMMAEXPOXY_wavenum.npy     | Wavenumbers        | (W)     -> (277)          |  cm^-1 |
|----------------------------------------------------------------------------------------------|

