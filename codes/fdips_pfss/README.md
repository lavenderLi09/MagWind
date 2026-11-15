# Finite-difference PFSS for the AR11158 background

This directory contains the Python solver used to prepare the potential-field source-surface (PFSS) background for the AR11158 magnetofrictional (MF) simulation. The numerical kernel and dipole check in [solver.py](solver.py) come directly from the production preparation script. [ar11158.py](ar11158.py) packages its GONG–ADAPT boundary preparation and common-potential export into a standalone example. [ar11158_grid.npz](ar11158_grid.npz) contains the four coordinate arrays from the original grid.

## Method

We solve Laplace's equation with **B = ∇Φ**, prescribe the radial field at the photosphere, and set Φ to zero at the source surface. The potential lies at cell centers and the normal magnetic components lie on cell faces. The discretization follows Tóth et al. Their FDIPS software uses ILU-preconditioned BiCGSTAB. This Python implementation Fourier-transforms the boundary data in periodic longitude, diagonalizes the area-weighted colatitude operator for each Fourier mode, and solves the resulting radial tridiagonal systems directly. [Tóth, van der Holst & Huang (2011)](https://doi.org/10.1088/0004-637X/732/2/102)

For AR11158 we average all 12 GONG–ADAPT realizations at 2011-02-10 02:00 UT in archive gauss. We remap the mean by cell overlap in sine latitude and longitude and subtract its area-weighted mean. Prescribing the resulting radial field directly preserves the flux assigned to each photospheric grid face and avoids the ringing associated with truncated spherical-harmonic reconstruction. The flux agreement applies to the remapped, balanced map. [Comparison in Figure 8 of Tóth et al.](https://doi.org/10.1088/0004-637X/732/2/102)

The PFSS grid has 201 radial cells from 1 to 2 solar radii, 240 colatitude cells and 480 longitude cells. The MF simulation stores its field at cell centers. To initialize it, the exporter reconstructs the scalar potential at the 202 radial table nodes and differentiates that common potential for all three components. It preserves the prescribed photospheric radial field exactly and gives zero transverse field at the source surface. The MF reader interpolates this table onto its own mesh. The export initialized the global relaxation from which the later driven run 365965 inherited its restart.

## Reproduce the calculation

Run from this directory in a Python environment with NumPy, SciPy, and Astropy. The reference run used Python 3.12, NumPy 2.2.5, SciPy 1.15.3, and Astropy 7.0.2. Allow several GiB of RAM and about 0.75 GB of disk space for the full example. The dipole check takes no input files.

```sh
python -m pip install numpy==2.2.5 scipy==1.15.3 astropy==7.0.2
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python solver.py

map=adapt40311_044012_201102100200_i00013600n1.fts.gz
curl -fLO "https://gong.nso.edu/adapt/maps/gong/2011/$map"
python ar11158.py "$map" ar11158_output
```

Supply a new output directory name. The example checks the compressed FITS file's SHA-256 before reading it. The grid arrays use solar radii for radius and radians for angles. Longitude increases periodically and colatitude increases from north to south.

| Output | Contents |
| --- | --- |
| `staggered_potential.npz` | Cell-centered potential, prescribed inner-face radial field, radial faces, colatitude centers and faces, and longitude centers. |
| `off.bin` | MF input table in gauss. Four little-endian int32 values give the shape (3, 202, 240, 480), followed by little-endian float64 values in Fortran order. Axes are component (radial, colatitudinal, longitudinal), radius, colatitude and longitude. |
| `coords.bin` | Concatenated radius, colatitude and longitude table nodes in little-endian float64. |
| `verification.json` | Removed mean field, relative cell flux-balance residual and maximum absolute divergence on the staggered PFSS grid. |

## Check the result

The axial dipole check compares the computed radial field with the analytic solution. The reference relative L2 error is 7.946e-5. For AR11158 the subtracted mean is −0.00514922261 G and the relative L2 flux-balance residual is 2.0950e-13. That residual is computed from the staggered face fluxes before exporting the collocated MF input.

The full example reproduced these archived SHA-256 values in the reference environment. Floating-point results can vary across numerical libraries and architectures.

| File | SHA-256 |
| --- | --- |
| GONG–ADAPT compressed FITS | `53cf330567b4a78e4ab3650f95a53a5bcc80d07ff0631ac27c00e5936e48edb1` |
| `staggered_potential.npz` | `306b9c1de696f65915975db21f09f8f2450c79620a54cf7992a9a12678390432` |
| `off.bin` | `88d37f2072ad17a4680273dbc3baea7b39d8daf3bd680c38e5579288f99e7c2f` |
| `coords.bin` | `f361fd4eeec7b5e44e0482d4223183d0db6b47fc1e3ba69fbf8616b18c136fed` |
