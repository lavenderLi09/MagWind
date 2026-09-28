# Magnetic calculation update (September 2026)

These changes bring the public solver closer to the calculations validated for the August 22, 2024 magnetic initialization, and add the newer PFSS/SCS interface options. Generated magnetic fields must be regenerated to use the corrections; updating the Python package does not modify existing NPY, BIN or VTK files.

## OFF

The wind uses the transonic Lambert-W solution, with branch 0 below the critical radius and branch -1 above it. `v1` is the prescribed speed in km/s at `Rs`; it controls the magnetic outflow model and is separate from any MHD initial-atmosphere prescription. Exact zero speed is outside the validated domain of this update. A local series supplies the regular sonic-point limits: `v=cs`, `dv/d(rho)=cs`, and `d2v/d(rho)2=0`.

With `rho = log(r)`, `w = W(-D)` and `v = cs*sqrt(-w)`, the derivative is

```text
dv/d(rho) = v/2 * D'/D / (1+w)
```

The radial equation now satisfies `G' + 2G = l(l+1)H`, where `G = H' + (1 - nu0*r*v)H`. The retained monopole uses `H0=0`, `G0=1/r^2`, conserving signed magnetic flux. Solar magnetograms should still be flux-balanced before use.

Backward RK4 applies the signed step to all intermediate states. Integration respects the iteration limit, evaluates the last inward step from its actual starting point, and supports an optional outward `rmax` with exact endpoints. `d2_vout` differentiates the same wind as `vout` and `d1_vout`. The shooting method keeps its existing `(log_r, solution)` return format.

OFF worker runs reject a pre-existing temporary directory, failed child processes, missing or unexpected partitions, and nonfinite or incorrectly shaped fields. The lookup-table interpolation and experimental latitude-dependent wind modes in local research copies are not part of this update.

## PFSS

Harmonic evaluation counts `m=0` once and includes the spherical `1/sin(theta)` factor in `Bphi`. Scalar and array queries use the same calculation. The shared harmonics and SCS helpers use Python's `math.factorial` instead of the removed NumPy alias.

The direct worker wrapper saves `(Br, Btheta, Bphi)` consistently, honors `pfss_file`, accepts a supplied boundary, and checks child exit status. The multi-device wrapper forwards the requested mesh and source radius and handles the single-worker fast-mode output. Fast harmonic evaluation can run on CPU for verification. The default loader reads the solver's NPY output; explicitly supplied legacy NPZ files remain supported.

## SCS interface

Saved SCS grids and direct harmonic evaluation use the same `Rcp` normalization. The cusp fit uses `lmax_scs`, honors supplied cusp data without mutating it, and uses the same sine-mode azimuthal sign as field evaluation. By default, it samples PFSS harmonics at the cusp rather than mixing interpolation and harmonic representations. `cusp_method='interpolation'` selects the old sampling method. Explicit `Alm` and `Blm` may be supplied when constructing the solver.

```python
scs = scs_solver(
    None, Rs=2.5, Rcp=2.4, Rtp=20.,
    lmax_scs=10, Nrtp_scs=[100, 90, 180],
    Alm=pfss.Alm, Blm=pfss.Blm,
)
field = scs.get_Brtp(
    points,                         # shape (3, ...) or one (r, theta, phi)
    method='harmonics',
    sc_split_radius=2.5,
    interface_blend_half_width=0.0,  # default: no blend
)
```

The default split is now `Rs`: PFSS is used for `r <= Rs`, SCS for `r > Rs`. The SCS radial expansion remains referenced to `Rcp`. Set `sc_split_radius=scs.Rcp` to split at the cusp instead. This changes the default behavior in the shell between `Rcp` and `Rs`; it should be considered when comparing old and regenerated fields.

A positive `interface_blend_half_width` enables a cubic smoothstep mixture of PFSS and SCS **for harmonic evaluation only**. Its units are solar radii, like the split radius. It is an optional interface treatment, not a solenoidal construction: the spatially varying weight can introduce `grad(w) dot (B_pfss-B_scs)` into the divergence. Assess divergence and flux explicitly before using a blended field for MHD. The blend should stay within the intended validity ranges of both models.

Polarity restoration uses the cusp cell's nearest angular sample, with periodic longitude. It does not replace field-line tracing of polarity through the outer volume.

## Reproducible validation

From the repository root, in an environment with the package dependencies:

```bash
python -m pip install -e ./codes
PYTHONPATH="$PWD/codes" python -m unittest discover -s codes/tests -v
```

Tests use small synthetic fields and CPU workers; no magnetogram download, CUDA device or production dataset is needed. They check finite-difference wind derivatives, the transonic equation, signed RK4, an independent SciPy radial integrator, monopole flux, analytic PFSS potential gradients, SCS coefficient recovery/polarity/split/blend behavior, saved formats and actual worker processes.

The PFSS worker tests supply analytic coefficient files. They do not validate the legacy multiprocessing coefficient-generation path on macOS/Windows, full-resolution GPU calculations, a full magnetogram-to-MHD pipeline, or MHD evolution stability. No production magnetic files are included.

Verified locally on September 28, 2026: **26 tests passed**, using Python 3.13.7, NumPy 2.3.3, SciPy 1.16.2 and PyTorch 2.9.1. The install requirements now declare the existing Plotly/scikit-image imports and constrain SciPy below 1.17 while the legacy `sph_harm` API remains in use.
