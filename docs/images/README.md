# README figure provenance

The two August-case PNGs are original diagnostic figures from the September 27, 2026 validation of the August 22, 2024 OFF magnetic initialization. They are copied unchanged from that research case. They were not regenerated with the public repository's September 28 update or with the synthetic README examples.

## Magnetic maps

File: `off-20240822-surface-maps.png`.

- Input record: `hmi.mrdailysynframe_polfil_720s[2024.08.22_12:00:00_TAI]`.
- Input FITS SHA-256: `3804eee4b4f1a8dd59ededaa787abf4128cc75e37a4cc816d387dbd1cdf6155e`.
- Area-weighted flux balancing; OFF `lmax=10`, `v1=0.001 km/s`, source sphere `2.6 R_sun`.
- Outside the source sphere: `Br(r)=Br(2.6)*(2.6/r)^2`, with `Btheta=Bphi=0`. No Parker spiral or solar rotation was applied.
- Saved domain: `1.8–20 R_sun`, on a `600 × 180 × 360` grid. First and last radial cell centers are approximately `1.803878` and `19.961224 R_sun`.
- Native simulation longitude is retained; it should not be relabeled Carrington longitude.
- The upper maps show `Br` in Gauss, on different scales. The lower maps show `r²Br` on the same scale. The photospheric map clips colors at the 99th percentile; the figure states that limit.

## Divergence comparison

File: `off-derivative-correction.png`.

The old and corrected fields share the source magnetogram, flux balancing, mesh, harmonic truncation, source radius and outer radial extension. The comparison isolates the `d1_vout` correction after the backward-RK4 fix had already been applied. It is not a comparison against every historical public MagWind version.

The shell diagnostic is `epsilon = r × RMS(div B) / RMS(|B|)`. RMS values use spherical cell-area weights. The derivative uses five-point polynomial stencils on the nonuniform radial/theta grid and a periodic fourth-order longitude stencil; theta indices `2:-2` and all longitudes are included. Each map uses its own field's shell RMS magnitude as the denominator, not the local pointwise magnitude.

| Radius (R_sun) | Previous epsilon | Corrected epsilon | Reduction |
| ---: | ---: | ---: | ---: |
| 1.998860922 | 2.86682331e-5 | 4.76568814e-6 | 6.02× |
| 2.298098757 | 2.12697242e-4 | 2.22973551e-6 | 95.39× |
| 2.498817884 | 2.85289163e-4 | 7.50391947e-7 | 380.19× |

These are initialization diagnostics, not the AMRVAC discrete divergence operator or a validation of MHD evolution. The residual is not asserted to be zero. Full production field arrays and case-specific generation scripts are not bundled here; the repository's small CPU regression suite tests the solver independently.

## Figure checksums

- `off-20240822-surface-maps.png`: `d44050d2b48eea3ab77742d143ee34077afc8040d8fc6d3f13051182a26c9010`
- `off-derivative-correction.png`: `9f22ffcaecb7a2ea4fea82350906492446632a419e72742dccacd4dfb5c0a151`

## Global field gallery

The following assets were supplied for the README gallery and copied byte-for-byte. No frames, colors, parameter labels, or image content were changed. The source filenames are `maglines_trans.gif`, `off_results.gif`, and a clipboard PNG now named `global-field-overview.png`.

These gallery assets are separate from the August-case diagnostics above. The source magnetogram, epoch, field-line color normalization, and generating solver revision are not independently documented here. Labels within `off_results.gif` are preserved from the supplied visualization; they are not a claim that all displayed model options are implemented in the current public API.

| File | Dimensions | Frames | Size | SHA-256 |
| --- | --- | ---: | ---: | --- |
| `maglines_trans.gif` | 1658 × 1642 | 20 | 7.48 MiB | `ac84ebafed7b04ad5cd9eee91dc7f1dc5bedf69e91bd9a8ea73da4734219af6c` |
| `off_results.gif` | 4056 × 3156 | 20 | 34.92 MiB | `953b90215831ff8e73c68f4dfb19feb121978104e3c6b936170597290a1cadf8` |
| `global-field-overview.png` | 338 × 224 | 1 | 0.09 MiB | `5d4ffde10a96a0cf6aad2817c2a9620991d927d60d6a5d1788423e7fc02cb10c` |
