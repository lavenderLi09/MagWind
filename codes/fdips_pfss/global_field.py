"""CPU PFSS on a full Carrington shell. Fields in gauss, angles in radians."""
import numpy as np
from astropy.io import fits
from scipy.special import sph_harm_y

from .solver import staggered_solve


def _overlap(source, target):
    width = np.minimum(target[1:, None], source[1:]) - np.maximum(target[:-1, None], source[:-1])
    return width.clip(min=0) / np.diff(target)[:, None]


def read_boundary(path, faces, nphi):
    """Conservatively remap a complete, axis-aligned 2D Carrington Br map."""
    data, h = fits.getdata(path, header=True)
    if not np.isfinite(data).all() or np.any(data == h.get("MISSING", np.nan)):
        raise ValueError("The magnetogram contains missing pixels.")
    axes = (h.get("CTYPE1"), h.get("CTYPE2"))
    if (h.get("MODEL"), h.get("LATTYPE"), axes) == ("ADAPT", 0, ("Long", "Lat")):
        edge = h["CRVAL1"] + (0.5 - h["CRPIX1"]) * h["CDELT1"]
        if not np.isclose((edge - h["CRLNGEDG"] + 180) % 360 - 180, 0):
            raise ValueError("ADAPT longitude reference differs from its Carrington edge.")
        if data.ndim == 3:
            data = data.mean(axis=0, dtype=float)
        axes = ("CRLN-CAR", "CRLT-CAR")
    if data.ndim != 2:
        raise ValueError("Supply one complete finite 2D radial-field map.")
    if str(h.get("BUNIT", "")).lower() not in {"g", "gauss", "mx/cm^2", "mx/cm2"}:
        raise ValueError("Expected a radial field in gauss or Mx/cm^2.")
    ny, nx = data.shape
    if axes not in {
        ("CRLN-CEA", "CRLT-CEA"), ("CRLN-CAR", "CRLT-CAR")
    } or h.get("CRVAL2") != 0:
        raise ValueError("Expected equator-referenced Carrington CEA or CAR axes.")
    if any(k.startswith(("CD1_", "CD2_")) for k in h) or any(
        h.get(k, default) != default for k, default in
        {"PC1_1": 1, "PC2_2": 1, "PC1_2": 0, "PC2_1": 0, "CROTA2": 0, "PV2_1": 1}.items()
    ):
        raise ValueError("Expected an unrotated, unscaled map.")
    degrees = {"deg", "degree", "degrees"}
    if h["CUNIT1"].lower() not in degrees or not np.isclose(nx * abs(h["CDELT1"]), 360):
        raise ValueError("Expected a full 360-degree longitude axis.")
    lon = h["CRVAL1"] + (np.arange(nx + 1) + 0.5 - h["CRPIX1"]) * h["CDELT1"]
    lat = (np.arange(ny + 1) + 0.5 - h["CRPIX2"]) * h["CDELT2"]
    unit = h["CUNIT2"].lower()
    if axes[1].endswith("CEA") and unit in {"sine latitude", "sin(latitude)", "1"}:
        mu = lat
    elif unit in degrees:
        if axes[1].endswith("CAR") and not np.allclose(sorted(lat[[0, -1]]), [-90, 90]):
            raise ValueError("Expected a pole-to-pole latitude axis.")
        mu = np.deg2rad(lat)
        if axes[1].endswith("CAR"):
            mu = np.sin(mu)
    else:
        raise ValueError("Unsupported latitude axis units.")
    if not np.allclose(sorted(mu[[0, -1]]), [-1, 1], atol=1e-5):
        raise ValueError("Expected full pole-to-pole coverage.")
    if mu[0] > mu[-1]:
        mu, data = mu[::-1], data[::-1]
    if lon[0] > lon[-1]:
        lon, data = lon[::-1], data[:, ::-1]
    mu[[0, -1]] = [-1, 1]
    edges = np.linspace(lon[0], lon[-1], nphi + 1)
    weights = _overlap(mu, np.cos(faces)[::-1])[::-1]
    bottom = weights @ data @ _overlap(lon, edges).T
    mean = np.average(bottom.mean(axis=1), weights=-np.diff(np.cos(faces)))
    phi = np.deg2rad(lon[0] % 360 + (np.arange(nphi) + 0.5) * 360 / nphi)
    return bottom - mean, phi, mean


def _harmonic(bottom, r, theta, phi, lmax):
    """MagWind midpoint harmonic projection and radial factors, evaluated on CPU."""
    nt, np_ = bottom.shape
    if not isinstance(lmax, (int, np.integer)) or not 1 <= lmax <= min(nt - 1, (np_ - 1) // 2):
        raise ValueError("lmax must be positive and resolved by the angular mesh.")
    fields = np.zeros((3, len(r), nt, np_), dtype=float)
    solid_angle = np.sin(theta) * (np.pi / nt) * (2 * np.pi / np_)
    radius = r[:, None, None]
    for m in range(lmax + 1):
        phase = np.exp(1j * m * phi)
        projection = (bottom @ phase.conj()) * solid_angle
        for l in range(max(1, m), lmax + 1):
            y, dy = sph_harm_y(l, m, theta, 0, diff_n=1)
            coef = (2 if m else 1) * np.sum(projection * y.conj())
            q = (radius / r[-1]) ** (2*l + 1)
            scale = radius ** (-l - 2) / (l + 1 + l * r[-1] ** (-2*l - 1))
            angular = coef * y[:, None] * phase
            fields[0] += scale * (l + 1 + l*q) * angular.real
            tangent = scale * (q - 1)
            fields[1] += tangent * (coef * dy[:, 0, None] * phase).real
            fields[2] += tangent * (1j * m * angular / np.sin(theta)[:, None]).real
    return fields


def solve(path, method="harmonic", R_ss=2.5, lmax=12, grid=(16, 48, 96)):
    """Return field components and their native coordinates as an NPZ-ready dict."""
    nr, nt, nphi = grid
    if not np.isfinite(R_ss) or R_ss <= 1 or min(grid) < 2:
        raise ValueError("Use R_ss > 1 and at least two cells in each direction.")
    r = np.linspace(1, R_ss, nr + 1)
    faces = np.linspace(0, np.pi, nt + 1)
    theta = (faces[:-1] + faces[1:]) / 2
    bottom, phi, mean = read_boundary(path, faces, nphi)
    out = dict(method=method, r=r, theta=theta, phi=phi, bottom=bottom, removed_mean_G=mean)
    if method == "harmonic":
        br, bt, bp = _harmonic(bottom, r, theta, phi, lmax)
        out.update(lmax=lmax)
    elif method == "fdips":
        _, br, bt, bp, diagnostics = staggered_solve(r, faces, theta, bottom)
        out.update(r_center=(r[1:] + r[:-1])/2, theta_face=faces,
                   phi_face=phi + np.pi/nphi, **diagnostics)
    else:
        raise ValueError("Choose 'harmonic' or 'fdips'.")
    return dict(out, Br=br, Btheta=bt, Bphi=bp)
