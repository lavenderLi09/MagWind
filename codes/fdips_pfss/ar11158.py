"""Reproduce the AR11158 GONG–ADAPT PFSS background and MF input table."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits

from solver import staggered_solve


INPUT_SHA256 = "53cf330567b4a78e4ab3650f95a53a5bcc80d07ff0631ac27c00e5936e48edb1"


def overlap_weights(edges, target):
    overlap = np.minimum(target[1:, None], edges[1:]) - np.maximum(target[:-1, None], edges[:-1])
    return np.maximum(0, overlap) / np.diff(target)[:, None]


def photospheric_field(path, faces, nphi):
    if hashlib.sha256(path.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("Supply the archived 2011-02-10 02:00 UT GONG–ADAPT FITS file.")
    ensemble, header = fits.getdata(path, header=True)
    assert ensemble.shape == (12, 180, 360) and header["BUNIT"] == "GAUSS"
    assert np.isfinite(ensemble).all() and not (ensemble == header["MISSING"]).any()
    raw = ensemble.astype(float).mean(axis=0)
    mu = np.sin(np.deg2rad(np.arange(-90, 91)))
    weights = overlap_weights(mu, np.cos(faces)[::-1])[::-1]
    lon_weights = overlap_weights(np.arange(361), np.linspace(0, 360, nphi+1))
    mean = float(np.sum(raw.mean(axis=1)*np.diff(mu))/2)
    bottom = weights @ raw @ lon_weights.T - mean
    area = -np.diff(np.cos(faces))[:, None]*2*np.pi/nphi
    assert abs(np.sum(bottom*area)) < 1e-9
    return bottom, mean


def export_field(directory, p, r, theta, phi, bottom):
    # Reconstruct a common scalar potential at the radial table nodes.
    center = (r[:-1]+r[1:])/2
    potential = np.empty((len(r), len(theta), len(phi)))
    w = ((r[1:-1]-center[:-1])/np.diff(center))[:, None, None]
    potential[1:-1] = (1-w)*p[:-1]+w*p[1:]
    potential[0] = p[0]-(center[0]-r[0])*bottom
    potential[-1] = 0

    binary = directory/"off.bin"
    shape = (3, *potential.shape)
    with binary.open("xb") as stream:
        stream.write(np.array(shape, "<i4").tobytes())
    b = np.memmap(binary, "<f8", mode="r+", offset=16, shape=shape, order="F")
    b[0] = np.gradient(potential, r, axis=0, edge_order=2)
    b[0, 0] = bottom
    b[1] = np.gradient(potential, theta, axis=1, edge_order=2)/r[:, None, None]
    b[2] = (np.roll(potential, -1, axis=2)-np.roll(potential, 1, axis=2))/(
        2*(phi[1]-phi[0])*r[:, None, None]*np.sin(theta)[None, :, None])
    b.flush()
    assert np.isfinite(b).all() and not b[1:, -1].any()
    assert np.array_equal(b[0, 0], bottom)
    (directory/"coords.bin").write_bytes(np.concatenate([r, theta, phi]).astype("<f8").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits", type=Path, help="Original compressed GONG–ADAPT magnetogram")
    parser.add_argument("output", type=Path, help="New output directory under an existing parent")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    with np.load(Path(__file__).with_name("ar11158_grid.npz")) as grid:
        r, faces, theta, phi = (grid[key] for key in ("r", "theta_faces", "theta", "phi"))
    bottom, mean = photospheric_field(args.fits, faces, len(phi))
    potential, br, bt, bp, diagnostics = staggered_solve(r, faces, theta, bottom)
    del br, bt, bp
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        directory = Path(tmp)/"field"
        directory.mkdir()
        np.savez(directory/"staggered_potential.npz", potential=potential, Br_inner=bottom,
                 r_faces=r, theta=theta, theta_faces=faces, phi=phi)
        export_field(directory, potential, r, theta, phi, bottom)
        diagnostics["removed_mean_G"] = mean
        (directory/"verification.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
        directory.rename(args.output)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
