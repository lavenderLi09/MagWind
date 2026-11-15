"""Finite-difference PFSS on full spherical shells with periodic longitude."""

import numpy as np
from scipy.linalg import eigh_tridiagonal


def staggered_solve(r, faces, theta, bottom):
    """Fourier in longitude, area-weighted eigenbasis in theta, Thomas in r."""
    nt, nphi = bottom.shape
    dp = 2*np.pi/nphi
    dr = np.diff(r)
    centers = (r[1:]+r[:-1])/2
    mu = -np.diff(np.cos(faces))
    weight = np.sqrt(mu)
    theta_link = np.sin(faces[1:-1])/np.diff(theta)
    theta_diag = (np.r_[0, theta_link]+np.r_[theta_link, 0])/mu
    theta_off = -theta_link/(weight[:-1]*weight[1:])
    phi_metric = np.diff(faces)/(mu*np.sin(theta))
    radial_link = r[1:-1]**2/np.diff(centers)
    radial_diag = np.r_[0, radial_link]+np.r_[radial_link, r[-1]**2/(dr[-1]/2)]
    modes = np.fft.rfft(bottom, axis=1)
    potential_modes = np.empty((len(dr), nt, modes.shape[1]), dtype=complex)
    for m in range(modes.shape[1]):
        wave = 4*(np.sin(np.pi*m/nphi)/dp)**2
        eigenvalues, vectors = eigh_tridiagonal(theta_diag+wave*phi_metric, theta_off)
        diagonal = radial_diag[:, None]+dr[:, None]*eigenvalues
        solution = np.zeros(diagonal.shape, dtype=complex)
        solution[0] = -r[0]**2*(vectors.T @ (weight*modes[:, m]))
        for i, link in enumerate(radial_link, 1):
            factor = link/diagonal[i-1]
            diagonal[i] -= factor*link
            solution[i] += factor*solution[i-1]
        solution[-1] /= diagonal[-1]
        for i in range(len(dr)-2, -1, -1):
            solution[i] = (solution[i]+radial_link[i]*solution[i+1])/diagonal[i]
        potential_modes[:, :, m] = (solution @ vectors.T)/weight
    potential = np.fft.irfft(potential_modes, n=nphi, axis=2)
    br = np.concatenate([bottom[None], np.diff(potential, axis=0)/
                         np.diff(centers)[:, None, None],
                         -potential[-1:]/(dr[-1]/2)])
    bt = np.zeros((len(dr), nt+1, nphi))
    bt[:, 1:-1] = np.diff(potential, axis=1)/(centers[:, None, None]*
                                           np.diff(theta)[None, :, None])
    bp = (np.roll(potential, -1, axis=2)-potential)/(centers[:, None, None]*
                                                     np.sin(theta)[None, :, None]*dp)
    radial_flux = np.diff(r[:, None, None]**2*br, axis=0)*mu[None, :, None]*dp
    sin_faces = np.sin(faces)
    sin_faces[[0, -1]] = 0
    theta_flux = (centers*dr)[:, None, None]*dp*np.diff(sin_faces[None, :, None]*bt, axis=1)
    phi_flux = (centers*dr)[:, None, None]*np.diff(faces)[None, :, None]*(bp-np.roll(bp, 1, axis=2))
    residual = radial_flux+theta_flux+phi_flux
    volumes = np.diff(r**3)[:, None, None]/3*mu[None, :, None]*dp
    bottom_flux = r[0]**2*bottom*mu[:, None]*dp
    check = dict(relative_flux_residual_L2=float(np.linalg.norm(residual.ravel())/
                                                  np.linalg.norm(bottom_flux.ravel())),
                 max_abs_div_G_per_Rsun=float(np.max(abs(residual/volumes))))
    assert check["relative_flux_residual_L2"] < 1e-9, check
    return potential, br, bt, bp, check


def dipole_check():
    r = np.linspace(1, 2, 65)
    faces = np.linspace(0, np.pi, 65)
    theta = (faces[1:]+faces[:-1])/2
    # Cell-average of cos(theta) over each face.
    bottom = np.broadcast_to(((np.cos(faces[1:])+np.cos(faces[:-1]))/2)[:, None], (64, 128))
    _, br, _, _, check = staggered_solve(r, faces, theta, bottom)
    # Phi proportional to (r - 8/r**2)*cos(theta), normalized at r=1.
    exact = (1+16/r**3)[:, None, None]*bottom/17
    check["Br_relative_L2"] = float(np.linalg.norm((br-exact).ravel())/np.linalg.norm(exact.ravel()))
    assert check["Br_relative_L2"] < 0.003, check
    return check


if __name__ == "__main__":
    print(dipole_check())
