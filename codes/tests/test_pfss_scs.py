from pathlib import Path
import tempfile
import unittest
import numpy as np
from pfss.pfss_module import pfss_solver
from pfss.scs_module import scs_solver


class HarmonicTests(unittest.TestCase):
    def make_pfss(self, m=0):
        s = pfss_solver(None, n_r=3, n_t=8, n_p=12, Rs=2.6)
        s.Alm = {0: {0: 0.}, 1: {-1: 0., 0: 0., 1: 0.}}
        s.Blm = {0: {0: 0.}, 1: {-1: 0., 0: 0., 1: 0.}}
        s.Blm[1][m] = 1.
        s.Alm[1][m] = -s.Rs**-3
        if m:
            s.Blm[1][-m] = -1.
            s.Alm[1][-m] = s.Rs**-3
        return s

    def test_pfss_matches_potential_gradient(self):
        points = np.array([[1.3, 1.8, 2.6], [.5, 1.2, 2.1], [.3, 1.1, 2.4]])
        r, t, p = points
        for m in (0, 1):
            s = self.make_pfss(m)
            def potential(r, t, p):
                result = np.zeros_like(r, dtype=complex)
                for order in range(-1, 2):
                    result += (s.Alm[1][order]*r+s.Blm[1][order]/r**2)*({0: np.sqrt(3/(4*np.pi))*np.cos(t),
                                   1: -np.sqrt(3/(8*np.pi))*np.sin(t)*np.exp(1j*p),
                                   -1: np.sqrt(3/(8*np.pi))*np.sin(t)*np.exp(-1j*p)}[order])
                return result.real
            h = 1e-5
            expected = np.array([
                -(potential(r+h,t,p)-potential(r-h,t,p))/(2*h),
                -(potential(r,t+h,p)-potential(r,t-h,p))/(2*h*r),
                -(potential(r,t,p+h)-potential(r,t,p-h))/(2*h*r*np.sin(t)),
            ])
            np.testing.assert_allclose(s.get_Brtp(points, method='harmonics', device='cpu'), expected, rtol=1e-8, atol=1e-10)

    def test_pfss_scalar_harmonics(self):
        s = self.make_pfss()
        point = np.array([1.5, .7, .9])
        np.testing.assert_allclose(s.get_Brtp(point, method='harmonics'), s.get_Brtp(point[:,None], method='harmonics')[:,0])

    def test_load_default_npy_and_legacy_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            s = self.make_pfss()
            field = np.arange(3*3*8*12).reshape(3,3,8,12).astype(float)
            s.pfss_file = str(Path(directory)/'field.npy')
            np.save(s.pfss_file, field)
            np.testing.assert_array_equal(s.load_Brtp(), field)
            old = Path(directory)/'field.npz'
            np.savez(old, Br=field[0], Bt=field[1], Bp=field[2])
            np.testing.assert_array_equal(s.load_Brtp(load_file=old), field)

    def make_scs(self):
        s = object.__new__(scs_solver)
        s.__dict__.update(self.make_pfss().__dict__)
        s.Rcp = 2.4
        s.Rtp = 5.
        s.lmax_scs = 0
        s.glm = {0: {0: 1.}}
        s.hlm = {0: {0: 0.}}
        s.mask = np.zeros((8,12), dtype=bool)
        s.Nrtp_scs = [3,8,12]
        s.scs_file = 'Brtp_scs.npy'
        return s

    def test_scs_split_and_smooth_blend(self):
        s = self.make_scs()
        points = np.array([[2.3, 2.5, 2.6, 2.7, 3.], [.7]*5, [.9]*5])
        pfss = pfss_solver.get_Brtp(s, points, method='harmonics')
        outer = np.zeros_like(points)
        outer[0] = (s.Rcp/points[0])**2
        sharp = s.get_Brtp(points, method='harmonics')
        expected = np.where((points[0]<=s.Rs)[None,:], pfss, outer)
        np.testing.assert_allclose(sharp, expected)
        blended = s.get_Brtp(points, method='harmonics', interface_blend_half_width=.1)
        expected[:,2] = .5*(pfss[:,2]+outer[:,2])
        np.testing.assert_allclose(blended, expected, atol=1e-14)
        shifted = s.get_Brtp(points, method='harmonics', sc_split_radius=2.4)
        np.testing.assert_allclose(shifted[:,1:], outer[:,1:])
        for i in range(points.shape[1]):
            np.testing.assert_allclose(s.get_Brtp(points[:,i], method='harmonics', interface_blend_half_width=.1), blended[:,i])

    def test_polarity_lookup_at_cell_centers_and_periodic_phi(self):
        s = self.make_scs()
        s.mask[2,3] = True
        theta = np.pi-(2+.5)*np.pi/8
        phi = (3+.5)*2*np.pi/12
        for angle in (phi, phi+2*np.pi, phi-2*np.pi):
            result = s.get_Brtp(np.array([3.,theta,angle]), method='harmonics')
            self.assertLess(result[0], 0.)
        self.assertGreater(s.get_Brtp(np.array([3.,theta,phi+2*np.pi/12]), method='harmonics')[0], 0.)

    def test_cusp_degree_and_input_not_modified(self):
        cusp = np.zeros((3,1,8,12))
        cusp[0] = -2.
        saved = cusp.copy()
        s = scs_solver(None, n_r=3, n_t=8, n_p=12, lmax_scs=1,
                       Nrtp_scs=[3,8,12], Brtp_cusp=cusp, device='cpu')
        np.testing.assert_array_equal(cusp, saved)
        np.testing.assert_allclose(s.glm[0][0], 2., rtol=1e-12)
        np.testing.assert_allclose(s.get_Brtp(np.array([3.,.7,.9]), method='harmonics'), [-2*(s.Rcp/3)**2,0.,0.], atol=1e-12)

    def test_scs_sine_mode_fit_matches_evaluation(self):
        from pfss.scs_module import build_SCS_Brtp
        t = np.pi-(np.arange(8)+.5)*np.pi/8
        p = (np.arange(12)+.5)*2*np.pi/12
        r, t, p = np.meshgrid([2.4], t, p, indexing='ij')
        g = {0: {0: 3.}, 1: {0: .2, 1: .1}}
        h = {0: {0: 0.}, 1: {0: 0., 1: .4}}
        cusp = build_SCS_Brtp(r,t,p,g,h,lmax=1,Rcp=2.4,device='cpu')
        s = scs_solver(None, lmax_scs=1, Nrtp_scs=[3,8,12], Brtp_cusp=cusp, device='cpu')
        for l in (0,1):
            for m in range(l+1):
                self.assertAlmostEqual(s.glm[l][m], g[l][m], places=12)
                self.assertAlmostEqual(s.hlm[l][m], h[l][m], places=12)

    def test_constructor_retains_explicit_pfss_coefficients(self):
        ps = self.make_pfss()
        s = scs_solver(None, n_r=3, n_t=8, n_p=12, lmax_scs=1,
                       Nrtp_scs=[3,8,12], Alm=ps.Alm, Blm=ps.Blm, device='cpu')
        point = np.array([[1.5],[.7],[.9]])
        np.testing.assert_allclose(s.get_Brtp(point, method='harmonics'), ps.get_Brtp(point, method='harmonics'))

    def test_scs_generated_grid_matches_harmonics(self):
        s = self.make_scs()
        with tempfile.TemporaryDirectory() as directory:
            grid = s.get_scs(fname=str(Path(directory)/'scs.npy'), device='cpu')
        expected = s.get_Brtp(s.get_rtp(), method='harmonics', sc_split_radius=s.Rcp-1e-6)
        np.testing.assert_allclose(grid, expected, atol=1e-14)

    def test_cusp_interpolation_loads_pfss_not_scs_file(self):
        with tempfile.TemporaryDirectory() as directory:
            field = np.zeros((3,3,8,12))
            field[0] = 2.
            path = Path(directory)/'pfss.npy'
            np.save(path, field)
            s = scs_solver(None, n_r=3, n_t=8, n_p=12, lmax_scs=0,
                           Nrtp_scs=[3,8,12], cusp_method='interpolation', load_file=path)
            self.assertAlmostEqual(s.glm[0][0], 2., places=12)
