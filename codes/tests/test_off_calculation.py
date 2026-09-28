import unittest
import numpy as np
from scipy.integrate import solve_ivp
from pfss.off_module import off_solver, rk45_solver, nu0, v0


def derivative(function, x):
    h = 1e-5
    return (-function(x+2*h)+8*function(x+h)-8*function(x-h)+function(x-2*h))/(12*h)


class OffCalculationTests(unittest.TestCase):
    def setUp(self):
        self.solver = off_solver(None, n_r=3, n_t=8, n_p=12, Rs=2.6, v1=.001, lmax=3)

    def test_wind_matches_prescribed_speed(self):
        for speed in (.001, 10., 50., 200.):
            solver = off_solver(None, Rs=2.6, v1=speed)
            self.assertAlmostEqual(solver.vout(np.log(solver.Rs))*v0/1e5, speed, places=8)

    def test_transonic_wind_equation(self):
        s = self.solver
        r = s.rc*np.array([.75, 1.25, 2.])
        cs = np.sqrt(9.54e4/(2*s.rc))*1e5/v0
        u = (s.vout(np.log(r))/cs)**2
        np.testing.assert_allclose(u-np.log(u), 4*np.log(r/s.rc)+4*s.rc/r-3, rtol=1e-10)

    def test_derivative_both_branches(self):
        s = self.solver
        r = np.r_[np.linspace(1., 2.6, 8), s.rc*np.array([.75, 1.25, 2.])]
        np.testing.assert_allclose(s.d1_vout(np.log(r)), derivative(s.vout, np.log(r)), rtol=5e-8, atol=1e-25)

    def test_second_derivative(self):
        s = self.solver
        rho = np.log(np.array([1.8, 2.6, .75*s.rc, 1.25*s.rc]))
        np.testing.assert_allclose(s.d2_vout(rho), derivative(s.d1_vout, rho), rtol=5e-7, atol=1e-25)

    def test_solenoidal_radial_equation(self):
        s = self.solver
        for r in (2., 2.3, 2.6):
            x = np.log(r)
            v, dv = s.vout(x), derivative(s.vout, x)
            for l in (1, 3, 10):
                h, dh = .7, -.3
                ddh = s.rfun(x, [h, dh], l=l)[1]
                g = dh+(1-nu0*r*v)*h
                dg = ddh+(1-nu0*r*v)*dh-nu0*r*(v+dv)*h
                self.assertLess(abs(dg+2*g-l*(l+1)*h), 1e-10)

    def test_backward_rk4(self):
        actual = rk45_solver(lambda x, y, **kw: y, 0., np.array([1.]), .01, sig=-1)
        np.testing.assert_allclose(actual, [np.exp(-.01)], rtol=1e-11)

    def test_integration_matches_independent_solver_at_endpoints(self):
        s = self.solver
        initial = np.array([0., 1.])
        x, y = s.time_integration(initial, l=2, dl=.01, rmax=3.)
        order = np.argsort(x)
        x, y = x[order], y[order]
        self.assertEqual(x[0], 0.)
        self.assertAlmostEqual(x[-1], np.log(3.), places=14)
        ref = solve_ivp(lambda x, y: s.rfun(x, y, l=2), [np.log(s.Rs), 0.], initial, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(y[0], ref.y[:,-1], rtol=1e-7)

    def test_iteration_limit(self):
        x, _ = self.solver.time_integration(np.array([0., 1.]), dl=.001, max_steps=2)
        self.assertLessEqual(len(x), 3)

    def test_monopole_conserves_signed_flux(self):
        r = np.linspace(1., 2.6, 10)
        h, g = self.solver.compute_HG(r, l=0)
        np.testing.assert_array_equal(h, np.zeros_like(r))
        np.testing.assert_allclose(r*r*g, 1., rtol=1e-14)

    def test_shooting_field_matches_independent_radial_solution(self):
        s = self.solver
        r = np.array([1., 1.8, 2.6])
        ref = solve_ivp(lambda x, y: s.rfun(x, y, l=2), [np.log(s.Rs), 0.], [0.,1.],
                        rtol=1e-11, atol=1e-12, dense_output=True)
        h, dh = ref.sol(np.log(r))
        g = dh+(1-nu0*r*s.vout(np.log(r)))*h
        actual_h, actual_g = s.compute_HG(r, l=2, aim_N=4)
        np.testing.assert_allclose(actual_h, h/g[0], rtol=1e-8, atol=1e-12)
        np.testing.assert_allclose(actual_g, g/g[0], rtol=1e-8, atol=1e-12)

    def test_sonic_point_is_regular(self):
        for speed in (50., np.sqrt(9.54e4/(2*2.6))):
            s = off_solver(None, Rs=2.6, v1=speed)
            cs = np.sqrt(9.54e4/(2*s.rc))*1e5/v0
            rho = np.log(s.rc)+np.array([-1e-7, 0., 1e-7])
            np.testing.assert_allclose(s.vout(rho), cs*(1+rho-np.log(s.rc)), rtol=1e-12)
            np.testing.assert_allclose(s.d1_vout(rho), cs, rtol=1e-12)
            self.assertEqual(s.d2_vout(np.log(s.rc)), 0.)

    def test_near_sonic_derivatives_remain_regular(self):
        s = off_solver(None, Rs=2.6, v1=50.)
        cs = np.sqrt(9.54e4/(2*s.rc))*1e5/v0
        x = np.array([-1e-3, -1e-5, -1.001e-6, 1.001e-6, 1e-5, 1e-3])
        rho = np.log(s.rc)+x
        expected = -x/2+3*x*x/5-3*x**3/8+19*x**4/224
        np.testing.assert_allclose(s.d2_vout(rho)/cs, expected, rtol=1e-6, atol=1e-10)
