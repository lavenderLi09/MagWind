import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from pfss.pfss_module import pfss_solver
from pfss.off_module import off_solver


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous = os.getcwd()
        os.chdir(self.directory.name)

    def tearDown(self):
        os.chdir(self.previous)
        self.directory.cleanup()

    def pfss(self):
        s = pfss_solver(None, n_r=3, n_t=8, n_p=12, lmax=1, Rs=2.6, Br=np.ones((8,12)))
        s.Alm = {0: {0: 0.}, 1: {-1: 0., 0: -s.Rs**-3, 1: 0.}}
        s.Blm = {0: {0: 0.}, 1: {-1: 0., 0: 1., 1: 0.}}
        for name in ('Alm', 'Blm'):
            Path(name+'.pkl').write_bytes(pickle.dumps(getattr(s, name)))
        return s

    def test_pfss_worker_matches_harmonics_and_saved_component_order(self):
        s = self.pfss()
        expected = s.get_Brtp(s.get_rtp(), method='harmonics', device='cpu')
        result = s.get_pfss(load_coef=True, fast_mode=True, device='cpu', python=sys.executable, pfss_file='custom.npy')
        np.testing.assert_allclose(result, expected, atol=1e-14)
        np.testing.assert_array_equal(np.load('custom.npy'), result)

    def test_single_worker_custom_grid(self):
        s = self.pfss()
        expected = s.get_Brtp(s.get_rtp(), method='harmonics', device='cpu')
        import subprocess
        popen = subprocess.Popen
        def checked_popen(command, *args, **kwargs):
            # Fail before launching a worker with the legacy 400x200x400 defaults.
            for argument in ('-nr 3', '-nt 8', '-np 12', '-rs 2.6', '--bound'):
                self.assertIn(argument, command)
            return popen(command, *args, **kwargs)
        with patch('pfss.pfss_module.subprocess.Popen', side_effect=checked_popen):
            result = s.multi_gpu_pfss(['cpu'], fast_mode=True, load_coef=True, python=sys.executable)
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_off_worker_matches_serial_monopole(self):
        s = off_solver(None, n_r=3, n_t=8, n_p=12, lmax=0, Rs=2.6, v1=.001)
        s.Br = np.ones((8,12))
        expected = s.get_outflow_field(n_cores=1, device='cpu')
        result = s.get_outflow_field(n_cores=1, fast_mode=True, devices=['cpu'], python=sys.executable)
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_off_rejects_stale_worker_directory(self):
        Path('OFF_temp_files').mkdir()
        s = off_solver(None, n_r=3, n_t=8, n_p=12, lmax=0)
        with self.assertRaises(FileExistsError):
            s.get_outflow_field(fast_mode=True, n_cores=1, device='cpu')
