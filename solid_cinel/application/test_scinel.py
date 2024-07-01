import unittest
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Application test:
from solid_cinel.application.scinel import main

# POO direct application:
from solid_cinel.core.scattering_function.sab import Sab
from solid_cinel.core.scattering_function.scatfunc import ScatFunc, TransferFunc
from solid_cinel.core.material.vibration.pdos import Pdos


def check_results(expected_result, *command_line_args):
    # Create a parser to simulate the command line arguments
    result = main(*command_line_args, write_to_file=False)

    # Check the returned result
    np.testing.assert_array_equal(result, expected_result)


class TestScinelSab(unittest.TestCase):
    def setUp(self):
        self.T = 300
        self.keyword = 'sab'
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_alpha = os.path.join(self.file_dir, 'inputTest/alphaGrid')
        self.file_beta = os.path.join(self.file_dir, 'inputTest/betaGrid')
        self.file_pdos = os.path.join(self.file_dir, 'inputTest/interp.300')
        self.pdos = Pdos.from_file([self.T], [self.file_pdos])

    def test_fgm(self):
        # Generate the expected result:
        expected_result = Sab.from_fgm(self.file_alpha, self.file_beta).data.values

        # Check the results:
        check_results(expected_result, self.keyword, 'fgm',
                      self.file_alpha, self.file_beta, str(self.T))

    def test_sct(self):
        # Generate the expected result:
        expected_result = Sab.from_sct(self.file_alpha, self.file_beta, self.T,
                                       self.pdos).data.values

        # Check the results:
        check_results(expected_result, self.keyword, 'sct',
                      self.file_alpha, self.file_beta, str(self.T), self.file_pdos)

    def test_pdos(self):
        # Generate the expected result:
        expected_result = Sab.from_pdos(self.file_alpha, self.file_beta, self.T,
                                        self.pdos).data.values

        # Check the results:
        check_results(expected_result, self.keyword, 'pdos',
                      self.file_alpha, self.file_beta, str(self.T), self.file_pdos)


class TestScinelScatFunc(unittest.TestCase):

    def setUp(self):
        self.T = 1000
        self.keyword = 'scatfunc'
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.Ein = 7.2
        self.M = 238.05077040419212
        self.file_Eout = os.path.join(self.file_dir, 'inputTest/EoutGrid')
        self.file_theta = os.path.join(self.file_dir, 'inputTest/thetaGrid')
        self.file_pdos = os.path.join(self.file_dir, 'inputTest/interp.300')
        self.Eout = np.loadtxt(self.file_Eout)
        self.mu = np.cos(np.deg2rad(np.loadtxt(self.file_theta)))
        self.pdos = Pdos.from_file([self.T], [self.file_pdos])

    def test_fgm(self):
        # Generate the expected result:
        expected_result = ScatFunc.from_fgm(self.Ein, self.M, self.T, self.Eout,
                                            self.mu).data.values

        # Check the results:
        check_results(expected_result, self.keyword, 'fgm',
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      self.file_theta)

    def test_sct(self):
        # Generate the expected result:
        expected_result = ScatFunc.from_sct(self.Ein, self.M, self.T, self.Eout,
                                            self.mu, self.pdos).data.values

        # Check the results:
        check_results(expected_result, self.keyword, "sct",
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      self.file_theta, self.file_pdos)

    def test_pdos(self):
        # Generate the expected result:
        expected_result = ScatFunc.from_pdos(self.Ein, self.M, self.T, self.Eout,
                                            self.mu, self.pdos).data.values

        # Check the results:
        check_results(expected_result, self.keyword, "pdos",
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      self.file_theta, self.file_pdos)


class TestScinelTransferFunc(unittest.TestCase):
    def setUp(self):
        self.T = 1000
        self.keyword = 'scatfunc'
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.Ein = 7.2
        self.M = 238.05077040419212
        self.file_Eout = os.path.join(self.file_dir, 'inputTest/EoutGrid')
        self.theta = 60
        self.file_pdos = os.path.join(self.file_dir, 'inputTest/interp.300')
        self.Eout = np.loadtxt(self.file_Eout)
        self.pdos = Pdos.from_file([self.T], [self.file_pdos])

    def test_fgm(self):
        # Generate the expected result:
        model = 'fgm'
        expected_result = TransferFunc.from_theta(self.Ein, self.M, self.T, self.Eout, self.theta,
                                                  model=model).data.values
        # Check the results:
        check_results(expected_result, self.keyword, model,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout, str(self.theta))

    def test_sct(self):
        # Generate the expected result:
        model = 'sct'
        expected_result = TransferFunc.from_theta(self.Ein, self.M, self.T, self.Eout, self.theta,
                                                  self.pdos, model=model).data.values

        # Check the results:
        check_results(expected_result, self.keyword, model,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout, str(self.theta),
                      self.file_pdos)

    def test_pdos(self):
        # Generate the expected result:
        model = 'pdos'
        expected_result = TransferFunc.from_theta(self.Ein, self.M, self.T, self.Eout, self.theta,
                                                  self.pdos, model=model).data.values

        # Check the results:
        check_results(expected_result, self.keyword, model,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout, str(self.theta),
                      self.file_pdos)

class TestScinelTeff(unittest.TestCase):

    def setUp(self):
        self.T = 300
        self.keyword = 'teff'
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_pdos = os.path.join(self.file_dir, 'inputTest/interp.300')
        self.pdos = Pdos.from_file([self.T], [self.file_pdos])

    def test_teff(self):
        # Generate the expected result
        expected_result = np.array([self.T, self.pdos.fix_T(self.T).Teff])

        # Check the results
        check_results(expected_result, 'Teff', str(self.T),
                      self.file_pdos)


if __name__ == '__main__':
    unittest.main()
