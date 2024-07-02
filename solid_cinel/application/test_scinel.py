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
from solid_cinel.core.xs import Dxs, Xs


def check_results(expected_result: np.array, *command_line_args) -> None:
    """
    Check the results of the command line arguments.

    Parameters
    ----------
    expected_result : np.array
        The expected result.
    command_line_args : list
        The command line arguments.

    Returns
    -------
    None
        Pass the test if the results are equal.
    """
    # Create a parser to simulate the command line arguments
    result = main(*command_line_args, write_to_file=False)

    # Check the returned result
    np.testing.assert_array_equal(result, expected_result)


class TestScinelTeff(unittest.TestCase):
    """
    Test the Effective temperature calculation in terminal application in the
    solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables
        """
        self.T = 300
        self.keyword = 'teff'
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_pdos = os.path.join(self.file_dir, 'inputTest/interp.300')
        self.pdos = Pdos.from_file([self.T], [self.file_pdos])

    def test_teff(self) -> None:
        """
        Test the effective temperature calculation
        """
        # Generate the expected result
        expected_result = np.array([self.T, self.pdos.fix_T(self.T).Teff])

        # Check the results
        check_results(expected_result, 'Teff', str(self.T),
                      self.file_pdos)


class TestScinelSab(unittest.TestCase):
    """
    Test the Sab class terminal application in the solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        self.T = 300
        self.keyword = 'sab'
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_alpha = os.path.join(self.file_dir, 'inputTest/alphaGrid')
        self.file_beta = os.path.join(self.file_dir, 'inputTest/betaGrid')
        self.file_pdos = os.path.join(self.file_dir, 'inputTest/interp.300')
        self.pdos = Pdos.from_file([self.T], [self.file_pdos])

    def test_fgm(self) -> None:
        """
        Test the fgm model for the generating S(alpha, -beta) tables.
        """
        # Generate the expected result:
        expected_result = Sab.from_fgm(self.file_alpha, self.file_beta).data.values

        # Check the results:
        check_results(expected_result, self.keyword, 'fgm',
                      self.file_alpha, self.file_beta, str(self.T))

    def test_sct(self) -> None:
        """
        Test the sct model for the generating S(alpha, -beta) tables.
        """
        # Generate the expected result:
        expected_result = Sab.from_sct(self.file_alpha, self.file_beta, self.T,
                                       self.pdos).data.values

        # Check the results:
        check_results(expected_result, self.keyword, 'sct',
                      self.file_alpha, self.file_beta, str(self.T), self.file_pdos)

    def test_pdos(self) -> None:
        """
        Test the pdos model for the generating S(alpha, -beta) tables.
        """
        # Generate the expected result:
        expected_result = Sab.from_pdos(self.file_alpha, self.file_beta, self.T,
                                        self.pdos).data.values

        # Check the results:
        check_results(expected_result, self.keyword, 'pdos',
                      self.file_alpha, self.file_beta, str(self.T), self.file_pdos)


class TestScinelScatFunc(unittest.TestCase):
    """
    Test the ScatFunc class terminal application in the solid_cinel package.
    """

    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
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

    def test_fgm(self) -> None:
        """
        Test the fgm model for the generating Van Hove scattering function tables.
        """
        # Generate the expected result:
        expected_result = ScatFunc.from_fgm(self.Ein, self.M, self.T, self.Eout,
                                            self.mu).data.values

        # Check the results:
        check_results(expected_result, self.keyword, 'fgm',
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      self.file_theta)

    def test_sct(self) -> None:
        """
        Test the sct model for the generating Van Hove scattering function tables.
        """
        # Generate the expected result:
        expected_result = ScatFunc.from_sct(self.Ein, self.M, self.T, self.Eout,
                                            self.mu, self.pdos).data.values

        # Check the results:
        check_results(expected_result, self.keyword, "sct",
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      self.file_theta, self.file_pdos)

    def test_pdos(self) -> None:
        """
        Test the pdos model for the generating Van Hove scattering function tables.
        """
        # Generate the expected result:
        expected_result = ScatFunc.from_pdos(self.Ein, self.M, self.T, self.Eout,
                                            self.mu, self.pdos).data.values

        # Check the results:
        check_results(expected_result, self.keyword, "pdos",
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      self.file_theta, self.file_pdos)


class TestScinelTransferFunc(unittest.TestCase):
    """
    Test the TransferFunc class terminal application in the solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
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

    def test_fgm(self) -> None:
        """
        Test the fgm model for generating transfer function based on an angle.
        """
        # Generate the expected result:
        model = 'fgm'
        expected_result = TransferFunc.from_theta(self.Ein, self.M, self.T, self.Eout, self.theta,
                                                  model=model).data.values
        # Check the results:
        check_results(expected_result, self.keyword, model,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout, str(self.theta))

    def test_sct(self) -> None:
        """
        Test the sct model for generating transfer function based on an angle.
        """
        # Generate the expected result:
        model = 'sct'
        expected_result = TransferFunc.from_theta(self.Ein, self.M, self.T, self.Eout, self.theta,
                                                  self.pdos, model=model).data.values

        # Check the results:
        check_results(expected_result, self.keyword, model,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout, str(self.theta),
                      self.file_pdos)

    def test_pdos(self) -> None:
        """
        Test the pdos model for generating transfer function based on an angle.
        """
        # Generate the expected result:
        model = 'pdos'
        expected_result = TransferFunc.from_theta(self.Ein, self.M, self.T, self.Eout, self.theta,
                                                  self.pdos, model=model).data.values

        # Check the results:
        check_results(expected_result, self.keyword, model,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout, str(self.theta),
                      self.file_pdos)


class TestScinelDxs(unittest.TestCase):
    """
    Test the Dxs class terminal application in the solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        self.T = 1000
        self.keyword = 'dxs'
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.Ein = 7.2
        self.M = 238.05077040419212
        self.file_Eout = os.path.join(self.file_dir, 'inputTest/EoutGrid')
        self.file_theta = os.path.join(self.file_dir, 'inputTest/thetaGrid')
        self.file_pdos = os.path.join(self.file_dir, 'inputTest/interp.300')
        self.file_xs0K = os.path.join(self.file_dir, 'inputTest/u238.0.2')
        self.Eout = np.loadtxt(self.file_Eout)
        self.pdos = Pdos.from_file([self.T], [self.file_pdos])
        self.xs0K = Xs.read_xs(self.file_xs0K)

    def test_fgm(self) -> None:
        """
        Test the fgm model for the generating differential cross section.
        """
        model = 'fgm'
        # Generate the expected result for single angle:
        theta = 60
        expected_result = Dxs.from_theta(self.xs0K, self.Ein, self.M, self.T,
                                         self.Eout, theta, model=model).data.values

        # Check the results:
        check_results(expected_result, self.keyword, model, self.file_xs0K,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      str(theta))

        # Generate the expected result for single angle:
        theta = np.loadtxt(self.file_theta)
        expected_result = Dxs.from_sab(self.xs0K, self.Ein, self.M, self.T,
                                       self.Eout, theta, model=model).data.values

        # Check the results:
        check_results(expected_result, self.keyword, model, self.file_xs0K,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      self.file_theta)

    def test_sct(self) -> None:
        """
        Test the sct model for the generating differential cross section.
        """
        model = 'SCT'
        # Generate the expected result for single angle:
        theta = 60
        expected_result = Dxs.from_theta(self.xs0K, self.Ein, self.M, self.T,
                                         self.Eout, theta, self.pdos,
                                         model=model).data.values

        # Check the results:
        check_results(expected_result, self.keyword, model, self.file_xs0K,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      str(theta), self.file_pdos)

        # Generate the expected result for single angle:
        theta = np.loadtxt(self.file_theta)
        expected_result = Dxs.from_sab(self.xs0K, self.Ein, self.M, self.T,
                                       self.Eout, theta, self.pdos,
                                       model=model).data.values

        # Check the results:
        check_results(expected_result, self.keyword, model, self.file_xs0K,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      self.file_theta, self.file_pdos)

    def test_pdos(self) -> None:
        """
        Test the pdos model for the generating differential cross section.
        """
        model = 'pdos'
        # Generate the expected result for single angle:
        theta = 60
        expected_result = Dxs.from_theta(self.xs0K, self.Ein, self.M, self.T,
                                         self.Eout, theta, self.pdos,
                                         model=model).data.values

        # Check the results:
        check_results(expected_result, self.keyword, model, self.file_xs0K,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      str(theta), self.file_pdos)

        # Generate the expected result for single angle:
        theta = np.loadtxt(self.file_theta)
        expected_result = Dxs.from_sab(self.xs0K, self.Ein, self.M, self.T,
                                       self.Eout, theta, self.pdos,
                                       model=model).data.values

        # Check the results:
        check_results(expected_result, self.keyword, model, self.file_xs0K,
                      str(self.Ein), str(self.M), str(self.T), self.file_Eout,
                      self.file_theta, self.file_pdos)


if __name__ == '__main__':
    unittest.main()
