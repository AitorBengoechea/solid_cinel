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


class TestScinel(unittest.TestCase):

    def check_results(self, expected_result, *command_line_args):
        # Create a parser to simulate the command line arguments
        result = main(*command_line_args, write_to_file=False)

        # Check the returned result
        np.testing.assert_array_equal(result, expected_result)

    def test_Teff(self):
        # Input simulation parameters:
        keyword = 'Teff'
        file_dir = os.path.dirname(os.path.abspath(__file__))

        # Input simulation parameters
        T = 300
        file = os.path.join(file_dir, 'inputTest/interp.300')

        # Generate the expected result:
        pdos = Pdos.from_file([T], [file])
        expected_result = np.array([T, pdos.fix_T(T).Teff])

        # Check the results
        self.check_results(expected_result, keyword, str(T), file)

    def test_Sab_fgm(self):
        # Input simulation parameters:
        T = 300
        keyword = 'sab'
        file_dir = os.path.dirname(os.path.abspath(__file__))

        # Input simulation parameters for FGM:
        model = 'fgm'
        file_alpha = os.path.join(file_dir, 'inputTest/alphaGrid')
        file_beta = os.path.join(file_dir, 'inputTest/betaGrid')

        # Generate the expected result:
        expected_result = Sab.from_fgm(file_alpha, file_beta).data.values

        # Check the results
        self.check_results(expected_result, keyword, model, file_alpha,
                           file_beta, str(T))

    def test_Sab_sct(self):
        # Input simulation parameters:
        T = 300
        keyword = 'sab'
        file_dir = os.path.dirname(os.path.abspath(__file__))

        # Input simulation parameters for Sct:
        model = 'sct'
        file_alpha = os.path.join(file_dir, 'inputTest/alphaGrid')
        file_beta = os.path.join(file_dir, 'inputTest/betaGrid')
        file_pdos = os.path.join(file_dir, 'inputTest/interp.300')

        # Generate the expected result:
        pdos = Pdos.from_file([T], [file_pdos])
        expected_result = Sab.from_sct(file_alpha, file_beta, T, pdos).data.values

        # Check the results
        self.check_results(expected_result, keyword, model, file_alpha,
                           file_beta, str(T), file_pdos)

    def test_Sab_pdos(self):
        # Input simulation parameters:
        T = 300
        keyword = 'sab'
        file_dir = os.path.dirname(os.path.abspath(__file__))

        # Input simulation parameters for Sct:
        model = 'pdos'
        file_alpha = os.path.join(file_dir, 'inputTest/alphaGrid')
        file_beta = os.path.join(file_dir, 'inputTest/betaGrid')
        file_pdos = os.path.join(file_dir, 'inputTest/interp.300')

        # Generate the expected result:
        pdos = Pdos.from_file([T], [file_pdos])
        expected_result = Sab.from_pdos(file_alpha, file_beta, T, pdos).data.values

        # Check the results
        self.check_results(expected_result, keyword, model, file_alpha,
                           file_beta, str(T), file_pdos)

    def test_ScatFunc_fgm(self):
        # Input simulation parameters:
        T = 1000
        keyword = 'scatfunc'
        file_dir = os.path.dirname(os.path.abspath(__file__))

        # Input simulation parameters for FGM:
        model = 'fgm'
        Ein = 7.2
        M = 238.05077040419212
        file_Eout = os.path.join(file_dir, 'inputTest/EoutGrid')
        file_theta = os.path.join(file_dir, 'inputTest/thetaGrid')

        # Generate the expected result:
        mu = np.cos(np.deg2rad(np.loadtxt(file_theta)))
        Eout = np.loadtxt(file_Eout)
        expected_result = ScatFunc.from_fgm(Ein, M, T, Eout, mu).data.values

        # Check the results:
        self.check_results(expected_result, keyword, model, str(Ein), str(M),
                           str(T), file_Eout, file_theta)

    def test_ScatFunc_sct(self):
        # Input simulation parameters:
        T = 1000
        keyword = 'scatfunc'
        file_dir = os.path.dirname(os.path.abspath(__file__))

        # Input simulation parameters for FGM:
        model = 'sct'
        Ein = 7.2
        M = 238.05077040419212
        file_Eout = os.path.join(file_dir, 'inputTest/EoutGrid')
        file_theta = os.path.join(file_dir, 'inputTest/thetaGrid')
        file_pdos = os.path.join(file_dir, 'inputTest/interp.300')

        # Generate the expected result:
        pdos = Pdos.from_file([T], [file_pdos])
        mu = np.cos(np.deg2rad(np.loadtxt(file_theta)))
        Eout = np.loadtxt(file_Eout)
        expected_result = ScatFunc.from_sct(Ein, M, T, Eout, mu, pdos).data.values

        # Check the results:
        self.check_results(expected_result, keyword, model, str(Ein), str(M),
                           str(T), file_Eout, file_theta, file_pdos)

    def test_ScatFunc_pdos(self):
        # Input simulation parameters:
        T = 1000
        keyword = 'scatfunc'
        file_dir = os.path.dirname(os.path.abspath(__file__))

        # Input simulation parameters for FGM:
        model = 'pdos'
        Ein = 7.2
        M = 238.05077040419212
        file_Eout = os.path.join(file_dir, 'inputTest/EoutGrid')
        file_theta = os.path.join(file_dir, 'inputTest/thetaGrid')
        file_pdos = os.path.join(file_dir, 'inputTest/interp.300')

        # Generate the expected result:
        pdos = Pdos.from_file([T], [file_pdos])
        mu = np.cos(np.deg2rad(np.loadtxt(file_theta)))
        Eout = np.loadtxt(file_Eout)
        expected_result = ScatFunc.from_pdos(Ein, M, T, Eout, mu, pdos).data.values

        # Check the results:
        self.check_results(expected_result, keyword, model, str(Ein), str(M),
                           str(T), file_Eout, file_theta, file_pdos)

if __name__ == '__main__':
    unittest.main()
