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
from solid_cinel.core.xs import Dxs, Xs, DDxs


class BaseTestScinel(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        self.T = 300
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.Ein = 7.2
        self.M = 238.05077040419212
        self.file_alpha = os.path.join(self.file_dir, 'inputTest/alphaGrid')
        self.file_beta = os.path.join(self.file_dir, 'inputTest/betaGrid')
        self.file_Eout = os.path.join(self.file_dir, 'inputTest/EoutGrid')
        self.file_theta = os.path.join(self.file_dir, 'inputTest/thetaGrid')
        self.file_pdos = os.path.join(self.file_dir, 'inputTest/interp.300')
        self.file_xs0K = os.path.join(self.file_dir, 'inputTest/u238.0.2')
        self.Eout = np.loadtxt(self.file_Eout)
        self.pdos = Pdos.from_file([self.T], [self.file_pdos])
        self.xs0K = Xs.read_xs(self.file_xs0K)
        self.theta = np.loadtxt(self.file_theta)

    @staticmethod
    def check_command_line_args(*command_line_args) -> list:
        """
        Check the command line arguments and convert them to string if needed.

        Parameters
        ----------
        command_line_args : list
            The command line arguments.

        Returns
        -------
        list
            The list of command line arguments converted to string.
        """
        return [str(arg) if not isinstance(arg, str) else arg for arg in
                command_line_args]

    def check_results(self, expected_result: np.array, *command_line_args) -> None:
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
        # Convert the command line arguments to string
        command_line_args = self.check_command_line_args(*command_line_args)

        # Create a parser to simulate the command line arguments
        result = main(*command_line_args, write_to_file=False)

        # Check the returned result
        np.testing.assert_array_equal(result, expected_result)


class TestScinelTeff(BaseTestScinel):
    """
    Test the Effective temperature calculation in terminal application in the
    solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables
        """
        super().setUp()
        self.keyword = 'teff'
        self.T = 300

    def test_teff(self) -> None:
        """
        Test the effective temperature calculation
        """
        # Generate the expected result
        expected_result = np.array([self.T, self.pdos.fix_T(self.T).Teff])

        # Command line arguments
        command_line_args = [self.keyword, self.T, self.file_pdos]

        # Check the results
        self.check_results(expected_result, *command_line_args)


class TestScinelSab(BaseTestScinel):
    """
    Test the Sab class terminal application in the solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        super().setUp()
        self.keyword = 'sab'

    @property
    def get_fgm_var(self) -> list:
        """
        Get the variables for the fgm model.
        """
        return [self.file_alpha, self.file_beta]

    @property
    def get_sct_var(self) -> list:
        """
        Get the variables for the sct model.
        """
        var_fgm = self.get_fgm_var
        return var_fgm.extend([self.T, self.file_pdos])

    def test_fgm(self) -> None:
        """
        Test the fgm model for the generating S(alpha, -beta) tables.
        """
        # Generate the expected result:
        expected_result = Sab.from_fgm(*self.get_fgm_var).data.values

        # Command line arguments
        command_line_args = [self.keyword, 'fgm', self.file_alpha, self.file_beta]

        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def test_sct(self) -> None:
        """
        Test the sct model for the generating S(alpha, -beta) tables.
        """
        # Generate the expected result:
        expected_result = Sab.from_sct(*self.get_sct_var).data.values

        # Command line arguments
        command_line_args = [self.keyword, 'sct', self.file_alpha, self.file_beta,
                             self.T, self.file_pdos]

        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def test_pdos(self) -> None:
        """
        Test the pdos model for the generating S(alpha, -beta) tables.
        """
        # Generate the expected result:
        expected_result = Sab.from_pdos(*self.get_sct_var).data.values

        # Command line arguments
        command_line_args = [self.keyword, 'pdos', self.file_alpha, self.file_beta,
                             self.T, self.file_pdos]

        # Check the results:
        self.check_results(expected_result, *command_line_args)


class TestScinelScatFunc(BaseTestScinel):
    """
    Test the ScatFunc class terminal application in the solid_cinel package.
    """

    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        super().setUp()
        self.keyword = 'scatfunc'
        self.mu = np.cos(np.deg2rad(self.theta))

    @property
    def get_fgm_var(self) -> list:
        """
        Get the variables for the fgm model.
        """
        return [self.Ein, self.M, self.T, self.Eout, self.mu]

    @property
    def get_sct_var(self) -> list:
        """
        Get the variables for the sct model.
        """
        var_fgm = self.get_fgm_var
        return var_fgm.extend([self.pdos])

    def test_fgm(self) -> None:
        """
        Test the fgm model for the generating Van Hove scattering function tables.
        """
        # Generate the expected result:
        expected_result = ScatFunc.from_fgm(*self.get_fgm_var).data.values

        # Command line arguments
        command_line_args = [self.keyword, 'fgm', self.Ein, self.M, self.T,
                             self.file_Eout, self.file_theta]
        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def test_sct(self) -> None:
        """
        Test the sct model for the generating Van Hove scattering function tables.
        """
        # Generate the expected result:
        expected_result = ScatFunc.from_sct(*self.get_sct_var).data.values

        # Command line arguments
        command_line_args = [self.keyword, 'sct', self.Ein, self.M, self.T,
                             self.file_Eout, self.file_theta, self.file_pdos]

        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def test_pdos(self) -> None:
        """
        Test the pdos model for the generating Van Hove scattering function tables.
        """
        # Generate the expected result:
        expected_result = ScatFunc.from_pdos(*self.get_sct_var).data.values

        # Command line arguments
        command_line_args = [self.keyword, 'pdos', self.Ein, self.M, self.T,
                             self.file_Eout, self.file_theta, self.file_pdos]

        # Check the results:
        self.check_results(expected_result, *command_line_args)


class TestScinelTransferFunc(BaseTestScinel):
    """
    Test the TransferFunc class terminal application in the solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        super().setUp()
        self.keyword = 'scatfunc'
        self.theta = 60

    @property
    def get_fgm_var(self) -> list:
        """
        Get the variables for the fgm model.
        """
        return [self.Ein, self.M, self.T, self.Eout, self.theta]

    @property
    def get_sct_var(self) -> list:
        """
        Get the variables for the sct model.
        """
        var_fgm = self.get_fgm_var
        return var_fgm.extend([self.pdos])

    def test_fgm(self) -> None:
        """
        Test the fgm model for generating transfer function based on an angle.
        """
        # Generate the expected result:
        model = 'fgm'
        expected_result = TransferFunc.from_theta(*self.get_fgm_var,
                                                  model=model).data.values

        # Command line arguments
        command_line_args = [self.keyword, model, self.Ein, self.M, self.T,
                             self.file_Eout, self.theta]
        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def test_sct(self) -> None:
        """
        Test the sct model for generating transfer function based on an angle.
        """
        # Generate the expected result:
        model = 'sct'
        expected_result = TransferFunc.from_theta(*self.get_sct_var, model=model).data.values

        # Command line arguments
        command_line_args = [self.keyword, model, self.Ein, self.M, self.T,
                             self.file_Eout, self.theta, self.file_pdos]

        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def test_pdos(self) -> None:
        """
        Test the pdos model for generating transfer function based on an angle.
        """
        # Generate the expected result:
        model = 'pdos'
        expected_result = TransferFunc.from_theta(*self.get_sct_var, model=model).data.values

        # Command line arguments
        command_line_args = [self.keyword, model, self.Ein, self.M, self.T,
                             self.file_Eout, self.theta, self.file_pdos]

        # Check the results:
        self.check_results(expected_result, *command_line_args)


class TestScinelDxs(BaseTestScinel):
    """
    Test the Dxs class terminal application in the solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        super().setUp()
        self.keyword = 'dxs'

    @property
    def get_fgm_var(self) -> list:
        """
        Get the variables for the fgm model.
        """
        return [self.xs0K, self.Ein, self.M, self.T, self.Eout, self.theta]

    @property
    def get_sct_var(self) -> list:
        """
        Get the variables for the sct model.
        """
        fgm_var = self.get_fgm_var
        return fgm_var.extend([self.pdos])

    def ModelFgm(self, method, theta: [float, np.array]) -> None:
        """
        Test the FGM model for the generating differential cross section.

        Parameters
        ----------
        method : method
            The method to use for the calculation.
        theta : [float, np.array]
            The scattering angle.
        """
        model = 'fgm'
        # Generate the expected result:
        expected_result = method(*self.get_fgm_var, model=model).data.values

        # Command line arguments
        command_line_args = [self.keyword, model, self.file_xs0K, self.Ein,
                             self.M, self.T, self.file_Eout, theta]

        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def ModelSct(self, method, theta: [float, np.array]) -> None:
        """
        Test the SCT model for the generating differential cross section.

        Parameters
        ----------
        method : method
            The method to use for the calculation.
        theta : [float, np.array]
            The scattering angle.
        """
        model = 'SCT'
        # Generate the expected result:
        expected_result = method(self.get_sct_var, model=model).data.values

        # Command line arguments
        command_line_args = [self.keyword, model, self.file_xs0K, self.Ein,
                             self.M, self.T, self.file_Eout, theta, self.file_pdos]

        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def ModelPdos(self, method, theta: [float, np.array]) -> None:
        """
        Test the PDOS model for the generating differential cross section.

        Parameters
        ----------
        method : method
            The method to use for the calculation.
        theta : [float, np.array]
            The scattering angle.
        """
        model = 'pdos'
        # Generate the expected result:
        expected_result = method(*self.get_sct_var, model=model).data.values

        # Command line arguments
        command_line_args = [self.keyword, model, self.file_xs0K, self.Ein,
                             self.M, self.T, self.file_Eout, theta, self.file_pdos]

        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def Modeltest(self, method, theta: [float, np.array]) -> None:
        """
        Test the differential cross section calculation.

        Parameters
        ----------
        method : method
            The method to use for the calculation.
        theta : [float, np.array]
            The scattering angle.
        """
        # Test FGM:
        self.ModelFgm(method, theta)

        # Test SCT:
        self.ModelSct(method, theta)

        # Test PDOS:
        self.ModelPdos(method, theta)

    def test_SingleAngle(self) -> None:
        """
        Test the fgm model for the generating differential cross section for a
        single angle.
        """
        self.Modeltest(Dxs.from_theta, 60)

    def test_MultipleAngles(self) -> None:
        """
        Test the fgm model for the generating differential cross section for
        multiple angles.
        """
        self.Modeltest(Dxs.from_sab, self.theta)


class TestScinelDDxs(BaseTestScinel):
    """
    Test the DDxs class terminal application in the solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        super().setUp()
        self.keyword = 'ddxs'
        self.xs = Xs(self.M, 0, self.xs0K)

    @property
    def get_fgm_var(self) -> list:
        """
        Get the variables for the fgm model.
        """
        return [self.xs, self.Ein, self.T, self.Eout, self.theta]

    @property
    def get_sct_var(self) -> list:
        """
        Get the variables for the sct model.
        """
        var_fgm = self.get_fgm_var
        return var_fgm.extend([self.pdos])

    def ModelFgm(self, algorithm: str, method):
        """
        Test the FGM model for the generating double differential scattering
        cross section.

        Parameters
        ----------
        algorithm : str
            The algorithm to use for the calculation.
        method : method
            The method to use for the calculation.
        """
        model = "fgm"
        # Generate the expected result:
        expected_result = method(*self.get_fgm_var, model=model).data.values

        # Command line arguments
        command_line_args = [self.keyword, algorithm, model, self.file_xs0K,
                             self.Ein, self.M, self.T, self.file_Eout, self.file_theta]

        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def ModelSct(self, algorithm: str, method) -> None:
        """
        Test the SCT model for the generating double differential scattering
        cross section.

        Parameters
        ----------
        algorithm : str
            The algorithm to use for the calculation.
        method : method
            The method to use for the calculation.
        """

        model = "sct"
        # Generate the expected result:
        expected_result = method(*self.get_sct_var, model=model).data.values

        # Command line arguments
        command_line_args = [self.keyword, algorithm, model, self.file_xs0K,
                             self.Ein, self.M, self.T, self.file_Eout, self.file_theta,
                             self.file_pdos]
        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def ModelPdos(self, algorithm: str, method) -> None:
        """
        Test the PDOS model for the generating double differential scattering
        cross section.

        Parameters
        ----------
        algorithm : str
            The algorithm to use for the calculation.
        method : method
            The method to use for the calculation.
        """
        model = "pdos"
        # Generate the expected result:
        expected_result = method(*self.get_sct_var, model=model).data.values

        # Command line arguments
        command_line_args = [self.keyword, algorithm, model, self.file_xs0K,
                             self.Ein, self.M, self.T, self.file_Eout, self.file_theta,
                             self.file_pdos]

        # Check the results:
        self.check_results(expected_result, *command_line_args)

    def Modeltest(self, algorithm: str, method) -> None:
        """
        Test the double differential scattering cross section calculation.

        Parameters
        ----------
        algorithm : str
            The algorithm to use for the calculation.
        method : method
            The method to use for the calculation.
        """
        # Test FGM:
        self.ModelFgm(algorithm, method)

        # Test SCT:
        self.ModelSct(algorithm, method)

        # Test PDOS:
        self.ModelPdos(algorithm, method)

    def test_sab(self) -> None:
        """
        Test the SAB algorithm for the generating double differential scattering
        cross section.
        """
        self.Modeltest('sab', DDxs.from_Sab)

    def test_4pcf(self) -> None:
        """
        Test the 4PCF algorithm for the generating double differential scattering
        cross section.
        """
        self.Modeltest('4pcf', DDxs.from_4PCF)


if __name__ == '__main__':
    unittest.main()
