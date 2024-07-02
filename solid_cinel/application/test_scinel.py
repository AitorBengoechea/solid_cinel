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

    def modelCheck(self, model: str, variables: list, method,
                   command_line: list) -> None:
        """
        Test the specified model for the calculation.

        Parameters
        ----------
        variables : list
            The variables for the model.
        method : method
            The method to use for the calculation.
        model : str
            The model to use for the calculation.
        command_line : list
            The command line arguments.
        """
        # Generate the expected result
        expected_result = method(*variables, model=model).data.values

        # Check the results
        self.check_results(expected_result, *command_line)

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
        return [self.file_alpha, self.file_beta, self.T]

    @property
    def get_sct_var(self) -> list:
        """
        Get the variables for the sct model.
        """
        return self.get_fgm_var + [self.pdos]

    def get_command(self, model: str) -> list:
        """
        Get the command line arguments for the model.

        Parameters
        ----------
        model : str
            The model to use for the calculation.

        Returns
        -------
        list
            The command line arguments.
        """
        command = [self.keyword, model, self.file_alpha, self.file_beta, self.T]
        return command if model == 'fgm' else command + [self.file_pdos]

    def modelTest(self, model: str) -> None:
        """
        Test the specified model for generating Van Hove scattering function tables.

        Parameters
        ----------
        model : str
            The model to test. Can be 'fgm', 'sct', or 'pdos'.
        """
        # Determine the appropriate variables based on the model
        variables = self.get_fgm_var if model == 'fgm' else self.get_sct_var

        # Check the results
        self.modelCheck(model, variables, Sab.from_model, self.get_command(model))

    def test_fgm(self) -> None:
        """
        Test the fgm model for the generating S(alpha, -beta) tables.
        """
        self.modelTest("fgm")

    def test_sct(self) -> None:
        """
        Test the sct model for the generating S(alpha, -beta) tables.
        """
        self.modelTest('sct')

    def test_pdos(self) -> None:
        """
        Test the pdos model for the generating S(alpha, -beta) tables.
        """
        self.modelTest('pdos')

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
        return [self.Ein, self.M, self.T, self.Eout, self.theta]

    @property
    def get_sct_var(self) -> list:
        """
        Get the variables for the sct model.
        """
        return self.get_fgm_var + [self.pdos]

    def get_command(self, model: str) -> list:
        """
        Get the command line arguments for the model.

        Parameters
        ----------
        model : str
            The model to use for the calculation.

        Returns
        -------
        list
            The command line arguments.
        """
        command = [self.keyword, model, self.Ein, self.M, self.T,
                   self.file_Eout, self.file_theta]
        return command if model == 'fgm' else command + [self.file_pdos]

    def modelTest(self, model: str) -> None:
        """
        Test the specified model for generating Van Hove scattering function tables.

        Parameters
        ----------
        model : str
            The model to test. Can be 'fgm', 'sct', or 'pdos'.
        """
        # Determine the appropriate variables based on the model
        variables = self.get_fgm_var if model == 'fgm' else self.get_sct_var

        # Check the results
        self.modelCheck(model, variables, ScatFunc.from_model, self.get_command(model))

    def test_fgm(self) -> None:
        """
        Test the fgm model for generating transfer function based on an angle.
        """
        self.modelTest('fgm')

    def test_sct(self) -> None:
        """
        Test the sct model for generating transfer function based on an angle.
        """
        self.modelTest('sct')

    def test_pdos(self) -> None:
        """
        Test the pdos model for generating transfer function based on an angle.
        """
        self.modelTest('pdos')


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
        return self.get_fgm_var + [self.pdos]

    def get_command(self, model: str) -> list:
        """
        Get the command line arguments for the model.

        Parameters
        ----------
        model : str
            The model to use for the calculation.

        Returns
        -------
        list
            The command line arguments.
        """
        command = [self.keyword, model, self.Ein, self.M, self.T,
                   self.file_Eout, self.theta]
        return command if model == 'fgm' else command + [self.file_pdos]

    def modelTest(self, model: str) -> None:
        """
        Test the specified model for generating transfer function based on an angle.

        Parameters
        ----------
        model : str
            The model to test. Can be 'fgm', 'sct', or 'pdos'.
        """
        # Determine the appropriate variables based on the model
        variables = self.get_fgm_var if model == 'fgm' else self.get_sct_var

        # Check the results
        self.modelCheck(model, variables, TransferFunc.from_theta, self.get_command(model))

    def test_fgm(self) -> None:
        """
        Test the fgm model for generating transfer function based on an angle.
        """
        self.modelTest('fgm')

    def test_sct(self) -> None:
        """
        Test the sct model for generating transfer function based on an angle.
        """
        self.modelTest('sct')

    def test_pdos(self) -> None:
        """
        Test the pdos model for generating transfer function based on an angle.
        """
        self.modelTest('pdos')


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

    def get_fgm_var(self, theta: [float, np.array]) -> list:
        """
        Get the variables for the fgm model.
        """
        return [self.xs0K, self.Ein, self.M, self.T, self.Eout, theta]

    def get_sct_var(self, theta: [float, np.array]) -> list:
        """
        Get the variables for the sct model.
        """
        return self.get_fgm_var(theta) + [self.pdos]

    def get_command(self, model: str, theta: [float, np.array]) -> list:
        """
        Get the command line arguments for the model.
        Parameters
        ----------
        model : str
            The model to use for the calculation.
        theta : [float, np.array]
            The scattering angle.

        Returns
        -------
        list
            The command line arguments.
        """
        command = [self.keyword, model, self.file_xs0K, self.Ein, self.M, self.T,
                   self.file_Eout]
        command += [theta] if isinstance(theta, (int, float)) else [self.file_theta]
        return command if model == 'fgm' else command + [self.file_pdos]

    def modelTest(self, method, model: str,  theta: [float, np.array]) -> None:
        """
        Test the differential cross section calculation for the specified model.

        Parameters
        ----------
        method : method
            The method to use for the calculation.
        model : str
            The model to use for the calculation.
        theta : [float, np.array]
            The scattering angle.
        """
        # Determine the appropriate variables based on the model
        variables = self.get_fgm_var if model == 'fgm' else self.get_sct_var

        # Check the results
        self.modelCheck(model, variables(theta), method, self.get_command(model, theta))

    def allModelTest(self, method, theta: [float, np.array]) -> None:
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
        self.model(method, "fgm", theta)

        # Test SCT:
        self.model(method, "sct", theta)

        # Test PDOS:
        self.model(method, "pdos", theta)

    def test_SingleAngle(self) -> None:
        """
        Test the fgm model for the generating differential cross section for a
        single angle.
        """
        self.allModelTest(Dxs.from_theta, 60)

    def test_MultipleAngles(self) -> None:
        """
        Test the fgm model for the generating differential cross section for
        multiple angles.
        """
        self.allModelTest(Dxs.from_sab, self.theta)


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
        return self.get_fgm_var + [self.pdos]

    def get_command(self, algorithm: str, model: str) -> list:
        """
        Get the command line arguments for the model.

        Parameters
        ----------
        model : str
            The model to use for the calculation.

        Returns
        -------
        list
            The command line arguments.
        """
        command = [self.keyword, algorithm, model, self.file_xs0K, self.Ein,
                   self.M, self.T, self.file_Eout, self.file_theta]
        return command if model == 'fgm' else command + [self.file_pdos]

    def modelTest(self, algorithm: str, method, model: str) -> None:
        # Determine the appropriate variables based on the model
        variables = self.get_fgm_var if model == 'fgm' else self.get_sct_var

        # Check the results
        self.modelCheck(model, variables, method, self.get_command(algorithm, model))

    def allModelTest(self, algorithm: str,  method) -> None:
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
        self.modelTest(algorithm, method, "fgm")

        # Test SCT:
        self.modelTest(algorithm, method, "sct")

        # Test PDOS:
        self.modelTest(algorithm, method, "pdos")

    def test_sab(self) -> None:
        """
        Test the SAB algorithm for the generating double differential scattering
        cross section.
        """
        self.allModelTest('sab', DDxs.from_Sab)

    def test_4pcf(self) -> None:
        """
        Test the 4PCF algorithm for the generating double differential scattering
        cross section.
        """
        self.allModelTest('4pcf', DDxs.from_4PCF)


if __name__ == '__main__':
    unittest.main()
