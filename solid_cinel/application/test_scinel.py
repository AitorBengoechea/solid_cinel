import unittest
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Application test:
from solid_cinel.application.scinel import main

# POO direct application:
from solid_cinel.core.dynamic_structure.sab import Sab
from solid_cinel.core.dynamic_structure.dynamicStruc import DynamicStruc
from solid_cinel.core.material import Pdos, Solid
from solid_cinel.core.xs import DDxs, Xs0K
from solid_cinel.application.xsApp import calc_alpha0


class BaseTestScinel(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        # Get the floats:
        self.T = 300
        self.Ein = 7.2
        self.M = 238.05077040419212

        # Get the files:
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_alpha = os.path.join(self.file_dir, 'inputTest/alphaGrid')
        self.file_beta = os.path.join(self.file_dir, 'inputTest/betaGrid')
        self.file_Eout = os.path.join(self.file_dir, 'inputTest/EoutGrid')
        self.file_theta = os.path.join(self.file_dir, 'inputTest/thetaGrid')
        self.file_pdos = os.path.join(self.file_dir, 'inputTest/interp.300')
        self.file_xs0K = os.path.join(self.file_dir, 'inputTest/u238.0.2')
        self.composition_file = os.path.join(self.file_dir, 'inputTest/UO2Composition')
        self.structure_file = os.path.join(self.file_dir, 'inputTest/UO2Structure')
        self.atomPos_file = os.path.join(self.file_dir, 'inputTest/UO2AtomPos')

        # Get the data in python:
        self.Eout = np.loadtxt(self.file_Eout)
        self.pdos = Pdos.from_file([self.T], [self.file_pdos])
        self.xs0K = Xs0K.read_xs(self.file_xs0K)
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

    def get_results(self, *command_line_args):
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

        return main(*command_line_args, write_to_file=False)

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
        expected_result = Expected_result.from_method(method, *variables, model=model)

        # Check the results
        expected_result.check_results(self.get_results(*command_line))

    def functionCheck(self, variables: list, method,
                      command_line: list) -> None:
        """
        Test the specified function for the calculation.

        Parameters
        ----------
        variables : list
            The variables for the function.
        method : method
            The method to use for the calculation.
        command_line : list
            The command line arguments.
        """
        # Generate the expected result
        expected_result = Expected_result.from_method(method, *variables)

        # Check the results
        expected_result.check_results(self.get_results(*command_line))


class Expected_result:
    def __init__(self, expected_result):
        self.expected_result = expected_result

    @classmethod
    def from_method(cls, method, *args, **kwargs):
        return cls(method(*args, **kwargs))

    @classmethod
    def from_array(cls, values):
        return cls(values)

    def check_results(self, other):
        if isinstance(other, dict):
            for key, value in other.items():
                expected_value = getattr(self.expected_result, key)
                np.testing.assert_array_equal(expected_value, value)
        else:
            np.testing.assert_array_equal(self.expected_result, other)


class TestScinel_Teff(BaseTestScinel):
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

    @property
    def get_command(self) -> list:
        """
        Get the command line arguments for the calculation.
        """
        return [self.keyword, self.T, self.file_pdos]

    def test_teff(self) -> None:
        """
        Test the effective temperature calculation
        """
        # Generate the expected result

        expected_result = Expected_result.from_array(
            np.array([self.T, self.pdos.fix_T(self.T).Teff])
        )

        # Get the results
        results = self.get_results(*self.get_command)

        # Check the results
        expected_result.check_results(results)


class TestScinel_Sab(BaseTestScinel):
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
        self.modelCheck(model, variables, Sab.from_model,
                        self.get_command(model))

    def test_fgm(self) -> None:
        """
        Test the fgm model for generating S(alpha, -beta) tables.
        """
        self.modelTest("fgm")

    def test_sct(self) -> None:
        """
        Test the sct model for generating S(alpha, -beta) tables.
        """
        self.modelTest('sct')

    def test_pdos(self) -> None:
        """
        Test the pdos model for generating S(alpha, -beta) tables.
        """
        self.modelTest('pdos')

class TestScinel_DynamicStruc(BaseTestScinel):
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        super().setUp()
        self.keyword = 'dsf'

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
        command = [self.keyword, model, self.Ein,
                   self.M, self.T, self.file_Eout, self.file_theta]

        return command if model == 'fgm' else command + [self.file_pdos]

    def modelTest(self, model: str) -> None:
        """
        Test the specified model for the calculation of the dynamic structure
        factor.

        Parameters
        ----------
        model : str
            The model to test. Can be 'fgm', 'sct', or 'pdos'.
        """
        # Determine the appropriate variables based on the model
        variables = self.get_fgm_var if model == 'fgm' else self.get_sct_var

        # Check the results
        self.modelCheck(model, variables, DynamicStruc.from_model,
                        self.get_command(model))

    def test_fgm(self) -> None:
        """
        Test the fgm model for generating Dynamic Structure Factors.
        """
        self.modelTest("fgm")

    def test_sct(self) -> None:
        """
        Test the sct model for generating Dynamic Structure Factors.
        """
        self.modelTest('sct')

    def test_pdos(self) -> None:
        """
        Test the pdos model for generating Dynamic Structure Factors.
        """
        self.modelTest('pdos')


class TestScinel_DDxs(BaseTestScinel):
    """
    Test the DDxs class terminal application in the solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        super().setUp()
        self.keyword = 'ddxs'
        self.xs = Xs0K(self.M, self.xs0K)

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
        command = [self.keyword, model, self.file_xs0K, self.Ein,
                   self.M, self.T, self.file_Eout, self.file_theta]
        return command if model == 'fgm' else command + [self.file_pdos]

    def modelTest(self, model: str) -> None:
        """
        Test the specified model for the calculation of the double differential
        scattering cross section.

        Parameters
        model : str
            The model to test. Can be 'fgm', 'sct', or 'pdos'.

        Returns
        -------
        None
            Pass the test if the results are equal.
        """
        # Determine the appropriate variables based on the model
        variables = self.get_fgm_var if model == 'fgm' else self.get_sct_var

        # Check the results
        self.modelCheck(model, variables, DDxs.from_4PCF,
                        self.get_command(model))

    def test_4pcf_fgm(self) -> None:
        """
        Test the fgm model for the generating S(alpha, -beta) tables.
        """
        self.modelTest("fgm")

    def test_4pcf_sct(self) -> None:
        """
        Test the sct model for the generating S(alpha, -beta) tables.
        """
        self.modelTest('sct')

    def test_4pcf_pdos(self) -> None:
        """
        Test the pdos model for the generating S(alpha, -beta) tables.
        """
        self.modelTest('pdos')


class TestScinel_BraggEdges(BaseTestScinel):
    """
    Test the Bragg Edges calculation in terminal application.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables
        """
        super().setUp()
        self.keyword = 'braggedges'
        self.energyCut = 0.1

    @property
    def get_command(self) -> list:
        """
        Get the command line arguments for the calculation.
        """
        return [self.keyword, self.composition_file, self.structure_file,
                self.atomPos_file, self.energyCut, self.T,
                self.file_pdos, self.file_pdos]

    @property
    def get_solid(self) -> Solid:
        """
        Get the solid object for the calculation.
        """
        # Initialize the solid object:
        solid = Solid.from_files(self.composition_file, self.structure_file,
                                 self.atomPos_file)

        # Introduce the partial density of states in the solid object:
        solid.set_pdos([self.pdos, self.pdos])
        return solid

    def calc_BraggEdges(self) -> np.ndarray:
        """
        Calculate the Bragg Edges information.
        """
        bragg = self.get_solid.get_BraggEdges(self.energyCut, self.T)
        return bragg.reset_index().values

    def test_xscoh(self) -> None:
        """
        Test the coherent cross section calculation
        """
        # Generate the expected result
        expected_result = Expected_result.from_array(self.calc_BraggEdges())

        # Get the results
        results = self.get_results(*self.get_command)

        # Check the results
        expected_result.check_results(results)


class TestScinel_XsCoh(TestScinel_BraggEdges):
    """
    Test the XsCoh class terminal application in the solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables
        """
        super().setUp()
        self.keyword = 'xscoh'

    def calc_xsCoh(self) -> np.ndarray:
        """
        Calculate the coherent cross section.
        """
        xsCoh = self.get_solid.get_XsCoh(self.energyCut, self.T)
        return np.column_stack((xsCoh.index.values, xsCoh.values))

    def test_xscoh(self) -> None:
        """
        Test the coherent cross section calculation
        """
        # Generate the expected result
        expected_result = Expected_result.from_array(self.calc_xsCoh())

        # Get the results
        results = self.get_results(*self.get_command)

        # Check the results
        expected_result.check_results(results)

class TestScinel_Xs(BaseTestScinel):
    """
    Test the DDxs class terminal application in the solid_cinel package.
    """
    def setUp(self) -> None:
        """
        Set up the test common variables.
        """
        super().setUp()
        self.keyword = 'xs'
        self.file_EinGrid = os.path.join(self.file_dir, 'inputTest/EinGrid')


    @property
    def get_var(self) -> list:
        """
        Get the variables for the model.
        """
        xs = Xs0K(self.M, self.xs0K).data
        EinGrid = np.loadtxt(self.file_EinGrid)
        return [xs, EinGrid, self.M, self.T, self.pdos]

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
        return [self.keyword, model, self.file_xs0K, self.file_EinGrid,
                self.M, self.T, self.file_pdos]

    def modelTest(self, model: str) -> None:
        """
        Test the specified model for the calculation of the double differential
        scattering cross section.

        Parameters
        model : str
            The model to test. Can be 'fgm', 'sct', or 'pdos'.

        Returns
        -------
        None
            Pass the test if the results are equal.
        """
        # Check the results
        self.functionCheck(self.get_var, calc_alpha0, self.get_command(model))

    def test_alpha0(self) -> None:
        """
        Test the sct model for the generating S(alpha, -beta) tables.
        """
        self.modelTest('alpha0')


if __name__ == '__main__':
    unittest.main()
