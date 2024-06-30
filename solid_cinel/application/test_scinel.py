import unittest
import os
import numpy as np

# Application test:
from solid_cinel.application.scinel import main

# POO direct application:
from solid_cinel.core.material.vibration.pdos import Pdos


class TestScinel(unittest.TestCase):

    def check_results(self, expected_result, *command_line_args):
        # Create a parser to simulate the command line arguments
        result = main(*command_line_args, write_to_file=False)

        # Check the returned result
        np.testing.assert_array_equal(result, expected_result)

    def test_Teff(self):
        # Test the calculation of the effective temperature
        T = 300
        keyword = 'Teff'
        file_dir = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(file_dir, '../data/pdos/interp.300')

        # Create Pdos object for the test:
        pdos = Pdos.from_file([T], [file])

        # Generate the expected result:
        expected_result = np.array([T, pdos.fix_T(T).Teff])

        # Check the results
        self.check_results(expected_result, keyword, str(T), file)

if __name__ == '__main__':
    unittest.main()
