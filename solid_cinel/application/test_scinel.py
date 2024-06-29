import unittest
import os
import numpy as np
from solid_cinel.application.scinel import main


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
        expected_result = np.array([T, 317.01138912013226])
        self.check_results(expected_result, keyword, str(T), file)

if __name__ == '__main__':
    unittest.main()
