# test_scinel.py
import unittest
import argparse
import numpy as np
from unittest.mock import patch
from solid_cinel.application.teff import handle_TeffArgs, add_TeffArgs

class TestScinel(unittest.TestCase):
    @patch('solid_cinel.application.scinel.handle_TeffArgs')
    def test_Teff(self, mock_handle_TeffArgs):
        T = 300
        file = '../data/pdos/interp.300'
        expected_result = np.array([T, 317.01138912013226])

        # Create a parser to simulate the command line arguments
        parser = argparse.ArgumentParser()

        # Add the necessary arguments to the parser
        add_TeffArgs(parser)

        # Simulate the command line arguments
        args = parser.parse_args([str(T), file])

        # Mock the output of handle_TeffArgs
        mock_handle_TeffArgs.return_value = expected_result

        # Call handle_TeffArgs directly
        result = handle_TeffArgs(args)

        # Check that handle_TeffArgs was called with the correct arguments
        mock_handle_TeffArgs.assert_called_once_with(args)

        # Check the returned result
        np.testing.assert_array_equal(result, expected_result)

if __name__ == '__main__':
    unittest.main()
