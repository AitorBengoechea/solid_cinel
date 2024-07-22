import unittest
import importlib
from solid_cinel.core import Solid, Pdos

class SolidTestBase(unittest.TestCase):
    def setUp(self):
        if self.solid_name is None:
            raise ValueError("solid_name must be defined in the subclass")

        # Dynamically import the module for the current solid
        self.solid_module = importlib.import_module(f"solids.{self.solid_name}")

        # Example of setting up a Solid object using imported variables
        # Assuming you have a method to create these file paths
        composition_file, structure_file, atomPos_file = self.get_file_paths(self.solid_name)
        self.solid = Solid.from_files(composition_file, structure_file, atomPos_file)
        self.pdos = Pdos.from_dE(self.solid_module.rho_in_energy, self.solid_module.interv_in_energy)
        self.solid.set_pdos(self.pdos)

    def get_file_paths(self, solid_name):
        # Implement this method based on your project's structure
        # Return paths for composition_file, structure_file, atomPos_file
        pass

# Subclass for each solid
class Al27Test(SolidTestBase):
    def setUp(self) -> None:
        """
        Set up the test common variables
        """
        self.solid_name = 'Al27'
        super().setUp()
        self.T = [20, 80, 293.6, 400, 600, 800]

    def test_BraggEdges(self):
        # Implement your test using self.solid and variables from self.solid_module
        pass


if __name__ == '__main__':
    unittest.main()