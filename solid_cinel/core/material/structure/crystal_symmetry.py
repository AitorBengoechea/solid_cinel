"""
Python file for working with crystal structures.

@author: AB272525
"""
import numpy as np
import pandas as pd
from typing import Iterable
from dataclasses import dataclass, field

@dataclass
class CrystalStructure:
    """
    Class for the crystal structure.

    Attributes
    ----------
    length : pd.Series
        direct vector lengths
    angles : pd.Series
        direct vector angles

    Properties
    ----------
    operator -> pd.DataFrame
        Generate the operator for obteining the direct lattice vectors
    dir_vec -> pd.DataFrame
        Vectors of the direct lattice with the keys ["a1", "a2", "a3"]
    reciproc_vec -> pd.DataFrame
        Vectors of the reciprocal lattice with the keys ["b1", "b2", "b3"]
    unit_cell_vol -> float
        Unit cell volume
    """
    lengthIter: Iterable
    anglesIter: Iterable
    length: pd.Series = field(init=False)
    angles: pd.Series = field(init=False)

    def __post_init__(self) -> "CrystalStructure":
        """
        Initialize the class
        """
        self.get_length(self.lengthIter)
        self.get_angles(self.anglesIter)

    def get_length(self, length: Iterable):
        """
        Get the direct vector lengths.

        Parameters
        ----------
        length : Iterable, (3,)
            direct vector lengths
        """
        if len(length) != 3:
            ValueError("The direct vector lengths array do not have the apropiate lenght")
        self.length = pd.Series(length, index=["a", "b", "c"], name="direct vectors length")

    def get_angles(self, angles: Iterable):
        """
        Get the direct vector angles.

        Parameters
        ----------
        angles : Iterable, (3,)
            direct vector angles in degrees
        """
        if len(angles) != 3:
            ValueError("The direct vector angles array do not have the apropiate lenght")
        self.angles = pd.Series(np.deg2rad(angles), index=["alpha", "beta", "gamma"], name="direct vectors angles")

    @property
    def operator(self) -> pd.DataFrame:
        """
        Generate the operator for obteining the direct lattice vectors.

        Returns
        -------
        "pd.DataFrame"
             The operator for obteining the direct lattice vectors

        Example
        -------
        Object initialization:
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> crys = CrystalStructure(dir_vec_length, dir_vec_angles)
        >>> cubic_vec = crys.operator.values

        Test the results:
        >>> assert all(cubic_vec[0].round(6) == np.array([1.      , 0.      , 0.]))
        >>> assert all(cubic_vec[1].round(6) == np.array([0.5     , 0.866025, 0.      ]))
        >>> assert all(cubic_vec[2].round(6) == np.array([0.5     , 0.288675, 0.816497]))
        """
        angles = self.angles
        a = np.array([1.,
                      0.,
                      0.])
        b = np.array([np.cos(angles["gamma"]),
                      np.sin(angles["gamma"]),
                      0.])
        c = np.array([np.cos(angles["beta"]),
                      np.cos(angles["alpha"]) - np.cos(angles["beta"]) * np.cos(angles["gamma"]),
                      1.0])
        c[1] /= np.sin(angles["gamma"])
        c[2] *= np.sqrt(1. - c[0] ** 2 - c[1] ** 2)
        return pd.DataFrame([a, b, c], index=["a1", "a2", "a3"], columns=["x", "y", "z"])

    @property
    def dir_vec(self) -> pd.DataFrame:
        """
        Vectors of the direct lattice with the keys ["a1", "a2", "a3"].

        Returns
        -------
        "pd.DataFrame"
            Direct lattice vectors.

        Example
        -------
        Object initialization:
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> crys = CrystalStructure(dir_vec_length, dir_vec_angles)
        >>> direct_vectors = crys.dir_vec

        Test the results:
        >>> assert all(direct_vectors.loc["a1"].values.round(6) == np.array([2.856711, 0.      , 0.      ]))
        >>> assert all(direct_vectors.loc["a2"].values.round(6) == np.array([1.428355, 2.473984, 0.      ]))
        >>> assert all(direct_vectors.loc["a3"].values.round(6) == np.array([1.428355, 0.824661, 2.332494]))

        """
        return self.operator * self.length.values

    @property
    def unit_cell_vol(self) -> float:
        """
        Unit cell volume.

        Returns
        -------
        "float"
            Unit cell volume

        Example
        -------
        Object initialization:
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> crys = CrystalStructure(dir_vec_length, dir_vec_angles)

        Test the results:
        >>> assert crys.unit_cell_vol.round(6) == 16.484804
        """
        vec = self.dir_vec
        return np.dot(vec.loc["a1"], np.cross(vec.loc["a2"], vec.loc["a3"]))

    @property
    def reciproc_vec(self) -> pd.DataFrame:
        """
        Lattice reciprocal vectors.

        Returns
        -------
        "pd.DataFrame"
            Reciprocal lattice vectors

        Example
        -------
        Object initialization:
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> crys = CrystalStructure(dir_vec_length, dir_vec_angles)
        >>> reciprocal_vector = crys.reciproc_vec

        Test the results:
        >>> assert all(reciprocal_vector.loc["b1"].values.round(6) == np.array([ 2.199448, -1.269852, -0.897921]))
        >>> assert all(reciprocal_vector.loc["b2"].values.round(6) == np.array([ 0.      ,  2.539703, -0.897921]))
        >>> assert all(reciprocal_vector.loc["b3"].values.round(6) == np.array([0.      , 0.      , 2.693762]))
        """
        dir_vec = self.dir_vec
        reci_coeff = np.array([
                                np.cross(dir_vec.loc["a2"], dir_vec.loc["a3"]),
                                np.cross(dir_vec.loc["a3"], dir_vec.loc["a1"]),
                                np.cross(dir_vec.loc["a1"], dir_vec.loc["a2"])])
        reci_coeff *= 2 * np.pi / self.unit_cell_vol
        return pd.DataFrame(reci_coeff, index=["b1", "b2", "b3"], columns=dir_vec.index)

    @classmethod
    def from_file(cls, file_path: str) -> "CrystalStructure":
        """
        Create a CrystalStructure object from a file.

        Parameters
        ----------
        file_path : "str"
            Path to the file with the crystal structure information.

        Returns
        -------
        "CrystalStructure"
            CrystalStructure object.

        Example
        -------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule:
        >>> file_path = os.path.join(file_dir, '../../../data/materials/Al27/Al27UnitCell')
        >>> unitCell = CrystalStructure.from_file(file_path)
        >>> unitCell.length.round(6)
        a    2.856711
        b    2.856711
        c    2.856711
        Name: direct vectors length, dtype: float64

        >>> unitCell.angles.round(6)
        alpha    1.047198
        beta     1.047198
        gamma    1.047198
        Name: direct vectors angles, dtype: float64
        """
        # Load the data from the file
        UnitCell = np.loadtxt(file_path)
        if UnitCell.shape != (2, 3):
            ValueError("The file do not have the apropiate lenght")

        # Create the object:
        return cls(UnitCell[0], UnitCell[1])

    @property
    def to_string(self) -> str:
        """
        Return a string with the crystal structure information.

        Returns
        -------
        "str"
            String with the crystal structure information.

        Example
        -------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule:
        >>> file_path = os.path.join(file_dir, '../../../data/materials/Al27/Al27UnitCell')
        >>> unitCell = CrystalStructure.from_file(file_path)
        >>> print(unitCell.to_string)
        # Unit cell information:
        # Direct vector length: (a, b, c)
        2.856710674519725 2.856710674519725 2.856710674519725
        # Direct vector angles: (alpha, beta, gamma)
        60.0 60.0 60.0
        """
        info_str = "\n".join([
            f"# Unit cell information:",
            f"# Direct vector length: (a, b, c)",
            f"{self.lengthIter[0]} {self.lengthIter[1]} {self.lengthIter[2]}",
            f"# Direct vector angles: (alpha, beta, gamma)",
            f"{self.anglesIter[0]} {self.anglesIter[1]} {self.anglesIter[2]}",
        ])
        return info_str

    def to_file(self, filename: str) -> None:
        """
        Write the crystal structure information to a file.

        Parameters
        ----------
        filename : "str"
            Path to the file where the information will be written.

        Example
        -------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule:
        >>> file_path = os.path.join(file_dir, '../../../data/materials/Al27/Al27UnitCell')
        >>> unitCell = CrystalStructure.from_file(file_path)
        >>> unitCell.to_file("Al27UnitCell")
        >>> unitCellWritten = CrystalStructure.from_file("Al27UnitCell")

        # Test the results:
        >>> assert unitCell.to_string == unitCellWritten.to_string

        # Remove the file after the test:
        >>> os.remove("Al27UnitCell")
        """
        # Open the file in write mode and write the string
        with open(filename, 'w') as file:
            file.write(self.to_string)


