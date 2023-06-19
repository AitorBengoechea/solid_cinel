import numpy as np
import pandas as pd
from typing import Iterable

class Crystal_structure():
    def __init__(self, length: Iterable, angles: Iterable):
        if len(length) != 3:
            ValueError("The direct vector lengths array do not have the apropiate lenght")
        if len(angles) != 3:
            ValueError("The direct vector angles array do not have the apropiate lenght")
        self.length = pd.Series(length,
                                index=["a", "b", "c"],
                                name="direct vectors length")
        self.angles = pd.Series(np.array(angles) * np.pi / 180,
                                index=["alpha", "beta", "gamma"],
                                name="direct vectors angles")

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
        >>> crys = Crystal_structure(dir_vec_length, dir_vec_angles)
        >>> cubic_vec = crys.operator.values

        Test the results:
        >>> assert all(cubic_vec[0].round(6) == np.array([1.      , 0.      , 0.]))
        >>> assert all(cubic_vec[1].round(6) == np.array([0.5     , 0.866025, 0.      ]))
        >>> assert all(cubic_ls
        vec[2].round(6) == np.array([0.5     , 0.288675, 0.816497]))
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
        return pd.DataFrame([a, b, c], index=["a1", "a2", "a3"],
                            columns=["x", "y", "z"])

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
        >>> crys = Crystal_structure(dir_vec_length, dir_vec_angles)
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
        >>> crys = Crystal_structure(dir_vec_length, dir_vec_angles)

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
        >>> crys = Crystal_structure(dir_vec_length, dir_vec_angles)
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
            np.cross(dir_vec.loc["a1"], dir_vec.loc["a2"]),
                               ])
        reci_coeff *= 2 * np.pi / self.unit_cell_vol
        return pd.DataFrame(reci_coeff,
                            index=["b1", "b2", "b3"],
                            columns=dir_vec.index)