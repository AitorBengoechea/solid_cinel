# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:46:42 2022

@author: Aitor Bengoechea
"""

from solid_cinel.data import elements
import numpy as np
import pandas as pd
import collections
collections.Callable = collections.abc.Callable
import pytest


class Crys_atom():
    """Class to store the properties of materials."""

    def __init__(self, temperature, A, Z, dir_vec_length, dir_vec_angles,
                 preferred_orientation=[0, 0, 1]):
        """
        Initialize the Target class.

        Parameters
        ----------
        temperature : `int`
            Temperature in Kelvin.
        A : ´int´
            Atomic number.
        Z : ´int´
            Number of protons.
        preferred_orientation : iterable or `np.array` of size (1, 3)
            Preferred orientation of the target.

        Returns
        -------
        ´Target´
            Class to store the properties of materials.

        """
        self.T = temperature
        self.A = A
        self.Z = Z
        self.material_name = elements.ELEMENTS[Z] + str(A)
        self.preferred_orientation = preferred_orientation
        self.dir_vec_length = pd.Series(dir_vec_length, index=["a",
                                                               "b",
                                                               "c"],
                                        name="direct vectors length")
        self.dir_vec_angles = pd.Series(dir_vec_angles, index=["$\alpha$",
                                                               "$\beta$",
                                                               "$\gamma$"],
                                        name="direct vectors angles")

    @property
    def preferred_orientation(self) -> pd.Series:
        """
        Preferred orientation of the target material.

        Returns
        -------
        ´pd.Series´
            Array with the preferred orientation. [x, y , z]

        Examples
        --------
        >>> temperature, A, Z = 300, 27, 13
        >>> dir_vec_length, dir_vec_angles = [1, 1, 1], [60, 60, 60]
        >>> test = Crys_atom(temperature, A, Z, dir_vec_length, dir_vec_angles, preferred_orientation=[0, 1, 1])
        >>> test.preferred_orientation
        x    0
        y    1
        z    1
        Name: preferred orientation, dtype: int64

        >>> Crys_atom(temperature, A, Z, dir_vec_length, dir_vec_angles).preferred_orientation
        x    0
        y    0
        z    1
        Name: preferred orientation, dtype: int64

        >>> with pytest.raises(ValueError): Crys_atom(temperature, A, Z, dir_vec_length, dir_vec_angles, preferred_orientation=[0, 1])
        """
        return self._preferred_orientation

    @preferred_orientation.setter
    def preferred_orientation(self, array) -> pd.Series:
        if len(array) == 3:
            self._preferred_orientation = pd.Series(array,
                                                    index=["x", "y", "z"],
                                                    name="preferred orientation")
        else:
            raise ValueError("The preferential orientation array do not have the apropiate lenght")
