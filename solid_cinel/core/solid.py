# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:24:18 2022

@author: AB272525
"""

from solid_cinel.data import elements
import numpy as np
import pandas as pd
import collections
import pytest
collections.Callable = collections.abc.Callable


class Crys_atom():
    """Class to store the properties of materials."""

    def __init__(self, A, Z, dir_vec_length, dir_vec_angles,
                 preferred_orientation, unit_pos, atom_mass, b_coh, b_incoh):
        """
        Initialize the crystaline structure formed by a single atom.

        Parameters
        ----------
        A : ´int´
            Atomic number.
        Z : ´int´
            Number of protons.
        dir_vec_length : iterable or `np.array` of size (1, 3)
            Direct lattice vectors length in fm.
        preferred_orientation : iterable or `np.array` of size (1, 3)
            Direct lattice vectors angles in ª.
        preferred_orientation : iterable or `np.array` of size (1, 3)
            Preferred orientation of the target.
        unit_pos : 1D iterable
            Unitary positions of atoms in the lattice unit cell.
        atom_mass : float
            Atom mass, amu.
        b_coh : float
            Bound coherent scattering length (fm).
        b_incoh : float
            Bound incoherent scattering length (fm).

        Returns
        -------
        ´Crys_atom´
            Class to store the properties and method related with a crystaline
            structure formed by a single atom.

        """
        if len(dir_vec_length) != 3:
            ValueError("The direct vector lengths array do not have the apropiate lenght")
        if len(dir_vec_angles) != 3:
            ValueError("The direct vector angles array do not have the apropiate lenght")
        if len(preferred_orientation) != 3:
            ValueError("The preferential orientation array do not have the apropiate lenght")
        self.A = A
        self.Z = Z
        self.preferred_orientation = pd.Series(preferred_orientation,
                                               index=["x", "y", "z"],
                                               name="preferred orientation")
        self.dir_vec_length = pd.Series(dir_vec_length,
                                        index=["a", "b", "c"],
                                        name="direct vectors length")
        self.dir_vec_angles = pd.Series(np.array(dir_vec_angles) * np.pi / 180,
                                        index=["alpha", "beta", "gamma"],
                                        name="direct vectors angles")
        self.unit_pos = np.array(unit_pos).reshape(-1, 3)
        self.atom_mass = atom_mass
        self.b = {"b_coh": b_coh, "b_incoh": b_incoh}

    @property
    def material_name(self) -> str:
        """
        Material name: element + A

        Example
        -------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 1, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.25, 0.25,0.75, 0.25])
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Crys_atom(A, Z, dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

        Test the results:
        >>> assert Al.material_name == "Al27"

        """
        return elements.ELEMENTS[self.Z] + str(self.A)

    @property
    def dir_vec(self) -> pd.Series:
        """
        Vectors of the direct lattice with the keys ["a1", "a2", "a3"].

        Example
        -------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 1, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.25, 0.25,0.75, 0.25])
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Crys_atom(A, Z, dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)
        >>> direct_vectors = Al.dir_vec

        Test the results:
        >>> assert all(direct_vectors["a1"].round(6) == np.array([2.856711, 0.      , 0.      ]))
        >>> assert all(direct_vectors["a2"].round(6) == np.array([1.428355, 2.473984, 0.      ]))
        >>> assert all(direct_vectors["a3"].round(6) == np.array([1.428355, 0.824661, 2.332494]))
        """
        a = np.array([1., 0., 0.])
        b = np.array([np.cos(self.dir_vec_angles["gamma"]),
                      np.sin(self.dir_vec_angles["gamma"]),
                      0.])
        c = np.array([np.cos(self.dir_vec_angles["beta"]),
                     np.cos(self.dir_vec_angles["alpha"]) - np.cos(self.dir_vec_angles["beta"]) * np.cos(self.dir_vec_angles["gamma"]),
                      1])
        c[1] /=  np.sin(self.dir_vec_angles["gamma"])
        c[2] *= np.sqrt(1. - c[0] ** 2 - c[1] ** 2)

        # Length multiplication:
        a *= self.dir_vec_length["a"]
        b *= self.dir_vec_length["b"]
        c *= self.dir_vec_length["c"]

        # Final result:
        return pd.Series([a, b, c], index=["a1", "a2", "a3"],
                         name="Lattice direct vectors")

    @property
    def unit_cell_vol(self) -> float:
        """
        Unit cell volume.

        Example
        -------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 1, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.25, 0.25,0.75, 0.25])
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Crys_atom(A, Z, dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

        Test the results:
        >>> assert Al.unit_cell_vol.round(6) == 16.484804
        """
        vec = self.dir_vec
        return np.dot(vec["a1"], np.cross(vec["a2"], vec["a3"]))

    @property
    def reciproc_vec(self) -> pd.Series:
        """
        Lattice reciprocal vectors.

        Example
        -------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 1, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.25, 0.25,0.75, 0.25])
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Crys_atom(A, Z, dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)
        >>> reciprocal_vector = Al.reciproc_vec

        Test the results:
        >>> assert all(reciprocal_vector["b1"].round(6) == np.array([ 2.199448, -1.269852, -0.897921]))
        >>> assert all(reciprocal_vector["b2"].round(6) == np.array([ 0.      ,  2.539703, -0.897921]))
        >>> assert all(reciprocal_vector["b3"].round(6) == np.array([0.      , 0.      , 2.693762]))
        """
        dir_vec = self.dir_vec
        reci_coeff = np.array([
            np.cross(dir_vec["a2"], dir_vec["a3"]),
            np.cross(dir_vec["a3"], dir_vec["a1"]),
            np.cross(dir_vec["a1"], dir_vec["a2"]),
                               ])
        reci_coeff *= 2 * np.pi / self.unit_cell_vol
        return pd.Series(list(reci_coeff), index=["b1", "b2", "b3"],
                         name="Lattice Reciprocal vectors")

    @property
    def atom_pos(self) -> np.array:
        """
        Position of atoms in the direct lattice

        Example
        -------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 1, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.25, 0.25,0.75, 0.25])
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Crys_atom(A, Z, dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

        Test the results:
        >>> Al.atom_pos.round(6)
        array([[1.428355, 0.824661, 0.583124],
               [2.856711, 0.824661, 0.583124],
               [2.856711, 2.473984, 1.749371],
               [4.285066, 2.473984, 1.749371],
               [3.570888, 1.236992, 1.749371],
               [2.142533, 1.236992, 1.749371],
               [3.570888, 2.061653, 0.583124],
               [2.142533, 2.061653, 0.583124]])

        """
        return np.stack(self.unit_pos.dot(self.dir_vec.values))

    @property
    def atom_number(self) -> int:
        """
        The number of atoms in the unit cell.

        Example
        -------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 1, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.25, 0.25,0.75, 0.25])
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Crys_atom(A, Z, dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

        Test the results:
        >>> assert Al.atom_number == 8
        """
        return self.unit_pos.shape[0]