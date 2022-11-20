# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:24:18 2022

@author: AB272525
"""

from solid_cinel.core.material.material_composition import *
from solid_cinel.core.material.crystal_symmetry import dir_vector_operator
import numpy as np
import pandas as pd
import collections
import pytest
collections.Callable = collections.abc.Callable


class Solid(Molecule):
    """Class to store the properties and methods for solid materials."""

    def __init__(self, dir_vec_length, dir_vec_angles,
                 preferred_orientation, unit_pos,
                 *args, sym="cubic", **kwargs):
        """
        Initialize the crystaline structure formed by a single atom.

        Parameters
        ----------
        dir_vec_length : iterable or `np.array` of size (1, 3)
            Direct lattice vectors length in fm.
        preferred_orientation : iterable or `np.array` of size (1, 3)
            Direct lattice vectors angles in ª.
        preferred_orientation : iterable or `np.array` of size (1, 3)
            Preferred orientation of the target.
        unit_pos : 'dict' or 1D iterable
            Unitary positions of atoms in the lattice unit cell. Two options:
            Single atom: 1D iterable
            Multiple atoms: 'dict'
                {"atom name" : [atom positions]}
        sym : 'str', optional
            Symmetry of the solid, by default "cubic"

        Parameters for Molecule
        -----------------------
        A : 1D iterable of 'int' or 'int'
            Atomic number.
        Z : 1D iterable of 'int' or 'int'
            Number of protons.
        atom_mass : 1D iterable of 'float' or 'float'
            Atom mass, amu.
        b_coh : 1D iterable of 'float' or 'float'
            Bound coherent scattering length (fm).
        b_incoh : 1D iterable of 'float' or 'float'
            Bound incoherent scattering length (fm).
        name : 'str', optional
            Name of the molecule, by default None

        Returns
        -------
        ´Solid´
            Class to store the properties and method related with a crystaline
            structure of a solid.

        """
        super().__init__(*args, **kwargs)

        if len(dir_vec_length) != 3:
            ValueError("The direct vector lengths array do not have the apropiate lenght")
        if len(dir_vec_angles) != 3:
            ValueError("The direct vector angles array do not have the apropiate lenght")
        if len(preferred_orientation) != 3:
            ValueError("The preferential orientation array do not have the apropiate lenght")
        self.preferred_orientation = pd.Series(preferred_orientation,
                                               index=["x", "y", "z"],
                                               name="preferred orientation")
        self.dir_vec_length = pd.Series(dir_vec_length,
                                        index=["a", "b", "c"],
                                        name="direct vectors length")
        self.dir_vec_angles = pd.Series(np.array(dir_vec_angles) * np.pi / 180,
                                        index=["alpha", "beta", "gamma"],
                                        name="direct vectors angles")
        self.unit_pos = unit_pos
        self.sym = sym

    @property
    def unit_pos(self) -> dict:
        return self._unit_pos

    @unit_pos.setter
    def unit_pos(self, unit_pos):
        if isinstance(unit_pos, dict):
            self._unit_pos = {element: single_unit_pos for element, single_unit_pos in unit_pos.items()}
        else:
            self._unit_pos = {self.name: np.array(unit_pos).reshape(-1, 3)}

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
        >>> Al = Solid(dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)
        >>> direct_vectors = Al.dir_vec

        Test the results:
        >>> assert all(direct_vectors.loc["a1"].values.round(6) == np.array([2.856711, 0.      , 0.      ]))
        >>> assert all(direct_vectors.loc["a2"].values.round(6) == np.array([1.428355, 2.473984, 0.      ]))
        >>> assert all(direct_vectors.loc["a3"].values.round(6) == np.array([1.428355, 0.824661, 2.332494]))
        """
        operator = dir_vector_operator(self.dir_vec_angles, self.sym)
        return operator * self.dir_vec_length.values

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
        >>> Al = Solid(dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

        Test the results:
        >>> assert Al.unit_cell_vol.round(6) == 16.484804
        """
        vec = self.dir_vec
        return np.dot(vec.loc["a1"], np.cross(vec.loc["a2"], vec.loc["a3"]))

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
        >>> Al = Solid(dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)
        >>> reciprocal_vector = Al.reciproc_vec

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
        >>> Al = Solid(dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

        Test the results:
        >>> Al.atom_pos["Al27"].round(6)
        array([[1.428355, 0.824661, 0.583124],
               [2.856711, 0.824661, 0.583124],
               [2.856711, 2.473984, 1.749371],
               [4.285066, 2.473984, 1.749371],
               [3.570888, 1.236992, 1.749371],
               [2.142533, 1.236992, 1.749371],
               [3.570888, 2.061653, 0.583124],
               [2.142533, 2.061653, 0.583124]])

        """
        return {element: np.stack(unit_pos.dot(self.dir_vec.values))
                for element, unit_pos in self.unit_pos.items()}

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
        >>> Al = Solid(dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

        Test the results:
        >>> assert Al.atom_number["Al27"] == 8
        """
        return {element: atom_pos.shape[0]
                for element, atom_pos in self.atom_pos.items()}

    def get_d_min(self, wavelength) -> float:
        """
        The minimum dspacing for the LEAPR module of NJOY

        Parameters
        ----------
        wavelength : 'float'
            Incident neutron wavelength in Anstrom.

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
        >>> Al = Solid(dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)
        
        Test the results:
        >>> wavelength = 0.18855129477888757
        >>> round(Al.get_d_min(wavelength), 6)
        0.089562
        """
        return 0.5 * wavelength * 0.95