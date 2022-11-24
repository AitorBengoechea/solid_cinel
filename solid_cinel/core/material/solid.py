# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:24:18 2022

@author: AB272525
"""

from solid_cinel.core.material.material_composition import Molecule
from solid_cinel.core.material.crystal_symmetry import Crystal_structure
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import collections
import pytest
collections.Callable = collections.abc.Callable


class Solid(Crystal_structure, Molecule):
    """Class to store the properties and methods for solid materials."""

    def __init__(self, preferred_orientation, unit_pos,
                 *args, **kwargs):
        """
        Initialize the crystaline structure formed by a single atom.

        Parameters
        ----------
        preferred_orientation : iterable or `np.array` of size (1, 3)
            Preferred orientation of the target.
        unit_pos : 'dict' or 1D iterable
            Unitary positions of atoms in the lattice unit cell. Two options:
            Single atom: 1D iterable
            Multiple atoms: 'dict'
                {"atom name" : [atom positions]}

        Parameters for Crystal_structure
        --------------------------------
        length : iterable or `np.array` of size (1, 3)
            Direct lattice vectors length in fm.
        angles: iterable or `np.array` of size (1, 3)
            Direct lattice vectors angles in ª.
        symmetry : 'str', optional
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
        Crystal_structure.__init__(self, *args[0:2], kwargs.pop("symmetry", "cubic"))
        Molecule.__init__(self, *args[2:], **kwargs)

        if len(preferred_orientation) != 3:
            ValueError("The preferential orientation array do not have the apropiate lenght")
        self.preferred_orientation = pd.Series(preferred_orientation,
                                               index=["x", "y", "z"],
                                               name="preferred orientation")
        self.unit_pos = unit_pos

    @property
    def unit_pos(self) -> dict:
        return self._unit_pos

    @unit_pos.setter
    def unit_pos(self, unit_pos):
        if isinstance(unit_pos, dict):
            _unit_pos = {element: single_unit_pos for element, single_unit_pos in unit_pos.items()}
        else:
            _unit_pos = {self.name: np.array(unit_pos).reshape(-1, 3)}
        self._unit_pos = pd.Series(_unit_pos)

    @property
    def atom_pos(self) -> np.ndarray:
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
        >>> Al = Solid(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

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
        return self.unit_pos.apply(lambda x: np.stack(x.dot(self.dir_vec.values)))

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
        >>> Al = Solid(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

        Test the results:
        >>> assert Al.atom_number["Al27"] == 8
        """
        return self.atom_pos.apply(lambda x: x.shape[0])


def hkl_max_value(rec_vecs, d_min, precision=1.0e-7) -> np.ndarray:
    """
    Get the maximun h, k and l integers for the constrain of d > d_min.

    Parameters
    ----------
    rec_vecs : 2D 'np.ndarray'
        Reciprocal vectors.
    d_min : 'float'
        The minimun d space.

    Example
    -------
    Object initialization:
    >>> a = 2.856710674519725
    >>> dir_vec_length = [a, a, a]
    >>> dir_vec_angles = [60, 60, 60]
    >>> crys = Crystal_structure(dir_vec_length, dir_vec_angles)
    >>> rec_vecs = crys.reciproc_vec.values
    >>> d_min = 0.2360746677309732
    >>> hkl_max_value(rec_vecs, d_min)
    array([12, 12, 12])
    """
    def hkl_range_minimization(x, i):
        return x[i]
    def constrain(x, d_min):
        vec_tau_hkl = x[0] * rec_vecs[0] + x[1] * rec_vecs[1] + x[2] * rec_vecs[2]
        vec_tau_hkl_norm = np.linalg.norm(vec_tau_hkl)                
        d_hkl = 2 * np.pi / vec_tau_hkl_norm
        return d_hkl - d_min
    result = []
    for i in range(3):
        result.append(minimize(lambda x: hkl_range_minimization(x, i),
                                    [-100, -100, -100],
                                    method='COBYLA',
                                    constraints=({'type': 'ineq', 'fun': constrain, 'args': [d_min]})).x)
    confidance = np.array(list(map(lambda x: constrain(x, d_min), result)))
    hkl_max = abs(np.array(result).diagonal().astype(int))
    return np.where(abs(confidance) <= precision,  hkl_max, 100)  # [h_max, k_max, l_max]
