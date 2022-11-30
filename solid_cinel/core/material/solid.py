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

# Example variables:
# 1 atom:
preferred_orientation_Al27 = np.array([0, 1, 1])
a_Al27 = 2.856710674519725
dir_vec_length_Al27 = [a_Al27, a_Al27, a_Al27]
dir_vec_angles_Al27 = [60, 60, 60]
unit_pos_Al27 = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.25, 0.25, 0.75, 0.25])
A_Al27 = 27
Z_Al27 = 13
atomic_mass_Al27 = 26.98153433356103
b_coh_Al27 = 3.449
b_incoh_Al27 = 0.256

# 2 atom:
preferred_orientation = np.array([0, 1, 1])
unit_pos_U_str = '''
0.500000  0.000000  0.000000
0.500000  0.500000  0.500000
0.000000  0.000000  0.500000
0.000000  0.500000  0.000000'''
unit_pos_U = np.fromstring(unit_pos_U_str, dtype=np.float64, sep=' ')\
               .reshape(-1, 3)
unit_pos_O_str = '''
0.250000  0.250000  0.250000
0.750000  0.250000  0.250000
0.250000  0.750000  0.750000
0.750000  0.750000  0.750000
0.750000  0.250000  0.750000
0.250000  0.250000  0.750000
0.750000  0.750000  0.250000
0.250000  0.750000  0.250000'''
unit_pos_O = np.fromstring(unit_pos_O_str, dtype=np.float64, sep=' ')\
               .reshape(-1, 3)
unit_pos = {"O16": unit_pos_O, "U238": unit_pos_U}
a = 5.54781
dir_vec_length = [a, a, a]
dir_vec_angles = [90, 90, 90]
energy_sup = 5.  # eV
energy_cut = 6.85e-1
A = [16, 238]
Z = [8, 92]
atom_mass = [15.99491399021626, 238.05077040419212]
b_coh = [5.878374042670532, 8.62912188811068]
b_incoh = [0.0, 0.19947114020071632]

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
    def unit_pos(self) -> pd.Series:
        """
        Pandas series containnig atom position in a unit cell for each atom

        Example
        -------
        Object initialization:
        >>> Al = Solid(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)
        >>> UO2 = Solid(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atom_mass, b_coh, b_incoh)

        Test the results:
        >>> Al.unit_pos.loc["Al27"]
              x     y     z
        0  0.25  0.25  0.25
        1  0.75  0.25  0.25
        2  0.25  0.75  0.75
        3  0.75  0.75  0.75
        4  0.75  0.25  0.75
        5  0.25  0.25  0.75
        6  0.75  0.75  0.25
        7  0.25  0.75  0.25

        >>> UO2.unit_pos.loc["O16"]
              x     y     z
        0  0.25  0.25  0.25
        1  0.75  0.25  0.25
        2  0.25  0.75  0.75
        3  0.75  0.75  0.75
        4  0.75  0.25  0.75
        5  0.25  0.25  0.75
        6  0.75  0.75  0.25
        7  0.25  0.75  0.25

        >>> UO2.unit_pos.loc["U238"]
             x    y    z
        0  0.5  0.0  0.0
        1  0.5  0.5  0.5
        2  0.0  0.0  0.5
        3  0.0  0.5  0.0
        """
        return self._unit_pos

    @unit_pos.setter
    def unit_pos(self, unit_pos):
        col = pd.Index(["x", "y", "z"])
        if isinstance(unit_pos, dict):
            _unit_pos = {element: pd.DataFrame(single_unit_pos.reshape(-1, 3), columns=col)
                         for element, single_unit_pos in unit_pos.items()}
        else:
            _unit_pos = {self.name: pd.DataFrame(np.array(unit_pos).reshape(-1, 3),
                                                 columns=col)}
        self._unit_pos = pd.Series(_unit_pos)

    @property
    def atom_pos(self) -> np.ndarray:
        """
        Position of atoms in the direct lattice

        Example
        -------
        Object initialization:
        >>> Al = Solid(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)
        >>> UO2 = Solid(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atom_mass, b_coh, b_incoh)

        Test the results:
        >>> Al.atom_pos["Al27"].round(6)
                   x      	y	       z
        0	1.428355	 0.824661	0.583124
        1	2.856711	 0.824661	0.583124
        2	2.856711	 2.473984	1.749371
        3	4.285066	 2.473984	1.749371
        4	3.570888	 1.236992	1.749371
        5	2.142533	 1.236992	1.749371
        6	3.570888	 2.061653	0.583124
        7	2.142533	 2.061653	0.583124

        >>> UO2.atom_pos.loc["O16"].round(6)
                  x         y         z
        0  1.386952  1.386952  1.386952
        1  4.160858  1.386952  1.386952
        2  1.386953  4.160858  4.160858
        3  4.160858  4.160858  4.160858
        4  4.160858  1.386953  4.160858
        5  1.386953  1.386953  4.160858
        6  4.160858  4.160858  1.386952
        7  1.386953  4.160858  1.386952

        >>> UO2.atom_pos.loc["U238"].round(6)
                  x         y         z
        0  2.773905  0.000000  0.000000
        1  2.773905  2.773905  2.773905
        2  0.000000  0.000000  2.773905
        3  0.000000  2.773905  0.000000
        """
        return self.unit_pos.apply(lambda x: pd.DataFrame(x.values.dot(self.dir_vec),
                                                          columns=x.columns))

    @property
    def atom_number(self) -> int:
        """
        The number of atoms in the unit cell.

        Example
        -------
        Object initialization:
        >>> Al = Solid(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)
        >>> UO2 = Solid(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atom_mass, b_coh, b_incoh)

        Test the results:
        >>> assert Al.atom_number == 8
        >>> assert UO2.atom_number == 12
        """
        return sum(self.atom_pos.apply(lambda x: x.shape[0]).values)


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
    >>> crys = Crystal_structure(dir_vec_length_Al27, dir_vec_angles_Al27)
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
