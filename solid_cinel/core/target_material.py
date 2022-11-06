# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:46:42 2022

@author: Aitor Bengoechea
"""

from solid_cinel.core.solid import Solid
from solid_cinel.core.pdos import Pdos
from scipy.constants import physical_constants as const
import scipy as sp
import numpy as np
import pandas as pd
import collections
collections.Callable = collections.abc.Callable
import pytest

# Examples variables:
rho_in_energy_str = '''
    0 .0066 .0264 .0594 .1055 .1649 .2374 .3232 .4221
    .5342 .6595 .7980 .9497 1.1146 1.2927 1.4839 1.6884
    2.0169 2.4373 2.9366 3.6133 4.6775 7.1346 7.3650
    7.5156 7.6733 7.8309 8.0740 8.4419 9.0595 9.6773
    7.3645 6.2674 5.1965 4.7958 4.8024 4.6841 4.4673
    4.1914 3.8169 3.3439 2.7855 3.2782 5.3082 8.5930
    12.3377 8.4616 5.6695 4.1585 2.6081 0.0
'''
rho_in_energy = np.fromstring(rho_in_energy_str, dtype=np.float64, sep=' ')
interv_in_energy = 0.0008


class Target_mat(Solid, Pdos):
    """Class to store all the Target material methods and atributtes."""

    def __init__(self, *args):
        """
        Class to store all the Target material methods and atributtes.

        Parameters for Crys_atom
        ------------------------
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

        Parameters for pdos
        ------------------------
        rho : list of 1D iterable
            rho values for each element.
        interval_energy : list of 'float'
            Energy interval in eV for each element.
        """
        Solid.__init__(self, *args[0:9])
        # Avoid data setter in Pdos:
        if isinstance(args[10], float):
            atom_pdos = Pdos(args[9],
                      index=pd.Index(np.arange(len(args[9])) * args[10],
                                     name="E"))
            self.pdos[self.name] = atom_pdos
        Pdos.__init__(self, args[9],
                      index=pd.Index(np.arange(len(args[9])) * args[10],
                                     name="E"))

    def B(self, T, anstrom=True) -> float:
        """
        Calculate mean square displacement for a certain pdos information.

        Parameters
        ----------
        T : 'int'
            Temperature in K
        anstrom : 'bool', optional
            Option to obtain the B unit in A^2. The default is True.

        Examples
        --------
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
        >>> Al = Target_mat(dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy, interv_in_energy)

        Test the results:
        >>> T = 20
        >>> Al.B(T)["Al27"].round(6)
        0.274871

        >>> T = 80
        >>> Al.B(T)["Al27"].round(6)
        0.337081
        """
        constant = (4 * sp.constants.c ** 2 * np.pi**2) * const["reduced Planck constant in eV s"][0] ** 2
        constant /= const["atomic mass unit-electron volt relationship"][0] * const["Boltzmann constant in eV/K"][0]
        B = {}
        for element, atom_mass in self.atom_mass.items():
            B[element] = constant * self.pdos[element].DebyeWallerCoeff(T) / (T * atom_mass)
            if anstrom:
                B *= 1.0e20
        return B
