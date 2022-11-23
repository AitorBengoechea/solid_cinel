# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:46:42 2022

@author: Aitor Bengoechea
"""

from solid_cinel.core.material.solid import Solid, hkl_max_value
from solid_cinel.core.material.pdos import Pdos
from solid_cinel.core._numba import hklloop
from solid_cinel.cinematic.lab import Neutron
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
        pdos = {}
        elements_name = self.atoms.index
        if isinstance(args[10], float):
            atom_pdos = Pdos(args[9],
                      index=pd.Index(np.arange(len(args[9])) * args[10],
                                     name="E"))
            pdos[elements_name[0]] = atom_pdos
        else:
            for i in range(len(args[10])):
                atom_pdos = Pdos(args[9][i],
                      index=pd.Index(np.arange(len(args[9][i])) * args[10][i],
                                     name="E"))
                pdos[elements_name[i]] = atom_pdos
        self.pdos = pd.Series(pdos)

    def get_Bfact(self, T, anstrom=True) -> float:
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
        >>> Al = Target_mat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy, interv_in_energy)

        Test the results:
        >>> T = 20
        >>> Al.get_Bfact(T)["Al27"].round(6)
        0.274871

        >>> T = 80
        >>> Al.get_Bfact(T)["Al27"].round(6)
        0.337081
        """
        constant = (4 * sp.constants.c ** 2 * np.pi**2) * const["reduced Planck constant in eV s"][0] ** 2
        constant /= const["atomic mass unit-electron volt relationship"][0] * const["Boltzmann constant in eV/K"][0]
        atom_masses = self.atoms.apply(lambda x: x.atom_mass)

        def get_Bfac(single_pdos):
            B = constant * single_pdos.DebyeWallerCoeff(T) / T
            if anstrom:
                B *= 1.0e20
            return B

        return self.pdos.apply(get_Bfac) / atom_masses

    def get_multiplicity(self, T, E) -> pd.DataFrame:
        """
        Obtain hkl data for the solid in a certain temperature and for a neutron
        certain energy filtering with the multiplicity.

        Parameters
        ----------
        T : 'float'
            Temperature in K
        E : 'float'
            Neutron energy

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
        >>> Al = Target_mat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy, interv_in_energy)

        Test the results:
        >>> T = 20
        >>> E = 2.301
        >>> multiplicity = Al.get_multiplicity(T, E)
        >>> multiplicity["Fsq"] = multiplicity["Fsq"].round(6)
        >>> multiplicity.shape[0]
        678
        >>> multiplicity.iloc[:10]
                            d       Fsq  Multiplicity
        h   k   l
        1.0 1.0 0.0  2.019999  0.000000           6.0
                1.0  2.332494  0.000000           8.0
        2.0 1.0 1.0  1.428355  0.000000          12.0
            2.0 0.0  1.010000  6.653574           6.0
                1.0  1.218106  0.000000          24.0
                2.0  1.166247  6.881496           8.0
        3.0 2.0 1.0  0.903371  0.000000          24.0
                2.0  0.926839  0.000000          24.0
            3.0 2.0  0.824661  0.000000          24.0
                3.0  0.777498  0.000000          32.0
        >>> multiplicity.iloc[667:677]
                            d  Fsq  Multiplicity
        h    k    l
        31.0 19.0 18.0  0.091254  0.0         168.0
                  19.0  0.090999  0.0         384.0
             20.0 17.0  0.090884  0.0         336.0
                  18.0  0.090815  0.0         552.0
                  19.0  0.090609  0.0         288.0
             21.0 15.0  0.089911  0.0         384.0
                  16.0  0.090157  0.0         168.0
                  17.0  0.090269  0.0         216.0
                  18.0  0.090247  0.0         192.0
                  19.0  0.090090  0.0         168.0
        """
        recs_vec = self.reciproc_vec.values
        d_min = Neutron(E).d_min
        hkl_max = hkl_max_value(recs_vec, d_min)
        B = self.get_Bfact(T)
        pos = self.atom_pos
        csl = self.atoms.apply(lambda x: x.b["b_coh"])
        hkl_data =  numba_hkl_data(d_min, hkl_max, recs_vec, B, pos, csl)
        return filter_hkl_data(hkl_data)

    def get_coherent_XS(self, T, d_min, multiplicity=True):
        multiplicity = self.get_multiplicity(T, d_min)
        # Bragg Edges:
        return


def numba_hkl_data(d_min, hkl_max, rec_vecs, Bfac, pos, csl) -> pd.DataFrame:
    """
    Obtain hkl data for the solid in a certain temperature and for a neutron
    certain energy.

    Parameters
    ----------
    d_min : 'float'
        The minimum dspacing for the LEAPR module of NJOY
    hkl_max : 'np.array'
        Maximun h, k, l index for generating a d>d_min
    rec_vecs : 'np.array'
        Reciprocal vectors
    Bfac : 'pd.Series'
        Pandas series with the B factor for Target_Material object elements.
    pos : 'pd.Series'
        Pandas series with atomic position of elements in Target_Material 
        object.
    csl : 'pd.Series'
        Coherent elastic length for each element of Target_Material object.

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
    >>> Al = Target_mat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy, interv_in_energy)

    Test the results:
    >>> T = 20
    >>> E = 2.301
    >>> recs_vec = Al.reciproc_vec.values
    >>> d_min = Neutron(E).d_min
    >>> hkl_max = hkl_max_value(recs_vec, d_min)
    >>> B = Al.get_Bfact(T)
    >>> pos = Al.atom_pos
    >>> csl = Al.atoms.apply(lambda x: x.b["b_coh"])
    >>> hkl_data = numba_hkl_data(d_min, hkl_max, recs_vec, B, pos, csl)
    >>> hkl_data.shape[0]
    95800
    >>> hkl_data.iloc[:10]
          h     k     l         d           Fsq   d_round  f_round
    0  31.0  21.0  20.0  0.089800  3.810648e-36  0.089800      0.0
    1  31.0  21.0  19.0  0.090090  4.251472e-36  0.090090      0.0
    2  31.0  21.0  18.0  0.090247  4.058640e-35  0.090247      0.0
    3  31.0  21.0  17.0  0.090269  1.136933e-36  0.090269      0.0
    4  31.0  21.0  16.0  0.090157  1.090059e-36  0.090157      0.0
    5  31.0  21.0  15.0  0.089911  3.577059e-35  0.089911      0.0
    6  31.0  20.0  21.0  0.089800  1.524259e-35  0.089800      0.0
    7  31.0  20.0  20.0  0.090269  1.941747e-34  0.090269      0.0
    8  31.0  20.0  19.0  0.090609  1.289999e-36  0.090609      0.0
    9  31.0  20.0  18.0  0.090815  3.593004e-34  0.090815      0.0
    """
    B_ = np.array(np.stack(Bfac.values))
    pos_ = np.array(np.stack(pos.values))
    cls_ = np.array(np.stack(csl.values))
    hkl_data = hklloop(d_min, hkl_max, rec_vecs, B_, pos_, cls_)
    columns = ["h", "k", "l", "d", "Fsq", "d_round", "f_round"]
    return pd.DataFrame(hkl_data, columns=columns)

def filter_hkl_data(hkl_data) -> pd.DataFrame:
    """
    Filter hkl data for being in order and for only having one hkl combination
    for the same d and Fsq

    Parameters
    ----------
    hkl_data : 'pd.DataFrame'
        hkl data before any filter is apply.
    """
    def _get_multiplicity(frame):
        data_ = frame.sort_values(by=["h", "k", "l"], ascending=False).iloc[0]
        data_["Multiplicity"] = frame.shape[0] 
        return data_

    return hkl_data.groupby(by=["d_round", "f_round"])\
                              .apply(_get_multiplicity)\
                              .sort_values(by=["h", "k", "l"])\
                              .set_index(["h", "k", "l"])\
                              .drop(columns = ["d_round", "f_round"])


