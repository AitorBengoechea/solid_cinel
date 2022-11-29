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
import numba as nb
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
        .. math::
            B_j= \dfrac{4\pi^2\hbar^2}{M_jk_BT}\lambda_s

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

    def get_multiplicity(self, T, E, precision=[6, 6]) -> pd.DataFrame:
        """
        Obtain hkl data for the solid in a certain temperature and for a neutron
        certain energy filtering with the multiplicity.

        Parameters
        ----------
        T : 'float'
            Temperature in K
        E : 'float'
            Neutron energy in eV
        precision: ['int', 'int'], optional
            Precision to get the multiplicity for d_hkl and Fsq_hkl. The
            default is [6, 6].

        Examples
        --------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 0, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0., 0., 0.])
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
        >>> multiplicity.shape[0]
        678
        >>> multiplicity.iloc[:10]
                      d       Fsq  Orientation angle  Multiplicity
        h k l
        1 1 0  2.019999  0.115016         125.264390           6.0
            1  2.332494  0.115989          70.528779           8.0
        2 1 1  1.428355  0.111207          90.000000          12.0
          2 0  1.010000  0.103962         125.264390           6.0
            1  1.218106  0.108433         100.024988          24.0
            2  1.166247  0.107523          70.528779           8.0
        3 2 1  0.903371  0.100519         104.963217          24.0
            2  0.926839  0.101369          82.388621          24.0
          3 2  0.824661  0.097189          90.000000          24.0
            3  0.777498  0.094765          70.528779          32.0
        >>> multiplicity.round(6).iloc[667:677]
                         d  Fsq  Orientation angle  Multiplicity
        h  k  l
        31 19 18  0.091254  0.0          87.009863         168.0
              19  0.090999  0.0          84.777020         384.0
           20 17  0.090884  0.0          90.000000         336.0
              18  0.090815  0.0          87.768637         552.0
              19  0.090609  0.0          85.544023         288.0
           21 15  0.089911  0.0          95.160350         384.0
              16  0.090157  0.0          92.954150         168.0
              17  0.090269  0.0          90.739152         216.0
              18  0.090247  0.0          88.521943         192.0
              19  0.090090  0.0          86.309150         168.0
        """
        recs_vec = self.reciproc_vec.values
        d_min = Neutron(E).d_min
        hkl_max = hkl_max_value(recs_vec, d_min)
        B = self.get_Bfact(T)
        pos = self.atom_pos
        csl = self.atoms.apply(lambda x: x.b["b_coh"])
        hkl_data = numba_hkl_data(d_min,
                                  hkl_max,
                                  recs_vec,
                                  B,
                                  pos,
                                  csl,
                                  self.preferred_orientation.values,
                                  np.array(precision)
                                  )
        return hkl_data.sort_values(by=["h", "k", "l"]).set_index(["h", "k", "l"])

    @staticmethod
    def get_pddf(data, kind=None, pddf_val=None) -> pd.DataFrame:
        """
        Add to the hkl data dataframe the Pole Density Distribution Function.
        March-dollase:
            .. math::
                \mathcal{P}_{hkl}(\Theta)=(P_1^2\cos^2(\Theta)+P_1^{-1}\sin^2(\Theta))^{-3/2}.
        Altomare:
            .. math::
                \mathcal{P}_{hkl}(\Theta)=\exp(P_1\cos(2\Theta))
        cvc:
            .. math::
                \mathcal{P}_{hkl}(\Theta)=\exp(-P_1(1-\cos^{P_2}(\Theta)))

        Parameters
        ----------
        data: 'pd.DataFrame'
            Frame with hkl data.
        kind : 'str', optional
            key to calculate PDDF. The default is None. Options:
                - march_dollase
                - altomare
                - cvc
        pddf_val : 1D iterable, optional
            Value to calculate PDDF. The default is None. Options:
                - march-dollase : shape(1, 1)
                - altomare: shape(1, 2)
                - cvc: shape(1, 2)

        Example
        -------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 0, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0., 0., 0.])
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Target_mat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy, interv_in_energy)
        >>> T = 20
        >>> E = 2.301
        >>> multiplicity = Al.get_multiplicity(T, E)
        >>> multiplicity.iloc[:10]
                      d       Fsq  Orientation angle  Multiplicity
        h k l
        1 1 0  2.019999  0.115016         125.264390           6.0
            1  2.332494  0.115989          70.528779           8.0
        2 1 1  1.428355  0.111207          90.000000          12.0
          2 0  1.010000  0.103962         125.264390           6.0
            1  1.218106  0.108433         100.024988          24.0
            2  1.166247  0.107523          70.528779           8.0
        3 2 1  0.903371  0.100519         104.963217          24.0
            2  0.926839  0.101369          82.388621          24.0
          3 2  0.824661  0.097189          90.000000          24.0
            3  0.777498  0.094765          70.528779          32.0

        Test the results:
        >>> Target_mat.get_pddf(multiplicity).iloc[:10]
                      d       Fsq  Orientation angle  Multiplicity  PDDF
        h k l
        1 1 0  2.019999  0.115016                0.0           6.0   1.0
            1  2.332494  0.115989                0.0           8.0   1.0
        2 1 1  1.428355  0.111207                0.0          12.0   1.0
          2 0  1.010000  0.103962                0.0           6.0   1.0
            1  1.218106  0.108433                0.0          24.0   1.0
            2  1.166247  0.107523                0.0           8.0   1.0
        3 2 1  0.903371  0.100519                0.0          24.0   1.0
            2  0.926839  0.101369                0.0          24.0   1.0
          3 2  0.824661  0.097189                0.0          24.0   1.0
            3  0.777498  0.094765                0.0          32.0   1.0

        >>> multiplicity = Al.get_multiplicity(T, E)
        >>> Target_mat.get_pddf(multiplicity, kind='march-dollase', pddf_val=2).iloc[:10]
                      d       Fsq  Orientation angle  Multiplicity      PDDF
        h k l
        1 1 0  2.019999  0.115016         125.264390           6.0  0.464758
            1  2.332494  0.115989          70.528779           8.0  1.193243
        2 1 1  1.428355  0.111207          90.000000          12.0  2.828427
          2 0  1.010000  0.103962         125.264390           6.0  0.464758
            1  1.218106  0.108433         100.024988          24.0  2.119463
            2  1.166247  0.107523          70.528779           8.0  1.193243
        3 2 1  0.903371  0.100519         104.963217          24.0  1.592384
            2  0.926839  0.101369          82.388621          24.0  2.377318
          3 2  0.824661  0.097189          90.000000          24.0  2.828427
            3  0.777498  0.094765          70.528779          32.0  1.193243

        >>> multiplicity = Al.get_multiplicity(T, E)
        >>> Target_mat.get_pddf(multiplicity, kind='altomare', pddf_val=[1, 1]).iloc[:10]
                      d       Fsq  Orientation angle  Multiplicity      PDDF
        h k l
        1 1 0  2.019999  0.115016         125.264390           6.0  1.716531
            1  2.332494  0.115989          70.528779           8.0  1.459426
        2 1 1  1.428355  0.111207          90.000000          12.0  1.367879
          2 0  1.010000  0.103962         125.264390           6.0  1.716531
            1  1.218106  0.108433         100.024988          24.0  1.390865
            2  1.166247  0.107523          70.528779           8.0  1.459426
        3 2 1  0.903371  0.100519         104.963217          24.0  1.420350
            2  0.926839  0.101369          82.388621          24.0  1.381017
          3 2  0.824661  0.097189          90.000000          24.0  1.367879
            3  0.777498  0.094765          70.528779          32.0  1.459426

        >>> multiplicity = Al.get_multiplicity(T, E)
        >>> Target_mat.get_pddf(multiplicity, kind='cvc', pddf_val=[1, 1]).iloc[:10]
                      d       Fsq  Orientation angle  Multiplicity      PDDF
        h k l
        1 1 0  2.019999  0.115016         125.264390           6.0  0.206522
            1  2.332494  0.115989          70.528779           8.0  0.513417
        2 1 1  1.428355  0.111207          90.000000          12.0  0.367879
          2 0  1.010000  0.103962         125.264390           6.0  0.206522
            1  1.218106  0.108433         100.024988          24.0  0.309104
            2  1.166247  0.107523          70.528779           8.0  0.513417
        3 2 1  0.903371  0.100519         104.963217          24.0  0.284165
            2  0.926839  0.101369          82.388621          24.0  0.419981
          3 2  0.824661  0.097189          90.000000          24.0  0.367879
            3  0.777498  0.094765          70.528779          32.0  0.513417
        """
        orientation_angle = data.loc[:, "Orientation angle"] * np.pi / 180

        if kind is None:
            data["Orientation angle"] = 0.
            PDDF_hkl = 1.
        elif kind.lower() == 'march-dollase' and isinstance(pddf_val, (int, float)):
            PDDF_hkl = (pddf_val ** 2 * np.cos(orientation_angle) ** 2 +
                        np.sin(orientation_angle) ** 2 / pddf_val) ** (-1.5)
        elif kind.lower() == 'altomare' and len(pddf_val) == 2:
            PDDF_hkl = np.exp(pddf_val[0] * np.cos(2 * orientation_angle)) + pddf_val[1]
        elif kind.lower() == 'cvc' and len(pddf_val) == 2:
            PDDF_hkl = np.exp(-pddf_val[0] * (1 - np.cos(orientation_angle) ** pddf_val[1]))
        else:
            ValueError("Introduced kind is not available")

        data["PDDF"] = PDDF_hkl
        return data

    @staticmethod
    def get_difrac_angles(data, E) -> pd.DataFrame:
        """
        Add to the hkl data dataframe the difraction angles(ª) vs hkl data
        .. math::
            2\theta_{hkl}=\arccos\left(1-\dfrac{\pi^2\hbar^2}{md_{hkl}^2E}\right)

        Parameters
        ----------
        data: 'pd.DataFrame'
            Frame with hkl data.
        E : 'float'
            Neutron energy in eV

        Example
        -------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 0, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0., 0., 0.])
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Target_mat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy, interv_in_energy)
        >>> T = 20
        >>> E = 2.301
        >>> multiplicity = Al.get_multiplicity(T, E)
        >>> multiplicity.iloc[:10]
                      d       Fsq  Orientation angle  Multiplicity
        h k l
        1 1 0  2.019999  0.115016         125.264390           6.0
            1  2.332494  0.115989          70.528779           8.0
        2 1 1  1.428355  0.111207          90.000000          12.0
          2 0  1.010000  0.103962         125.264390           6.0
            1  1.218106  0.108433         100.024988          24.0
            2  1.166247  0.107523          70.528779           8.0
        3 2 1  0.903371  0.100519         104.963217          24.0
            2  0.926839  0.101369          82.388621          24.0
          3 2  0.824661  0.097189          90.000000          24.0
            3  0.777498  0.094765          70.528779          32.0

        Test the results:
        >>> Target_mat.get_difrac_angles(multiplicity, E).iloc[:10]
                      d       Fsq  Orientation angle  Multiplicity     theta
        h k l
        1 1 0  2.019999  0.115016         125.264390           6.0   5.350060
            1  2.332494  0.115989          70.528779           8.0   4.632867
        2 1 1  1.428355  0.111207          90.000000          12.0   7.568882
          2 0  1.010000  0.103962         125.264390           6.0  10.711827
            1  1.218106  0.108433         100.024988          24.0   8.877727
            2  1.166247  0.107523          70.528779           8.0   9.273329
        3 2 1  0.903371  0.100519         104.963217          24.0  11.980567
            2  0.926839  0.101369          82.388621          24.0  11.676144
          3 2  0.824661  0.097189          90.000000          24.0  13.128861
            3  0.777498  0.094765          70.528779          32.0  13.929091

        """
        constant = const["reduced Planck constant in eV s"][0] ** 2 * sp.constants.c ** 2
        constant /= (const["neutron mass energy equivalent in MeV"][0] * 1.0e6)
        constant *= 1.0e20  # Coherence with Bfac that is in anstrom
        d = data.loc[:, "d"]
        angle_value = np.clip(1 - np.pi ** 2 * constant / (d ** 2 * E), -1, 1)
        data["theta"] = np.arccos(angle_value) * 180 / np.pi
        return data

    @staticmethod
    def get_BraggEdges_Xs(data, unit_cell_vol, atom_number,
                          threshold=1.e-30) -> pd.DataFrame:
        """
        Add to the hkl data dataframe the cross section related with the
        Bragg Edges.
        .. math::
            \sigma_{\textrm{coh}}^{\textrm{el}}(E_{hkl})=\dfrac{\pi^2\hbar^2}{mN_{uc}V_{uc}E_{hkl}}M_{hkl}d_{hkl}\left|F(\vec{\tau}_{hkl})\right|^2\mathcal{P}_{hkl}(\Theta_{hkl})

        Parameters
        ----------
        data: 'pd.DataFrame'
            Frame with hkl data.
        unit_cell_vol : 'float'
            Unit cell volume.
        atom_number : 'int'
            The number of atoms in the unit cell.
        threshold : 'float', optional
            Minimun value of xs. The default is 1.e-30.

        Example
        -------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 0, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0., 0., 0.])
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Target_mat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy, interv_in_energy)
        >>> T = 20
        >>> E = 2.301
        >>> unit_cell_vol = Al.unit_cell_vol
        >>> atom_number = Al.atom_number
        >>> BraggEdges = Al.get_BraggEdges(T, E, xs=False, theta=False)
        >>> BraggEdges.iloc[:10]
                      d       Fsq  Orientation angle  Multiplicity  PDDF         E
        h k l
        1 1 1  2.332494  0.115989                0.0           8.0   1.0  0.003759
            0  2.019999  0.115016                0.0           6.0   1.0  0.005012
        2 1 1  1.428355  0.111207                0.0          12.0   1.0  0.010024
          2 1  1.218106  0.108433                0.0          24.0   1.0  0.013783
            2  1.166247  0.107523                0.0           8.0   1.0  0.015036
            0  1.010000  0.103962                0.0           6.0   1.0  0.020048
        3 2 2  0.926839  0.101369                0.0          24.0   1.0  0.023807
            1  0.903371  0.100519                0.0          24.0   1.0  0.025060
          3 2  0.824661  0.097189                0.0          24.0   1.0  0.030072
            3  0.777498  0.094765                0.0          32.0   1.0  0.033831

        Test the results:
        >>> Target_mat.get_BraggEdges_Xs(BraggEdges, unit_cell_vol, atom_number).loc[::, "Xs"].iloc[:10]
        h  k  l
        1  1  1    0.005370
              0    0.003459
        2  1  1    0.004729
           2  1    0.007865
              2    0.002489
              0    0.001563
        3  2  2    0.005595
              1    0.005407
           3  2    0.004773
              3    0.005850
        Name: Xs, dtype: float64
        """
        constant = const["reduced Planck constant in eV s"][0] ** 2 * sp.constants.c ** 2
        constant /= (const["neutron mass energy equivalent in MeV"][0] * 1.0e6)
        constant *= 1.0e20  # Coherence with Bfac that is in anstrom
        if "PDDF" not in data.columns:
            Target_mat.get_pddf(data)
        data["Xs"] = data["d"] * data["Fsq"] * data["Multiplicity"] * data["PDDF"]
        data["Xs"] *= constant * np.pi ** 2
        data["Xs"] /= unit_cell_vol * atom_number
        if threshold:
            data["Xs"][data["Xs"] < threshold] = 0.0
        return data

    def get_BraggEdges(self, *args, xs=True, file_BraggEdges=None,
                       theta=True, **kwargs) -> pd.DataFrame:
        """
        Get BraggEdges.
        .. math::
            E_n=\dfrac{h^2}{2m\lambda^2}=\dfrac{h^2}{2m(2d_{hkl})^2}

        Parameters
        ----------
        xs : 'bool', optional
            Option to get the xs related with Bragg Edges. The default is
            True
        theta : 'bool', optional
            Option to get the difraction angles vs hkl data. The default is
            True
        file_BraggEdges : 'str', optional
            Name of the file to write Bragg Edges data. The default is False.

        Parameters for get_multiplicity
        -------------------------------
        T : 'float'
            Temperature in K
        E : 'float'
            Neutron energy in eV
        precision: ['int', 'int'], optional
            Precision to get the multiplicity for d_hkl and Fsq_hkl. The
            default is [6, 6].

        Parameters for get_ppdf
        -----------------------
        kind : 'str', optional
            key to calculate PDDF. The default is None. Options:
                - march_dollase
                - altomare
                - cvc
        pddf_val : 1D iterable, optional
            Value to calculate PDDF. The default is None. Options:
                - march-dollase : shape(1, 1)
                - altomare: shape(1, 2)
                - cvc: shape(1, 2)

        Parameters for get_BraggEdges_Xs
        --------------------------------
        threshold : 'float', optional
            Minimun value of xs. The default is 1.e-30.

        Returns
        -------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 0, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0., 0., 0.])
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Target_mat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy, interv_in_energy)
        >>> T = 20
        >>> E = 2.301

        Test the results:
        >>> Al.get_BraggEdges(T, E).round(6).iloc[:10, :4]
                      d       Fsq  Orientation angle  Multiplicity
        h k l
        1 1 1  2.332494  0.115989                0.0           8.0
            0  2.019999  0.115016                0.0           6.0
        2 1 1  1.428355  0.111207                0.0          12.0
          2 1  1.218106  0.108433                0.0          24.0
            2  1.166247  0.107523                0.0           8.0
            0  1.010000  0.103962                0.0           6.0
        3 2 2  0.926839  0.101369                0.0          24.0
            1  0.903371  0.100519                0.0          24.0
          3 2  0.824661  0.097189                0.0          24.0
            3  0.777498  0.094765                0.0          32.0

        >>> Al.get_BraggEdges(T, E).round(6).iloc[:10, 4::]
               PDDF         E        Xs      theta
        h k l
        1 1 1   1.0  0.003759  0.005370   4.632867
            0   1.0  0.005012  0.003459   5.350060
        2 1 1   1.0  0.010024  0.004729   7.568882
          2 1   1.0  0.013783  0.007865   8.877727
            2   1.0  0.015036  0.002489   9.273329
            0   1.0  0.020048  0.001563  10.711827
        3 2 2   1.0  0.023807  0.005595  11.676144
            1   1.0  0.025060  0.005407  11.980567
          3 2   1.0  0.030072  0.004773  13.128861
            3   1.0  0.033831  0.005850  13.929091
        """
        # Get multiplicity
        data = self.get_multiplicity(*args,
                                     precision=kwargs.pop("precision", [6, 6])
                                     )
        # Get PDDF:
        self.get_pddf(data,
                      kwargs.pop("kind", None),
                      kwargs.pop("pddf_val", None)
                      )

        # Get Bragg Edges:
        constant = const["reduced Planck constant in eV s"][0] ** 2 * sp.constants.c ** 2
        constant /= (const["neutron mass energy equivalent in MeV"][0] * 1.0e6)
        constant *= 1.0e20  # Coherence with Bfac that is in anstrom
        data["E"] = np.pi ** 2 * constant / (2 * data["d"] ** 2)
        data = data.sort_values(by=["E"])

        # Optional argument:
        # Xs:
        if xs:
            self.get_BraggEdges_Xs(data,
                                   self.unit_cell_vol,
                                   self.atom_number,
                                   threshold=kwargs.get("threshold", 1.e-30)
                                   )

        # difraction angles vs hkl data
        if theta:
            self.get_difrac_angles(data,
                                   args[1]
                                   )

        # Get the final result
        if file_BraggEdges:
            data.to_csv(file_BraggEdges,
                        sep='\t',
                        float_format="%20.10e")
        return data

    def get_coherent_Xs(self, energy_cut, energy_sup, *args,
                        file_Xs=None, **kwargs) -> pd.DataFrame:
        """
        Get coherent Xs.

        Parameters
        ----------
        energy_cut : 'float'
            energy cut-off, above which the peak will not be stored.
        energy_sup : 'float', optional
            Superior bound of energy. The default is False.
        file_Xs : 'str', optional
            Name of the file to write Xs the data. The default is False.

        Parameters for get_BraggEdges
        -----------------------------
        theta : 'bool', optional
            Option to get the difraction angles vs hkl data. The default is
            True
        file_BraggEdges : 'str', optional
            Name of the file to write Bragg Edges data. The default is False.

        Parameters for get_multiplicity
        -------------------------------
        T : 'float'
            Temperature in K
        E : 'float'
            Neutron energy in eV
        precision: ['int', 'int'], optional
            Precision to get the multiplicity for d_hkl and Fsq_hkl. The
            default is [6, 6].

        Parameters for get_ppdf
        -----------------------
        kind : 'str', optional
            key to calculate PDDF. The default is None. Options:
                - march_dollase
                - altomare
                - cvc
        pddf_val : 1D iterable, optional
            Value to calculate PDDF. The default is None. Options:
                - march-dollase : shape(1, 1)
                - altomare: shape(1, 2)
                - cvc: shape(1, 2)

        Parameters for get_BraggEdges_Xs
        --------------------------------
        threshold : 'float', optional
            Minimun value of xs. The default is 1.e-30.

        Returns
        -------
        Object initialization:
        >>> preferred_orientation = np.array([ 0, 0, 1 ])
        >>> a = 2.856710674519725
        >>> dir_vec_length = [a, a, a]
        >>> dir_vec_angles = [60, 60, 60]
        >>> unit_pos = np.array([0., 0., 0.])
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Target_mat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy, interv_in_energy)
        >>> T = 20
        >>> E = 2.301

        Test the results:
        >>> energy_cut = 2.301
        >>> energy_sup = 10.0
        >>> Al.get_coherent_Xs(energy_cut, energy_sup, T, E).round(6).iloc[:10]
        ZAM         130270
        MT               2
        E
        0.003759  1.428617
        0.005012  1.761562
        0.010024  1.352593
        0.013783  1.554360
        0.015036  1.590374
        0.020048  1.270752
        0.023807  1.305112
        0.025060  1.455634
        0.030072  1.371739
        0.033831  1.392243
        """
        BraggEdges_Xs = self.get_BraggEdges(*args, **kwargs)\
                            .reset_index()\
                            .loc[::, ["E", "Xs"]]
        BraggEdges_Xs["E"] = BraggEdges_Xs["E"].round(6)
        BraggEdges_Xs = BraggEdges_Xs[BraggEdges_Xs["E"] <= energy_cut]

        if (BraggEdges_Xs["E"] > energy_sup).any():
            raise ValueError("Energy superior bonds not properly introduce")
        else:
            bound_sup = pd.DataFrame([[energy_sup, 0.0]],
                                     columns=["E", "Xs"])
            BraggEdges_Xs = pd.concat([BraggEdges_Xs, bound_sup],
                                      axis=0,
                                      ignore_index=True)
        xs = BraggEdges_Xs.set_index("E")
        if xs.index.has_duplicates:
            xs = xs.groupby(by="E").sum()
        xs["Xs"] = np.cumsum(xs["Xs"]) / xs.index.values
        xs.columns = pd.MultiIndex.from_product(
            [self.atoms.apply(lambda x: x.zam).values, [2]],
            names=["ZAM", "MT"])
        # Optional argument:
        if file_Xs:
            xs.to_csv(file_Xs,
                      sep='\t',
                      float_format="%20.10e")
        return xs


def numba_hkl_data(d_min, hkl_max, rec_vecs, Bfac, pos, csl,
                   preferred_orientation, precision) -> pd.DataFrame:
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
    precision: 'np.array':
        Array containing:
            0: Precision to reagroup in multiplicity the d_hkl
            1: Precision to reagroup in multiplicity the Fsq_hkl

    Examples
    --------
    Object initialization:
    >>> preferred_orientation = np.array([ 0, 0, 1 ])
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
    >>> precision = np.array([6, 6])
    >>> preferred_orientation = Al.preferred_orientation.values
    >>> hkl_data = numba_hkl_data(d_min, hkl_max, recs_vec, B, pos, csl, preferred_orientation, precision)
    >>> hkl_data.shape[0]
    678
    >>> hkl_data.round(6).iloc[:10]
        h   k   l         d  Fsq  Orientation angle  Multiplicity
    0  31  21  20  0.089800  0.0          84.107323         336.0
    1  31  21  19  0.090090  0.0          86.309150         168.0
    2  31  21  18  0.090247  0.0          88.521943         192.0
    3  31  21  17  0.090269  0.0          90.739152         216.0
    4  31  21  16  0.090157  0.0          92.954150         168.0
    5  31  21  15  0.089911  0.0          95.160350         384.0
    6  31  20  19  0.090609  0.0          85.544023         288.0
    7  31  20  18  0.090815  0.0          87.768637         552.0
    8  31  20  17  0.090884  0.0          90.000000         336.0
    9  31  19  19  0.090999  0.0          84.777020         384.0
    """
    # Preparation of variables to be accept in numba nopython mode:
    Bfac_ = nb.typed.Dict.empty(
            key_type=nb.core.types.unicode_type,
            value_type=nb.core.types.float64,
        )
    pos_ = nb.typed.Dict.empty(
            key_type=nb.core.types.unicode_type,
            value_type=nb.core.types.float64[:, :],
        )
    csl_ = nb.typed.Dict.empty(
            key_type=nb.core.types.unicode_type,
            value_type=nb.core.types.float64,
        )
    for element, value in Bfac.items():
        Bfac_[element] = value
        pos_[element] = pos[element]
        csl_[element] = csl[element]

    preferred_orientation_ = np.array(preferred_orientation, dtype=float)

    # Execute numba
    hkl_data_dict = hklloop(d_min, hkl_max, rec_vecs, Bfac_, pos_, csl_,
                            preferred_orientation_, precision)

    # Order the output
    columns = ["h", "k", "l", "d", "Fsq", "Orientation angle", "Multiplicity"]

    return pd.DataFrame([[h, k, l, d_hkl, Fsq_hkl, orientation, mul]
                         for (h, k, l), [d_hkl, Fsq_hkl, orientation, mul]
                         in hkl_data_dict.items()], columns=columns)
