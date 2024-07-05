# -*- coding: utf-8 -*-
"""
Python file for working with Target Material.

@author: AB272525
"""

from solid_cinel.core.material.structure.solid import Solid, hkl_max_value
from solid_cinel.core.material.vibration.pdos import Pdos
from solid_cinel.core.cinematic.frames import Neutron
from scipy.constants import physical_constants as const
import numpy as np
from math import pi, cos, sin, acos, exp
import pandas as pd
import numba as nb
import collections
from scipy.constants import c


collections.Callable = collections.abc.Callable

# Constants:
h = const["reduced Planck constant in eV s"][0]
m_to_eV = const["atomic mass unit-electron volt relationship"][0]
mn_to_MeV = const["neutron mass energy equivalent in MeV"][0]
kb = const["Boltzmann constant in eV/K"][0]
BfacUnitChange = (4 * c ** 2 * pi**2) * h ** 2 / (m_to_eV * kb)
BraggUnitChange = 1.0e20 * h ** 2 * c ** 2 / (mn_to_MeV * 1.0e6)


class TargetMat(Solid):
    """
    Class to store all the Target material methods and attributes.

    Attributes
    ----------
    A : list[int]
        Atomic number.
    Z : list[int]
        Number of protons.
    dir_vec_length : iterable or `np.array` of size (1, 3)
        Direct lattice vectors length in fm.
    preferred_orientation : iterable or `np.array` of size (1, 3)
        Direct lattice vectors angles in ª.
    preferred_orientation : iterable or `np.array` of size (1, 3)
        Preferred orientation of the target.
    unit_pos : dict{"element name": 1D iterable}
        Unitary positions of atoms in the lattice unit cell.
    atom_mass : list[float]
        Atom mass, amu.
    b_coh : list[float]
        Bound coherent scattering length (fm).
    b_incoh : list[float]
        Bound incoherent scattering length (fm).
    rho_in_energy : list[1D iterable]
        Density of states in energy, eV.
    interv_in_energy : list[1D iterable]
        Energy intervals in which the density of states is defined, eV.
    energy_sup : float
        Upper limit of the energy range, eV.
    energyCut : float
        Energy cut for the density of states, eV.

    Methods
    -------
    get_Bfact: float or pd.Series
        Calculate mean square displacement for a certain pdos information
    get_multiplicity: pd.DataFrame
        Obtain hkl data for the solid in a certain temperature and for a neutron
        certain energy filtering with the multiplicity
    get_BraggEdges: pd.DataFrame
        Get Bragg Edges
    get_XsCoh: pd.DataFrame
        Get coherent xs for the materials in the class
    get_Sab: pd.Series
        Generate S(alpha, -beta) matrix for the materials in the class
    get_inelastic_Xs: pd.DataFrame
        Get inelastic Xs for the Target material based on the S(alpha, -beta)
    """

    def __init__(self, *args):
        """
        Class to store all the Target material methods and atributtes.

        Parameters for Crys_atom
        ------------------------
        A : list[int]
            Atomic number.
        Z : list[int]
            Number of protons.
        dir_vec_length : iterable or `np.array` of size (1, 3)
            Direct lattice vectors length in fm.
        preferred_orientation : iterable or `np.array` of size (1, 3)
            Direct lattice vectors angles in ª.
        preferred_orientation : iterable or `np.array` of size (1, 3)
            Preferred orientation of the target.
        unit_pos : dict{"element name": 1D iterable}
            Unitary positions of atoms in the lattice unit cell.
        atom_mass : list[float]
            Atom mass, amu.
        b_coh : list[float]
            Bound coherent scattering length (fm).
        b_incoh : list[float]
            Bound incoherent scattering length (fm).

        Parameters for Pdos
        ------------------------
        rho : list of 1D iterable
            rho values for each element.
        interval_energy : list of 'float'
            Energy interval in eV for each element.
        """
        Solid.__init__(self, *args[0:9])
        # Avoid data setter in Pdos:
        if len(args) > 9:
            self.compute_pdos(args[9])
        else:
            self.pdos = None

    def compute_pdos(self, pdosDict: [Pdos, dict[Pdos]]):
        atoms = self.atoms.index
        if len(atoms) == 1:
            pdosCheck = pd.Series(pdosDict, index=atoms)
        else:
            pdosCheck = pd.Series({element: pdosDict.get(element) for element in atoms})
            if pdosCheck.isnull().any():
                raise ValueError("The pdosDict must have the same elements as the object.")
        self.pdos = pdosCheck


    def  get_Bfact(self, T: float, pdosDict: dict[Pdos] = None,
                  anstrom: bool = True) -> [float, pd.Series]:
        """
        Calculate mean square displacement for a certain pdos information.
        .. math::
            B_j= \frac{4\pi^2\hbar^2}{M_jk_BT}\lambda_s

        Parameters
        ----------
        T : 'float'
            Temperature in K
        pdosDict : 'dict[Pdos]', optional
            Dictionary with the pdos for each element. The default is None.
        anstrom : 'bool', optional
            Option to obtain the B unit in A^2. The default is True.

        Returns
        -------
        "float" or "pd.Series"
            Mean square displacement.

        Examples
        --------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> from solid_cinel.tests.materials.Al27.examples import rho_in_energy, interv_in_energy
        >>> pdosAl27 = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> Al = TargetMat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass, b_coh, b_incoh)
        >>> from solid_cinel.data.materials.UO2 import *
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy, interv_in_energy
        >>> pdosUO2 = {"O16": Pdos.from_dE(rho_in_energy[0], interv_in_energy[0]), "U238": Pdos.from_dE(rho_in_energy[1], interv_in_energy[1])}
        >>> UO2 = TargetMat(unit_pos, dir_vec_length, dir_vec_angles, preferred_orientation, A, Z, atom_mass, b_coh, b_incoh, pdosUO2)

        Test the results:
        >>> T = 20
        >>> float(Al.get_Bfact(T, pdosAl27)["Al27"].round(6))
        0.274871

        >>> T = 80
        >>> float(Al.get_Bfact(T, pdosAl27)["Al27"].round(6))
        0.337081

        >>> T = 296
        >>> UO2.get_Bfact(T).round(6)
        O16     0.468604
        U238    0.253845
        dtype: float64

        >>> T = 400
        >>> UO2.get_Bfact(T).round(6)
        O16     0.595531
        U238    0.340297
        dtype: float64
        """
        if pdosDict is not None:
            self.compute_pdos(pdosDict)
        elif self.pdos is None:
            raise ValueError("The pdosDict must be defined or initialized in the object.")
        Bfact = BfacUnitChange * self.pdos.apply(lambda x: x.DebyeWallerCoeff(T))
        Bfact /= T * self.atoms.apply(lambda x: x.M)
        return Bfact * 1.0e20 if anstrom else Bfact

    def get_multiplicity(self, energyCut: float, T: float,
                         precision: list = [6, 6], **kwargs) -> pd.DataFrame:
        """
        Obtain hkl data for the solid in a certain temperature and for a neutron
        certain energy filtering with the multiplicity.

        Parameters
        ----------
        T : 'float'
            Temperature in K
        energyCut : 'float'
            Energy limit for d espace limit in eV
        precision: ['int', 'int'], optional
            Precision to get the multiplicity for d_hkl and Fsq_hkl. The
            default is [6, 6].

        Parameters for get_Bfact
        ------------------------
        pdosDict : 'dict[Pdos]', optional
            Dictionary with the pdos for each element. The default is None.
        anstrom : 'bool', optional
            Option to obtain the B unit in A^2. The default is True.

        Returns
        -------
        "pd.DataFrame", (N, 4)
            Multiplicity, d, Fsq and orientation of bragg planes.

        Examples
        --------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> from solid_cinel.tests.materials.Al27.examples import rho_in_energy, interv_in_energy
        >>> pdosAl27 = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> Al = TargetMat(unit_pos, dir_vec_length, dir_vec_angles, preferred_orientation, A, Z, atomic_mass, b_coh, b_incoh, pdosAl27)

        Test the results:
        >>> T = 20
        >>> energyCut = 2.301
        >>> multiplicity = Al.get_multiplicity(energyCut, T)
        >>> multiplicity.shape[0]
        678
        >>> multiplicity.iloc[:10] #doctest: +NORMALIZE_WHITESPACE
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

        >>> multiplicity.round(6).iloc[667:677] #doctest: +NORMALIZE_WHITESPACE
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
        hkl_data = numba_hkl_data(Neutron(energyCut).d_min,
                                  self.reciproc_vec.values,
                                  self.get_Bfact(T, **kwargs),
                                  self.atom_pos,
                                  self.atoms.apply(lambda x: x.b["b_coh"]),
                                  self.preferred_orientation.values,
                                  precision)
        return hkl_data.sort_values(by=["h", "k", "l"])\
                       .set_index(["h", "k", "l"])

    @staticmethod
    def _get_pddf(data: pd.DataFrame, kind: str = None, pddf_val: str = None) -> pd.DataFrame:
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
        data: 'pd.DataFrame', (N, M)
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

        Returns
        -------
        "pd.DataFrame", (N, M + 1)
            Pole Density Distribution Function column

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> from solid_cinel.tests.materials.Al27.examples import rho_in_energy, interv_in_energy
        >>> pdosAl27 = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> Al = TargetMat(unit_pos, dir_vec_length, dir_vec_angles, preferred_orientation, A, Z, atomic_mass, b_coh, b_incoh, pdosAl27)
        >>> T = 20
        >>> energyCut = 2.301
        >>> multiplicity = Al.get_multiplicity(energyCut, T)
        >>> multiplicity.iloc[:10] #doctest: +NORMALIZE_WHITESPACE
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
        >>> TargetMat._get_pddf(multiplicity).iloc[:10] #doctest: +NORMALIZE_WHITESPACE
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

        >>> multiplicity = Al.get_multiplicity(energyCut, T)
        >>> TargetMat._get_pddf(multiplicity, kind='march-dollase', pddf_val=2).iloc[:10] #doctest: +NORMALIZE_WHITESPACE
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

        >>> multiplicity = Al.get_multiplicity(energyCut, T)
        >>> TargetMat._get_pddf(multiplicity, kind='altomare', pddf_val=[1, 1]).iloc[:10] #doctest: +NORMALIZE_WHITESPACE
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

        >>> multiplicity = Al.get_multiplicity(energyCut, T)
        >>> TargetMat._get_pddf(multiplicity, kind='cvc', pddf_val=[1, 1]).iloc[:10] #doctest: +NORMALIZE_WHITESPACE
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
        orientation_angle = np.deg2rad(data.loc[:, "Orientation angle"])

        if kind is None:
            data["Orientation angle"] = 0.
            data["PDDF"] = 1.

        elif kind.lower() == 'march-dollase' and isinstance(pddf_val, (int, float)):
            value = (pddf_val * np.cos(orientation_angle)) ** 2
            value += np.sin(orientation_angle) ** 2 / pddf_val
            data["PDDF"] = value ** (-1.5)

        elif kind.lower() == 'altomare' and len(pddf_val) == 2:
            value = pddf_val[0] * np.cos(2 * orientation_angle)
            data["PDDF"] = np.exp(value) + pddf_val[1]

        elif kind.lower() == 'cvc' and len(pddf_val) == 2:
            value = pddf_val[0] * (1 - np.cos(orientation_angle) ** pddf_val[1])
            data["PDDF"] = np.exp(- value)

        else:
            ValueError("Introduced kind is not available")
        return data

    @staticmethod
    def _get_difracAngles(data: pd.DataFrame, energyCut: float) -> pd.DataFrame:
        """
        Add to the hkl data dataframe the difraction angles(ª) vs hkl data.
        .. math::
            2\theta_{hkl}=\arccos\left(1-\dfrac{\pi^2\hbar^2}{md_{hkl}^2E}\right)

        Parameters
        ----------
        data: 'pd.DataFrame', (N, M)
            Frame with hkl data.
        energyCut : 'float'
            Energy cut for d espace in eV

        Returns
        -------
        "pd.DataFrame", (N, M + 1)
            Difraction angle column.

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> from solid_cinel.tests.materials.Al27.examples import rho_in_energy, interv_in_energy
        >>> pdosAl27 = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> Al = TargetMat(unit_pos, dir_vec_length, dir_vec_angles, preferred_orientation, A, Z, atomic_mass, b_coh, b_incoh, pdosAl27)
        >>> T = 20
        >>> energyCut = 2.301
        >>> multiplicity = Al.get_multiplicity(energyCut, T)
        >>> multiplicity.iloc[:10] #doctest: +NORMALIZE_WHITESPACE
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
        >>> TargetMat._get_difracAngles(multiplicity, energyCut).iloc[:10] #doctest: +NORMALIZE_WHITESPACE
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
        d = data.loc[:, "d"]
        angle_value = np.clip(1 - pi ** 2 * BraggUnitChange / (d ** 2 * energyCut),
                              -1, 1)
        data["theta"] = np.rad2deg(np.arccos(angle_value))
        return data

    @staticmethod
    def _get_BraggEdgesXs(data: pd.DataFrame, unit_cell_vol: float,
                           atom_number: int,
                           threshold: float = 1.e-30) -> pd.DataFrame:
        """
        Add to the hkl data dataframe the cross section related with the
        Bragg Edges.
        .. math::
            \sigma_{\textrm{coh}}^{\textrm{el}}(E_{hkl})=\dfrac{\pi^2\hbar^2}{mN_{uc}V_{uc}E_{hkl}}M_{hkl}d_{hkl}\left|F(\vec{\tau}_{hkl})\right|^2\mathcal{P}_{hkl}(\Theta_{hkl})

        Parameters
        ----------
        data: 'pd.DataFrame', (N, M)
            Frame with hkl data.
        unit_cell_vol : 'float'
            Unit cell volume.
        atom_number : 'int'
            The number of atoms in the unit cell.
        threshold : 'float', optional
            Minimun value of xs. The default is 1.e-30.

        Returns
        -------
        "pd.DataFrame", (N, M + 1)
            Xs related to the Bragg Edges.

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> from solid_cinel.tests.materials.Al27.examples import rho_in_energy, interv_in_energy
        >>> pdosAl27 = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> Al = TargetMat(unit_pos, dir_vec_length, dir_vec_angles, preferred_orientation, A, Z, atomic_mass, b_coh, b_incoh, pdosAl27)
        >>> T = 20
        >>> energyCut = 2.301
        >>> unit_cell_vol = Al.unit_cell_vol
        >>> atom_number = Al.atom_number
        >>> BraggEdges = Al.get_BraggEdges(energyCut, T, xs=False, theta=False)
        >>> BraggEdges.iloc[:10]  #doctest: +NORMALIZE_WHITESPACE
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
        >>> TargetMat._get_BraggEdgesXs(BraggEdges, unit_cell_vol, atom_number).loc[::, "Xs"].iloc[:10]  #doctest: +NORMALIZE_WHITESPACE
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
        if "PDDF" not in data.columns:
            TargetMat.get_pddf(data)
        data["Xs"] = data["d"] * data["Fsq"] * data["Multiplicity"] * data["PDDF"]
        data["Xs"] *= BraggUnitChange * pi ** 2 / (unit_cell_vol * atom_number)
        if threshold:
            data.loc[data["Xs"] < threshold, "Xs"] = 0.0
        return data

    def get_BraggEdges(self, *args, xs: bool = True, file_BraggEdges: str = None,
                       theta: bool = True, **kwargs) -> pd.DataFrame:
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
        energyCut : 'float'
            Energy cut for d espace in eV
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
        "pd.DataFrame", (N, M)
            DataFrame with the selected information.

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> from solid_cinel.tests.materials.Al27.examples import rho_in_energy, interv_in_energy
        >>> pdosAl27 = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> Al = TargetMat(unit_pos, dir_vec_length, dir_vec_angles, preferred_orientation, A, Z, atomic_mass, b_coh, b_incoh, pdosAl27)
        >>> T = 20
        >>> energyCut = 2.301

        Test the results:
        >>> Al.get_BraggEdges(energyCut, T).round(6).iloc[:10, :4] #doctest: +NORMALIZE_WHITESPACE
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

        >>> Al.get_BraggEdges(energyCut, T).round(6).iloc[:10, 4::] #doctest: +NORMALIZE_WHITESPACE
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
        precision = kwargs.pop("precision", [6, 6])
        data = self.get_multiplicity(*args, precision=precision)

        # Get PDDF:
        self._get_pddf(data, kwargs.pop("kind", None), kwargs.pop("pddf_val", None))

        # Get Bragg Edges:
        data["E"] = pi ** 2 * BraggUnitChange / (2 * data["d"] ** 2)
        data.sort_values(by=["E"], inplace=True)

        # Optional argument:
        # Xs:
        if xs:
            self._get_BraggEdgesXs(data, self.unit_cell_vol, self.atom_number,
                                   threshold=kwargs.get("threshold", 1.e-30))

        # difraction angles vs hkl data
        if theta:
            self._get_difracAngles(data, args[0])

        # Get the final result
        if file_BraggEdges:
            data.to_csv(file_BraggEdges, sep='\t', float_format="%20.10e")
        return data

    def get_XsCoh(self, energyCut: float, *args, file_Xs: str = None, **kwargs) -> pd.Series:
        """
        Get coherent Xs.

        Parameters
        ----------
        energyCut : 'float'
            energy cut-off, above which the peak will not be stored.
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
        "pd.DataFrame", (N, M)
            Dataframe with the coherent xs for each atom of the objetc.

        Examples
        --------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> from solid_cinel.tests.materials.Al27.examples import rho_in_energy, interv_in_energy
        >>> pdosAl27 = Pdos.from_dE(rho_in_energy, interv_in_energy)
        >>> Al = TargetMat(unit_pos, dir_vec_length, dir_vec_angles, preferred_orientation, A, Z, atomic_mass, b_coh, b_incoh, pdosAl27)
        >>> from solid_cinel.data.materials.UO2 import *
        >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy, interv_in_energy
        >>> pdosUO2 = {"O16": Pdos.from_dE(rho_in_energy[0], interv_in_energy[0]), "U238": Pdos.from_dE(rho_in_energy[1], interv_in_energy[1])}
        >>> UO2 = TargetMat(unit_pos, dir_vec_length, dir_vec_angles, preferred_orientation, A, Z, atom_mass, b_coh, b_incoh, pdosUO2)

        Test the results:
        >>> T = 20
        >>> energyCut = 2.301
        >>> Al.get_XsCoh(energyCut, T).round(6).iloc[:10] #doctest: +NORMALIZE_WHITESPACE
        E
        0.003759    1.428610
        0.005012    1.761554
        0.010024    1.352587
        0.013783    1.554352
        0.015036    1.590366
        0.020048    1.270746
        0.023807    1.305106
        0.025060    1.455627
        0.030072    1.371732
        0.033831    1.392236
        Name: Xs, dtype: float64

        >>> UO2.get_XsCoh(energyCut, T).round(6).iloc[:10] #doctest: +NORMALIZE_WHITESPACE
        E
        0.000664    0.000000
        0.001329    0.000000
        0.001993    3.049080
        0.002658    2.327766
        0.003322    1.862213
        0.003987    1.551844
        0.005316    6.027579
        0.005980    5.357848
        0.006645    4.822063
        0.007309    5.677412
        Name: Xs, dtype: float64
        """
        # Get the Bragg Edges
        BraggEdgesXs = self.get_BraggEdges(energyCut, *args, **kwargs)

        # Extract the energy and cross-section information and convert to a Series
        xs = BraggEdgesXs.loc[:, ["E", "Xs"]].set_index("E").iloc[::, 0]

        # Filter out data where energy exceeds the cut-off
        xs = xs[xs.index <= energyCut]

        # Sum the data if there are duplicate energy values
        if xs.index.has_duplicates:
            xs = xs.groupby(level=0).sum()

        # Calculate the cumulative sum of the cross-sections and normalize by energy
        xs = xs.cumsum() / xs.index.values

        # Save the data to a file if a filename is provided
        if file_Xs:
            xs.to_csv(file_Xs, sep='\t', float_format="%20.10e")

        return xs


def numba_hkl_data(d_min: float, rec_vecs: np.ndarray, Bfac: pd.Series,
                   pos: pd.Series, csl: pd.Series, preferred_orientation: pd.Series,
                   precision: list) -> pd.DataFrame:
    """
    Obtain hkl data for the solid in a certain temperature and for a neutron
    certain energy.
    2 atoms test in test folder.

    Parameters
    ----------
    d_min : 'float'
        The minimum dspacing for the LEAPR module of NJOY
    rec_vecs : 'np.ndarray', (3, 3)
        Reciprocal vectors
    Bfac : 'pd.Series'
        Pandas series with the B factor for TargetMaterial object elements.
    pos : 'pd.Series'
        Pandas series with atomic position of elements in TargetMaterial
        object.
    csl : 'pd.Series'
        Coherent elastic length for each element of TargetMaterial object.
    precision: 'list', (2,):
        Array containing:
            0: Precision to reagroup in multiplicity the d_hkl
            1: Precision to reagroup in multiplicity the Fsq_hkl

    Returns
    -------
    "pd.DataFrame"
        Dataframe containing the hkl information.

    Examples
    --------
    Object initialization:

    >>> unit_pos_Al27 = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.25, 0.25,0.75, 0.25])
    >>> from solid_cinel.data.materials.Al27 import *
    >>> from solid_cinel.tests.materials.Al27.examples import rho_in_energy, interv_in_energy
    >>> pdosAl27 = Pdos.from_dE(rho_in_energy, interv_in_energy)
    >>> Al = TargetMat(unit_pos_Al27, dir_vec_length, dir_vec_angles, preferred_orientation, A, Z, atomic_mass, b_coh, b_incoh, pdosAl27)
    >>> from solid_cinel.data.materials.UO2 import *
    >>> from solid_cinel.tests.materials.UO2_O16_U238.examples import rho_in_energy, interv_in_energy
    >>> pdosUO2 = {"O16": Pdos.from_dE(rho_in_energy[0], interv_in_energy[0]), "U238": Pdos.from_dE(rho_in_energy[1], interv_in_energy[1])}
    >>> UO2 = TargetMat(unit_pos, dir_vec_length, dir_vec_angles, preferred_orientation, A, Z, atom_mass, b_coh, b_incoh, pdosUO2)

    Test the results:
    >>> T = 20
    >>> E = 2.301
    >>> recs_vec = Al.reciproc_vec.values
    >>> d_min = Neutron(E).d_min
    >>> B = Al.get_Bfact(T)
    >>> pos = Al.atom_pos
    >>> csl = Al.atoms.apply(lambda x: x.b["b_coh"])
    >>> precision = np.array([6, 6])
    >>> preferred_orientation = Al.preferred_orientation.values
    >>> hkl_data = numba_hkl_data(d_min, recs_vec, B, pos, csl, preferred_orientation, precision)
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

    >>> T = 296
    >>> E = 6.85e-1
    >>> recs_vec = UO2.reciproc_vec.values
    >>> d_min = Neutron(E).d_min
    >>> B = UO2.get_Bfact(T)
    >>> pos = UO2.atom_pos
    >>> csl = UO2.atoms.apply(lambda x: x.b["b_coh"])
    >>> precision = np.array([6, 6])
    >>> preferred_orientation = UO2.preferred_orientation.values
    >>> hkl_data = numba_hkl_data(d_min, recs_vec, B, pos, csl, preferred_orientation, precision)
    >>> hkl_data.round(6).iloc[:10]
        h  k  l         d       Fsq  Orientation angle  Multiplicity
    0  33  7  2  0.164168  0.000000          86.607081         216.0
    1  33  7  1  0.164384  0.108687          88.302052         384.0
    2  33  7  0  0.164456  0.000000          90.000000         144.0
    3  33  6  4  0.164240  0.000000          83.199199         288.0
    4  33  6  3  0.164746  0.000000          84.888910         432.0
    5  33  6  2  0.165110  0.000000          86.587580         192.0
    6  33  6  1  0.165330  0.000000          88.292276         264.0
    7  33  6  0  0.165404  0.000000          90.000000         432.0
    8  33  5  4  0.165037  0.000000          83.166021         528.0
    9  33  5  3  0.165551  0.116100          84.863872         120.0
    """
    # hkl_max calculation
    hkl_max = hkl_max_value(rec_vecs, d_min)

    # Preparation of variables to be accept in numba nopython mode:
    keys, values = nb.core.types.unicode_type, nb.core.types.float64
    Bfac_ = nb.typed.Dict.empty(key_type=keys, value_type=values)
    pos_ = nb.typed.Dict.empty(key_type=keys, value_type=values[:, :])
    csl_ = nb.typed.Dict.empty(key_type=keys, value_type=values)
    for element in Bfac.index:
        Bfac_[element], csl_[element] = Bfac[element], csl[element]
        pos_[element] = pos[element].values
    preferred_orientation_ = np.array(preferred_orientation, dtype=float)

    # Execute numba
    hkl_data_dict = hklloop(d_min, hkl_max, rec_vecs, Bfac_, pos_, csl_,
                            preferred_orientation_, np.array(precision))

    # Order the output
    columns = ["h", "k", "l", "d", "Fsq", "Orientation angle", "Multiplicity"]
    values = [[h, k, l, d_hkl, Fsq_hkl, orientation, mul] for (h, k, l), [d_hkl, Fsq_hkl, orientation, mul] in hkl_data_dict.items()]
    return pd.DataFrame(values, columns=columns)


@nb.jit(nopython=True, cache=True)
def hklloop(d_min: float, hkl_max: np.ndarray, rec_vecs: np.ndarray,
            Bfac: dict, pos: dict, csl: dict, preferred_orientation: np.ndarray,
            precision: np.ndarray) -> dict:
    """
    Get the F_hkl and d_hkl for all the posible h, k, l plane combination that
    fill the condition of d_hkl > d_min.
    .. math::
        d_{hkl} = \frac{2\pi}{\tau_{hkl}}
        F(\vec{\tau}_{hkl})=\sum_{j=1}^{N_{uc}}b_j\exp\left(-\dfrac{\hbar^2\tau_{hkl}^2}{4M_jk_BT}\Lambda_j(T)\right) e^{i\vec{\tau}_{hkl}\cdot\vec{p}_j}

    Parameters
    ----------
    d_min : 'float'
        The minimum dspacing for the LEAPR module of NJOY
    hkl_max : 'np.ndarray', (3,)
        Maximun h, k, l index for generating a d > d_min
    rec_vecs : 'np.ndarray' (3, 3)
        Reciprocal vectors
    Bfac : 'nb.typed.Dict'
        Dict with the B factor for TargetMaterial object elements.
    pos : 'nb.typed.Dict'
        Dict with atomic position of elements in TargetMaterial object.
    csl : 'nb.typed.Dict'
        Coherent elastic length for each element of TargetMaterial object.
    preferred_orientation: "np.ndarray", (3)
        Array with the preferred orientation of the solid.
    precision: "float"
        Precision of the rounding in the calculation to merge different plane
        values

    Returns
    -------
    "dict"
        Dictionary containing the hkl planes, the d_hkl, Fsq, orientation_angle.
    """
    # Get the output variables:
    hklM = {}
    hkldF = {}

    # Get the orientation norm:
    orientation_norm = np.linalg.norm(preferred_orientation)

    # Loop over the hkl planes:
    h_range, k_range, l_range = [np.arange(-x, x + 1) for x in hkl_max]

    # Loop over the hkl planes:
    for h in h_range[::-1]:  # to get positive hkl order
        for k in k_range[::-1]:
            for l in l_range[::-1]:
                if h ** 2 + k ** 2 + l ** 2 == 0:  # (0, 0, 0) is excluded
                    continue

                # Get the d_hkl and Fsq_hkl
                vec_tau_hkl = h * rec_vecs[0] + k * rec_vecs[1] + l * rec_vecs[2]
                d_hkl = 2 * np.pi / np.linalg.norm(vec_tau_hkl)

                # Check if d_hkl is greater than d_min
                if d_hkl < d_min:
                    continue

                # Get the Fsq_hkl squared
                Fsq = Fsq_hkl(vec_tau_hkl, Bfac, csl, pos)

                # same dspacing and Fsquared with precision will be regrouped
                d_rnd, Fsq_rnd = round(d_hkl, precision[0]), round(Fsq, precision[1])

                # Write the output
                if (d_rnd, Fsq_rnd) in hkldF:
                    hklM[hkldF[(d_rnd, Fsq_rnd)]][-1] += 1
                else:
                    hkldF[(d_rnd, Fsq_rnd)] = (h, k, l)
                    OA_num = np.sum(vec_tau_hkl * preferred_orientation)
                    OA_den = np.linalg.norm(vec_tau_hkl) * orientation_norm
                    hklM[(h, k, l)] = np.array([d_hkl, Fsq,
                                                acos(OA_num / OA_den) * 180 / pi,
                                                1])
    return hklM


@nb.jit(nopython=True, cache=True)
def Fsq_hkl(vec_tau_hkl: np.ndarray, Bfac: dict, csl: dict, pos: dict) -> float:
    """
    Get F_hkl:
    .. math::
        F(\vec{\tau}_{hkl})=\sum_{j=1}^{N_{uc}}b_j\exp\left(-\dfrac{\hbar^2\tau_{hkl}^2}{4M_jk_BT}\Lambda_j(T)\right) e^{i\vec{\tau}_{hkl}\cdot\vec{p}_j}

    Parameters
    ----------
    vec_tau_hkl : 'np.ndarray', (3, 3)
        Reciprocal vectors
    Bfac : 'nb.typed.Dict'
        Dict with the B factor for TargetMaterial object elements.
    pos : 'nb.typed.Dict'
        Dict with atomic position of elements in TargetMaterial object.
    csl : 'nb.typed.Dict'
        Coherent elastic length for each element of TargetMaterial object.

    Returns
    -------
    "float"
        Fsq_hkl value for that (h, k, l) plane
    """
    real = 0.
    imag = 0.
    constant = - 0.5 * np.linalg.norm(vec_tau_hkl) ** 2 / (8 * pi ** 2)
    for element in Bfac:
        for iep in range(len(pos[element])):
            real += cos(np.sum(vec_tau_hkl * pos[element][iep]))
            imag += sin(np.sum(vec_tau_hkl * pos[element][iep]))
        expon_hkl = exp(constant * Bfac[element])
        real *= csl[element] * 0.1 * expon_hkl
        imag *= csl[element] * 0.1 * expon_hkl
    return real ** 2 + imag ** 2
