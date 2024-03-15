# -*- coding: utf-8 -*-
"""
Python file for working with Target Material.

@author: AB272525
"""

from solid_cinel.core.material.structure.solid import Solid, hkl_max_value
from solid_cinel.core.material.vibration.pdos import Pdos
from solid_cinel.core.scattering_function import Beta
from solid_cinel.core.scattering_function.alpha import Alpha
from solid_cinel.core.scattering_function import Sab
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
Bfac_unit_change = (4 * c ** 2 * pi**2) * h ** 2 / (m_to_eV * kb)
Bragg_unit_change = 1.0e20 * h ** 2 * c ** 2 / (mn_to_MeV * 1.0e6)

# Examples variables:
# 1 atom:
rho_in_energy_str_Al27 = '''
        0 .0066 .0264 .0594 .1055 .1649 .2374 .3232 .4221
        .5342 .6595 .7980 .9497 1.1146 1.2927 1.4839 1.6884
        2.0169 2.4373 2.9366 3.6133 4.6775 7.1346 7.3650
        7.5156 7.6733 7.8309 8.0740 8.4419 9.0595 9.6773
        7.3645 6.2674 5.1965 4.7958 4.8024 4.6841 4.4673
        4.1914 3.8169 3.3439 2.7855 3.2782 5.3082 8.5930
        12.3377 8.4616 5.6695 4.1585 2.6081 0.0
    '''
rho_in_energy_Al27 = np.fromstring(rho_in_energy_str_Al27, dtype=np.float64, sep=' ')
interv_in_energy_Al27 = 0.0008
alpha0_str_Al27 = '''
  .005 .010 .015 .020 .025 .030 .035 .040 .045 .050
  .060 .070 .080 .090 .100 .125 .150 .175 .200 .225
  .250 .275 .300 .325 .350 .375 .400 .425 .450 .475
  .500 .525 .550 .575 .600 .625 .675 .700 .725 .750
  .800 .850 .900 .950 1.00 1.05 1.10 1.15 1.20 1.25
  1.30 1.35 1.40 1.50 1.60 1.70 1.80 1.90 2.00 2.10
  2.20 2.30 2.40 2.50 2.60 2.70 2.80 2.90 3.00 3.10
  3.20 3.30 3.40 3.50 3.60 3.80 4.00 4.20 4.40 4.60
  4.80 5.00 5.20 5.40 5.60 5.80 6.00 6.20 6.40 6.60
  6.80 7.00 7.40 7.80 8.20 8.60 9.00 9.40 9.80 10.2
  10.6 11.0 11.5 12.0 12.5 13.0 13.5 14.0 14.5 15.0
  15.5 16.0 16.5 17.0 17.5 18.0 18.5 19.0 19.5 20.0
  21.0 22.0 23.0 24.0 24.5 25.0 26.0 27.0 28.0 29.0
  30.0 32.5 35.0 37.5 40.0 42.5 45.0 47.5 50.0 52.5
  55.0 57.5 60.0 62.5 65.0 67.5 70.0 72.5 75.0
'''
alpha0_Al27 = np.fromstring(alpha0_str_Al27, dtype=np.float64, sep=' ')
beta0_str_Al27 = '''
  .000 .025 .050 .075 .100 .125 .150 .175 .200 .225
  .250 .275 .300 .325 .350 .375 .400 .425 .450 .475
  .500 .525 .550 .575 .600 .625 .650 .675 .700 .725
  .750 .775 .800 .825 .850 .875 .900 .925 .950 .975
  1.00 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45
  1.50 1.55 1.60 1.70 1.80 1.90 2.00 2.10 2.20 2.30
  2.40 2.50 2.60 2.70 2.80 2.90 3.00 3.10 3.20 3.30
  3.40 3.50 3.60 3.70 3.80 3.90 4.00 4.10 4.20 4.30
  4.40 4.50 4.60 4.70 4.80 4.90 5.00 5.10 5.20 5.30
  5.40 5.50 5.60 5.70 5.80 5.90 6.00 6.25 6.50 6.75
  7.00 7.50 8.00 8.50 9.00 10.0 11.0 12.0 13.0 14.0
  15.0 16.0 17.0 18.0 19.0 20.0 22.5 25.0 27.5 30.0
  32.5 35.0 37.5 40.0 42.5 45.0 47.5 50.0 52.5 55.0
  57.5 60.0 62.5 65.0 67.5 70.0 72.5 75.0 77.5 80.0
  82.5 85.0 87.5 90.0
'''
beta0_Al27 = np.fromstring(beta0_str_Al27, dtype=np.float64, sep=' ')

# 2 atom:
rho_in_energy_O16_str = '''
0.000000E+00 6.923874E-03 2.497670E-02 5.488348E-02
9.504920E-02 1.479389E-01 2.139513E-01 2.889902E-01
3.722217E-01 4.694096E-01 5.797566E-01 7.142103E-01
8.648680E-01 1.037820E+00 1.270585E+00 1.816629E+00
1.726038E+00 1.064790E+00 8.615329E-01 8.017556E-01
7.808027E-01 7.815640E-01 7.991818E-01 8.341637E-01
8.819783E-01 9.379855E-01 1.239156E+00 2.544143E+00
5.493169E+00 7.542890E+00 6.200347E+00 3.899265E+00
3.176137E+00 3.982750E+00 5.293972E+00 5.997778E+00
6.310875E+00 6.750886E+00 7.348876E+00 9.615086E+00
1.342006E+01 1.785015E+01 2.514254E+01 3.308876E+01
3.394329E+01 3.018225E+01 2.677197E+01 2.401069E+01
2.264379E+01 2.170413E+01 2.108204E+01 2.061835E+01
2.034441E+01 2.020180E+01 2.021090E+01 2.032422E+01
2.048390E+01 2.089081E+01 2.149127E+01 2.224180E+01
2.335361E+01 2.456841E+01 2.400029E+01 2.378412E+01
2.057435E+01 1.340375E+01 1.410426E+01 1.986035E+01
2.335756E+01 2.336811E+01 2.139441E+01 2.074719E+01
2.063757E+01 2.095150E+01 2.158940E+01 2.254504E+01
2.422993E+01 2.691331E+01 3.179124E+01 4.563659E+01
5.752641E+01 5.609342E+01 4.689648E+01 2.438494E+01
4.135546E+00 0.000000E+00 0.000000E+00 0.000000E+00
0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
0.000000E+00 5.255256E-01 2.237281E+00 4.016499E+00
5.186199E+00 6.762413E+00 8.981823E+00 1.161543E+01
1.599614E+01 3.174364E+01 3.582525E+01 2.322394E+01
1.668530E+01 1.262719E+01 1.004853E+01 8.184454E+00
6.946893E+00 6.009886E+00 5.284656E+00 4.766380E+00
4.441574E+00 4.272284E+00 3.664723E+00 2.362237E+00
8.264423E-01 1.493527E-02 0.000000E+00'''
rho_in_energy_O16 = np.fromstring(rho_in_energy_O16_str, dtype=np.float64,
                                  sep=' ')
interv_in_energy_O16 = interv_in_energy_U238 = 6.956193E-04
rho_in_energy_U238_str = '''
0.000000E+00 1.041128E-01 3.759952E-01 8.354039E-01
1.469796E+00 2.335578E+00 3.467660E+00 4.841392E+00
6.492841E+00 8.608376E+00 1.131303E+01 1.504441E+01
2.006807E+01 2.750471E+01 4.171597E+01 1.585670E+02
1.978483E+02 1.144621E+02 7.555927E+01 4.831100E+01
4.389081E+01 4.246484E+01 4.103699E+01 3.986249E+01
3.827959E+01 3.592088E+01 3.272170E+01 3.914602E+01
8.144694E+01 9.693959E+01 5.503795E+01 2.619253E+01
1.763331E+01 1.475875E+01 1.522465E+01 1.213117E+01
6.175029E+00 2.483519E+00 1.445581E+00 1.423177E+00
1.502350E+00 1.718768E+00 2.211346E+00 3.061686E+00
3.550530E+00 3.349917E+00 2.768379E+00 2.177488E+00
1.856123E+00 1.622775E+00 1.445254E+00 1.300794E+00
1.180078E+00 1.075748E+00 9.928057E-01 9.238564E-01
8.577708E-01 8.073819E-01 7.634820E-01 7.172257E-01
6.728183E-01 6.251482E-01 5.496737E-01 4.992486E-01
3.945195E-01 2.206960E-01 1.452214E-01 1.246671E-01
9.863893E-02 7.855588E-02 6.536053E-02 6.568678E-02
7.308199E-02 8.388478E-02 1.026265E-01 1.245221E-01
1.487740E-01 1.757085E-01 2.055793E-01 2.473042E-01
3.128097E-01 3.455081E-01 3.048708E-01 1.621507E-01
2.653572E-02 0.000000E+00 0.000000E+00 0.000000E+00
0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
0.000000E+00 7.105193E-03 5.274518E-02 1.324974E-01
2.310275E-01 4.042710E-01 6.421137E-01 8.073457E-01
9.162074E-01 1.077923E+00 1.142595E+00 1.092532E+00
1.060668E+00 1.000020E+00 8.769838E-01 7.610532E-01
6.898200E-01 6.324347E-01 5.857072E-01 5.563076E-01
5.468099E-01 5.515587E-01 4.871045E-01 3.198787E-01
1.132118E-01 2.066306E-03 0.000000E+00
'''
rho_in_energy_U238 = np.fromstring(rho_in_energy_U238_str, dtype=np.float64,
                                   sep=' ')
rho_in_energy = [rho_in_energy_O16, rho_in_energy_U238]
interv_in_energy = [interv_in_energy_O16, interv_in_energy_U238]


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
    energy_cut : float
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
    get_coherent_Xs: pd.DataFrame
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
        if len(args) > 9:
            self.add_pdos(args[9:])
        # Avoid data setter in Pdos:
        elements_name = self.atoms.index
        if isinstance(args[10], float):
            pdos = {elements_name[0]: Pdos.from_dE(args[9], args[10])}
        else:
            pdos = {elements_name[i]: Pdos.from_dE(args[9][i], args[10][i])
                    for i in range(len(args[10]))}
        self.pdos = pd.Series(pdos)

    def add_pdos(self, pdosDict: dict[Pdos]):
        return

    def get_Bfact(self, T: float, anstrom: bool = True) -> [float, pd.Series]:
        """
        Calculate mean square displacement for a certain pdos information.
        .. math::
            B_j= \frac{4\pi^2\hbar^2}{M_jk_BT}\lambda_s

        Parameters
        ----------
        T : 'float'
            Temperature in K
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
        >>> Al = TargetMat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass, b_coh, b_incoh, rho_in_energy_Al27, interv_in_energy_Al27)
        >>> from solid_cinel.data.materials.UO2 import *
        >>> UO2 = TargetMat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atom_mass, b_coh, b_incoh, rho_in_energy, interv_in_energy)

        Test the results:
        >>> T = 20
        >>> Al.get_Bfact(T)["Al27"].round(6)
        0.274871

        >>> T = 80
        >>> Al.get_Bfact(T)["Al27"].round(6)
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

        Bfact = Bfac_unit_change * self.pdos.apply(lambda x: x.DebyeWallerCoeff(T))
        Bfact /= T * self.atoms.apply(lambda x: x.atom_mass)
        return Bfact * 1.0e20 if anstrom else Bfact

    def get_multiplicity(self, T: float, energy_cut: float,
                         precision: list = [6, 6]) -> pd.DataFrame:
        """
        Obtain hkl data for the solid in a certain temperature and for a neutron
        certain energy filtering with the multiplicity.

        Parameters
        ----------
        T : 'float'
            Temperature in K
        energy_cut : 'float'
            Energy limit for d espace limit in eV
        precision: ['int', 'int'], optional
            Precision to get the multiplicity for d_hkl and Fsq_hkl. The
            default is [6, 6].

        Returns
        -------
        "pd.DataFrame", (N, 4)
            Multiplicity, d, Fsq and orientation of bragg planes.

        Examples
        --------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> Al = TargetMat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass, b_coh, b_incoh, rho_in_energy_Al27, interv_in_energy_Al27)

        Test the results:
        >>> T = 20
        >>> E = 2.301
        >>> multiplicity = Al.get_multiplicity(T, E)
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
        recs_vec = self.reciproc_vec.values
        d_min = Neutron(energy_cut).d_min
        hkl_max = hkl_max_value(recs_vec, d_min)
        hkl_data = numba_hkl_data(d_min,
                                  hkl_max,
                                  recs_vec,
                                  self.get_Bfact(T),
                                  self.atom_pos,
                                  self.atoms.apply(lambda x: x.b["b_coh"]),
                                  self.preferred_orientation.values,
                                  np.array(precision)
                                  )
        return hkl_data.sort_values(by=["h", "k", "l"])\
                       .set_index(["h", "k", "l"])

    @staticmethod
    def _get_pddf(data: pd.DataFrame, kind: str = None,
                  pddf_val: str = None) -> pd.DataFrame:
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
        >>> Al = TargetMat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass, b_coh, b_incoh, rho_in_energy_Al27, interv_in_energy_Al27)
        >>> T = 20
        >>> E = 2.301
        >>> multiplicity = Al.get_multiplicity(T, E)
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

        >>> multiplicity = Al.get_multiplicity(T, E)
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

        >>> multiplicity = Al.get_multiplicity(T, E)
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

        >>> multiplicity = Al.get_multiplicity(T, E)
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
    def _get_difrac_angles(data: pd.DataFrame, energy_cut: float) -> pd.DataFrame:
        """
        Add to the hkl data dataframe the difraction angles(ª) vs hkl data.
        .. math::
            2\theta_{hkl}=\arccos\left(1-\dfrac{\pi^2\hbar^2}{md_{hkl}^2E}\right)

        Parameters
        ----------
        data: 'pd.DataFrame', (N, M)
            Frame with hkl data.
        energy_cut : 'float'
            Energy cut for d espace in eV

        Returns
        -------
        "pd.DataFrame", (N, M + 1)
            Difraction angle column.

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> Al = TargetMat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass, b_coh, b_incoh, rho_in_energy_Al27, interv_in_energy_Al27)
        >>> T = 20
        >>> E = 2.301
        >>> multiplicity = Al.get_multiplicity(T, E)
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
        >>> TargetMat._get_difrac_angles(multiplicity, E).iloc[:10] #doctest: +NORMALIZE_WHITESPACE
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
        angle_value = np.clip(1 - pi ** 2 * Bragg_unit_change / (d ** 2 * energy_cut),
                              -1, 1)
        data["theta"] = np.rad2deg(np.arccos(angle_value))
        return data

    @staticmethod
    def _get_BraggEdges_Xs(data: pd.DataFrame, unit_cell_vol: float,
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
        >>> Al = TargetMat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass, b_coh, b_incoh, rho_in_energy_Al27, interv_in_energy_Al27)
        >>> T = 20
        >>> E = 2.301
        >>> unit_cell_vol = Al.unit_cell_vol
        >>> atom_number = Al.atom_number
        >>> BraggEdges = Al.get_BraggEdges(T, E, xs=False, theta=False)
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
        >>> TargetMat._get_BraggEdges_Xs(BraggEdges, unit_cell_vol, atom_number).loc[::, "Xs"].iloc[:10]  #doctest: +NORMALIZE_WHITESPACE
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
        data["Xs"] *= Bragg_unit_change * pi ** 2 / (unit_cell_vol * atom_number)
        if threshold:
            data["Xs"][data["Xs"] < threshold] = 0.0
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
        energy_cut : 'float'
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
        >>> Al = TargetMat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass, b_coh, b_incoh, rho_in_energy_Al27, interv_in_energy_Al27)
        >>> T = 20
        >>> E = 2.301

        Test the results:
        >>> Al.get_BraggEdges(T, E).round(6).iloc[:10, :4] #doctest: +NORMALIZE_WHITESPACE
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

        >>> Al.get_BraggEdges(T, E).round(6).iloc[:10, 4::] #doctest: +NORMALIZE_WHITESPACE
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
        self._get_pddf(data,
                       kwargs.pop("kind", None),
                       kwargs.pop("pddf_val", None)
                       )

        # Get Bragg Edges:
        data["E"] = pi ** 2 * Bragg_unit_change / (2 * data["d"] ** 2)
        data = data.sort_values(by=["E"])

        # Optional argument:
        # Xs:
        if xs:
            self._get_BraggEdges_Xs(data,
                                    self.unit_cell_vol,
                                    self.atom_number,
                                    threshold=kwargs.get("threshold", 1.e-30)
                                    )

        # difraction angles vs hkl data
        if theta:
            self._get_difrac_angles(data,
                                    args[1]
                                    )

        # Get the final result
        if file_BraggEdges:
            data.to_csv(file_BraggEdges,
                        sep='\t',
                        float_format="%20.10e")
        return data

    def get_coherent_Xs(self, energy_cut: float, energy_sup: float, *args,
                        file_Xs: str = None, **kwargs) -> pd.DataFrame:
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
        >>> Al = TargetMat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass, b_coh, b_incoh, rho_in_energy_Al27, interv_in_energy_Al27)
        >>> T = 20
        >>> E = 2.301

        Test the results:
        >>> energy_cut = 2.301
        >>> energy_sup = 10.0
        >>> Al.get_coherent_Xs(energy_cut, energy_sup, T).round(6).iloc[:10] #doctest: +NORMALIZE_WHITESPACE
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
        BraggEdges_Xs = self.get_BraggEdges(*args, energy_cut, **kwargs)\
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
        xs = pd.concat([xs] * len(self.atoms), axis=1)
        xs.columns = pd.MultiIndex.from_product(
            [self.atoms.apply(lambda x: x.zam).values, [2]],
            names=["ZAM", "MT"])
        # Optional argument:
        if file_Xs:
            xs.to_csv(file_Xs,
                      sep='\t',
                      float_format="%20.10e")
        return xs


def numba_hkl_data(d_min: float,
                   hkl_max: np.ndarray,
                   rec_vecs: np.ndarray,
                   Bfac: pd.Series,
                   pos: pd.Series,
                   csl: pd.Series,
                   preferred_orientation: pd.Series,
                   precision: np.ndarray) -> pd.DataFrame:
    """
    Obtain hkl data for the solid in a certain temperature and for a neutron
    certain energy.
    2 atoms test in test folder.

    Parameters
    ----------
    d_min : 'float'
        The minimum dspacing for the LEAPR module of NJOY
    hkl_max : 'np.ndarray', (3)
        Maximun h, k, l index for generating a d>d_min
    rec_vecs : 'np.ndarray', (3, 3)
        Reciprocal vectors
    Bfac : 'pd.Series'
        Pandas series with the B factor for TargetMaterial object elements.
    pos : 'pd.Series'
        Pandas series with atomic position of elements in TargetMaterial
        object.
    csl : 'pd.Series'
        Coherent elastic length for each element of TargetMaterial object.
    precision: 'np.ndarray', (2,):
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
    >>> Al = TargetMat(preferred_orientation, unit_pos_Al27, dir_vec_length, dir_vec_angles, A, Z, atomic_mass, b_coh, b_incoh, rho_in_energy_Al27, interv_in_energy_Al27)
    >>> from solid_cinel.data.materials.UO2 import *
    >>> UO2 = TargetMat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atom_mass, b_coh, b_incoh, rho_in_energy, interv_in_energy)

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

    >>> T = 296
    >>> E = 6.85e-1
    >>> recs_vec = UO2.reciproc_vec.values
    >>> d_min = Neutron(E).d_min
    >>> hkl_max = hkl_max_value(recs_vec, d_min)
    >>> B = UO2.get_Bfact(T)
    >>> pos = UO2.atom_pos
    >>> csl = UO2.atoms.apply(lambda x: x.b["b_coh"])
    >>> precision = np.array([6, 6])
    >>> preferred_orientation = UO2.preferred_orientation.values
    >>> hkl_data = numba_hkl_data(d_min, hkl_max, recs_vec, B, pos, csl, preferred_orientation, precision)
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
        pos_[element] = pos.loc[element].values
        csl_[element] = csl[element]

    preferred_orientation_ = np.array(preferred_orientation, dtype=float)
    # Execute numba
    hkl_data_dict = hklloop(d_min, hkl_max, rec_vecs, Bfac_, pos_, csl_,
                            preferred_orientation_, precision)

    # Order the output
    columns = ["h", "k", "l", "d", "Fsq", "Orientation angle", "Multiplicity"]

    return pd.DataFrame([[h, k, l, d_hkl, Fsq_hkl, orientation, mul]
                         for (h, k, l), [d_hkl, Fsq_hkl, orientation, mul]
                         in hkl_data_dict.items()],
                        columns=columns)


@nb.jit(nopython=True, nogil=True)
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
    hklM = {}
    hkldF = {}
    orientation_norm = np.linalg.norm(preferred_orientation)
    h_range, k_range, l_range = [np.arange(-x, x + 1) for x in hkl_max]

    for h in h_range[::-1]:  # to get positive hkl order
        for k in k_range[::-1]:
            for l in l_range[::-1]:
                if h ** 2 + k ** 2 + l ** 2 == 0:  # (0, 0, 0) is excluded
                    continue

                # d_hkl:
                vec_tau_hkl = h * rec_vecs[0] + k * rec_vecs[1] + l * rec_vecs[2]
                d_hkl = 2 * np.pi / np.linalg.norm(vec_tau_hkl)

                if d_hkl < d_min:  # d < d_min is excluded
                    continue

                Fsq = Fsq_hkl(vec_tau_hkl, Bfac, csl, pos)  # Fsquared

                # same dspacing and Fsquared with precision will be regrouped
                d_rnd = round(d_hkl, precision[0])
                Fsq_rnd = round(Fsq, precision[1])
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


@nb.jit(nopython=True, nogil=True, cache=True)
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
    return real ** 2 + imag ** 2  # Fsquared
