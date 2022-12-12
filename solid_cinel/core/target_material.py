# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:46:42 2022

@author: Aitor Bengoechea
"""

from solid_cinel.core.material.solid import Solid, hkl_max_value
from solid_cinel.core.material.pdos import Pdos
from solid_cinel.core.s import S
from solid_cinel.core._numba import hklloop
from solid_cinel.cinematic.lab import Neutron
from scipy.constants import physical_constants as const
import scipy as sp
import numpy as np
import pandas as pd
import numba as nb
import collections
import pytest
collections.Callable = collections.abc.Callable


# Examples variables:
# 1 atom
rho_in_energy_Al27_str = '''
    0 .0066 .0264 .0594 .1055 .1649 .2374 .3232 .4221
    .5342 .6595 .7980 .9497 1.1146 1.2927 1.4839 1.6884
    2.0169 2.4373 2.9366 3.6133 4.6775 7.1346 7.3650
    7.5156 7.6733 7.8309 8.0740 8.4419 9.0595 9.6773
    7.3645 6.2674 5.1965 4.7958 4.8024 4.6841 4.4673
    4.1914 3.8169 3.3439 2.7855 3.2782 5.3082 8.5930
    12.3377 8.4616 5.6695 4.1585 2.6081 0.0
'''
rho_in_energy_Al27 = np.fromstring(rho_in_energy_Al27_str, dtype=np.float64,
                                   sep=' ')
interv_in_energy_Al27 = 0.0008

preferred_orientation_Al27 = np.array([0, 0, 1])
a_Al27 = 2.856710674519725
dir_vec_length_Al27 = [a_Al27, a_Al27, a_Al27]
dir_vec_angles_Al27 = [60, 60, 60]
unit_pos_Al27 = np.array([0.0, 0.0, 0.0])
A_Al27 = 27
Z_Al27 = 13
atomic_mass_Al27 = 26.98153433356103
b_coh_Al27 = 3.449
b_incoh_Al27 = 0.256

alpha0_str = '''
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
alpha0_ = np.fromstring(alpha0_str, dtype = np.float64, sep = ' ')
beta0_str = '''
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
beta0_ = np.fromstring(beta0_str, dtype = np.float64, sep = ' ')

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
preferred_orientation = np.array([0, 0, 1])
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
        >>> Al = Target_mat(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy_Al27, interv_in_energy_Al27)
        >>> UO2 = Target_mat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atom_mass, b_coh, b_incoh, rho_in_energy, interv_in_energy)

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
        constant = (4 * sp.constants.c ** 2 * np.pi**2) * const["reduced Planck constant in eV s"][0] ** 2
        constant /= const["atomic mass unit-electron volt relationship"][0] * const["Boltzmann constant in eV/K"][0]
        atom_masses = self.atoms.apply(lambda x: x.atom_mass)

        def get_Bfac(single_pdos):
            B = constant * single_pdos.DebyeWallerCoeff(T) / T
            if anstrom:
                B *= 1.0e20
            return B

        return self.pdos.apply(get_Bfac) / atom_masses

    def get_multiplicity(self, T, energy_cut, precision=[6, 6]) -> pd.DataFrame:
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

        Examples
        --------
        Object initialization:
        >>> Al = Target_mat(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy_Al27, interv_in_energy_Al27)

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
        return hkl_data.sort_values(by=["h", "k", "l"]).set_index(["h", "k", "l"])

    @staticmethod
    def _get_pddf(data, kind=None, pddf_val=None) -> pd.DataFrame:
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
        >>> Al = Target_mat(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy_Al27, interv_in_energy_Al27)
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
        >>> Target_mat._get_pddf(multiplicity).iloc[:10]
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
        >>> Target_mat._get_pddf(multiplicity, kind='march-dollase', pddf_val=2).iloc[:10]
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
        >>> Target_mat._get_pddf(multiplicity, kind='altomare', pddf_val=[1, 1]).iloc[:10]
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
        >>> Target_mat._get_pddf(multiplicity, kind='cvc', pddf_val=[1, 1]).iloc[:10]
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
    def _get_difrac_angles(data, energy_cut) -> pd.DataFrame:
        """
        Add to the hkl data dataframe the difraction angles(ª) vs hkl data
        .. math::
            2\theta_{hkl}=\arccos\left(1-\dfrac{\pi^2\hbar^2}{md_{hkl}^2E}\right)

        Parameters
        ----------
        data: 'pd.DataFrame'
            Frame with hkl data.
        energy_cut : 'float'
            Energy cut for d espace in eV

        Example
        -------
        Object initialization:
        >>> Al = Target_mat(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy_Al27, interv_in_energy_Al27)
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
        >>> Target_mat._get_difrac_angles(multiplicity, E).iloc[:10]
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
        angle_value = np.clip(1 - np.pi ** 2 * constant / (d ** 2 * energy_cut), -1, 1)
        data["theta"] = np.arccos(angle_value) * 180 / np.pi
        return data

    @staticmethod
    def _get_BraggEdges_Xs(data, unit_cell_vol, atom_number,
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
        >>> Al = Target_mat(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy_Al27, interv_in_energy_Al27)
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
        >>> Target_mat._get_BraggEdges_Xs(BraggEdges, unit_cell_vol, atom_number).loc[::, "Xs"].iloc[:10]
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
        Object initialization:
        >>> Al = Target_mat(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy_Al27, interv_in_energy_Al27)
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
        self._get_pddf(data,
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
        >>> Al = Target_mat(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy_Al27, interv_in_energy_Al27)
        >>> T = 20
        >>> E = 2.301

        Test the results:
        >>> energy_cut = 2.301
        >>> energy_sup = 10.0
        >>> Al.get_coherent_Xs(energy_cut, energy_sup, T).round(6).iloc[:10]
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

    def _get_Sab_multiple(self, *args, model, **kwargs) -> pd.Series:
        """
        Generate S(alpha, -beta) matrix for the selected multiple atoms. The
        available options are:
            model = "fgm": Free Gas Model
            model = "sct": Short Collision Time
            model = "phonon expansion": Phonon Expansion

        Parameters for Free Gas model
        -----------------------------
        alpha_grid : 'dict' of 1D iterable
            Alpha grid.
        beta_grid : 'dict' of 1D iterable
            beta grid.
        T : 'float', optional
            Option to scale beta and alpha grid with the method scale_grid. The
            default is None.
        w_t: 'dict', optional
            normalization for continuous (vibrational) part. For solid is 1.
        
        Parameters for Short Collision Time
        -----------------------------------
        alpha_grid :'dict' of 1D iterable
            Alpha grid.
        beta_grid : 'dict' of 1D iterable
            beta grid.
        T : 'float'
            Temperature in K.
        w_s: 'dict', optional
            normalization for continuous (vibrational) part. For solid is 1.
        
        Parameters for Phonon Expansion
        -------------------------------
        T : 'float'
            Temperature in K.
        alpha_grid : 'dict' of 1D iterable
            Alpha grid.
        beta_grid : 'dict' of 1D iterable
            beta grid.
        scale : 'bool', optional
            Option to scale beta and alpha grid with the method scale_grid. The
            default is False.
        threshold : 'dict', optional
            Minimun value to take into account. The default is 1.0e-14.
        nphonon : 'dict', optional
            Phonon expansion order. The default is 1000.

        Raises
        ------
        ValueError
            The model key is not in the available list.

        Returns
        -------
        Sab : 'pd.Series' of 'solid_cinel.core.s.S' objects
            S(alpha, -beta) matrix divide by a atom key.

        """
        index = self.pdos.index
        alpha_grid = args[0]
        beta_grid = args[1]
        if model.lower() == "fgm":
            w_t = {key: kwargs.get("w_t", 1).get(key, 1) for key in index}
            T = kwargs.pop("T", None)
            Sab = self.pdos.groupby(by=index)\
                      .apply(lambda x: S.from_fgm(alpha_grid[x.name], beta_grid[x.name],
                                                  w_t=w_t[x.name], T=T))
        else:
            T = args[2]
            scale = kwargs.pop("scale", False)
            if model.lower() == "sct":
                w_s = {key: kwargs.get("w_s", 1).get(key, 1) for key in index}
                Sab = self.pdos.groupby(by=index)\
                          .apply(lambda x: S.from_sct(alpha_grid[x.name], beta_grid[x.name], T, x,
                                                      w_s=w_s[x.name], scale=scale))
            elif model.lower() == "phonon expansion":
                threshold = {key: kwargs.get("threshold", 0.0).get(key, 0.0) for key in index}
                nphonon = {key: kwargs.get("nphonon", 1000).get(key, 1000) for key in index}
                Sab = self.pdos.groupby(by=index)\
                          .apply(lambda x: S.from_pdos(alpha_grid[x.name], beta_grid[x.name], T, x,
                                                       threshold=threshold[x.name], nphonon= nphonon[x.name], scale=scale))
            else:
                raise ValueError("The selected model is not available")
        return Sab

    def _get_Sab_single(self, *args, model, **kwargs) -> pd.Series:
        """
        Generate S(alpha, -beta) matrix for the selected atom. The
        available options are:
            model = "fgm": Free Gas Model
            model = "sct": Short Collision Time
            model = "phonon expansion": Phonon Expansion

        Parameters for Free Gas model
        -----------------------------
        alpha_grid : 1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.
        model : 'str', optional
            The model to calculate matrix values. The default is "FGM".
        T : 'float', optional
            Option to scale beta and alpha grid with the method scale_grid. The
            default is None.
        w_t: 'dict', optional
            normalization for continuous (vibrational) part. For solid is 1.
        
        Parameters for Short Collision Time
        -----------------------------------
        alpha_grid :1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.
        T : 'float'
            Temperature in K.
        w_s: 'dict', optional
            normalization for continuous (vibrational) part. For solid is 1.
        
        Parameters for Phonon Expansion
        -------------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        T : 'float'
            Temperature in K.
        alpha_grid : 1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.
        scale : 'bool', optional
            Option to scale beta and alpha grid with the method scale_grid. The
            default is False.
        threshold : 'float', optional
            Minimun value to take into account. The default is 1.0e-14.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Raises
        ------
        ValueError
            The model key is not in the available list.

        Returns
        -------
        Sab : 'pd.Series' of 'solid_cinel.core.s.S' objects
            S(alpha, -beta) matrix divide by a atom key.

        """
        if model.lower() == "fgm":
            Sab = self.pdos.apply(lambda x: S.from_fgm(*args, **kwargs))
        else:
            if model.lower() == "sct":
                method = S.from_sct
            elif model.lower() == "phonon expansion":
                method = S.from_pdos
            else:
                raise ValueError("The selected model is not available")
            Sab = self.pdos.apply(lambda x: method(*args, x, **kwargs))
        return Sab

    def get_Sab(self, *args, model="phonon expansion", **kwargs):
        """
        Generate S(alpha, -beta) matrix for the selected material. The
        available options are:
            model = "fgm": Free Gas Model
            model = "sct": Short Collision Time
            model = "phonon expansion": Phonon Expansion

        Parameters for Free Gas model
        -----------------------------
        alpha_grid : 1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.
        model : 'str', optional
            The model to calculate matrix values. The default is "FGM".
        T : 'float', optional
            Option to scale beta and alpha grid with the method scale_grid. The
            default is None.
        w_t: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Parameters for Short Collision Time
        -----------------------------------
        alpha_grid : 1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.
        T : 'float'
            Temperature in K.
        w_s: 'float', optional
            normalization for continuous (vibrational) part. For solid is 1.

        Parameters for Phonon Expansion
        -------------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        T : 'float'
            Temperature in K.
        alpha_grid : 1D iterable
            Alpha grid.
        beta_grid : 1D iterable
            beta grid.
        scale : 'bool', optional
            Option to scale beta and alpha grid with the method scale_grid. The
            default is False.
        threshold : 'float', optional
            Minimun value to take into account. The default is 1.0e-14.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Raises
        ------
        ValueError
            The model key is not in the available list.

        Returns
        -------
        Sab : 'pd.Series' of 'solid_cinel.core.s.S' objects
            S(alpha, -beta) matrix divide by a atom key.

        Object initialization:
        >>> Al = Target_mat(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy_Al27, interv_in_energy_Al27)

        Test the results:
        FGM:
        >>> T = 300
        >>> from solid_cinel.core.s import gen_beta, gen_alpha
        >>> beta_grid = gen_beta(300)
        >>> alpha_grid = gen_alpha(300, 26)
        >>> Al.get_Sab(alpha_grid, beta_grid, model="fgm")["Al27"].data.iloc[:10, :5].round(6)
        beta	      0.000000	0.012894	0.025788	0.038682	0.051576
        alpha
        0.001050	8.701463	8.417992	7.524148	6.213536	4.740815
        0.001087	8.553363	8.285768	7.435678	6.181592	4.760714
        0.001125	8.407781	8.155251	7.346923	6.147319	4.777252
        0.001164	8.264674	8.026439	7.257961	6.110841	4.790511
        0.001205	8.124000	7.899326	7.168869	6.072279	4.800575
        0.001247	7.985718	7.773908	7.079717	6.031753	4.807533
        0.001291	7.849787	7.650178	6.990574	5.989379	4.811476
        0.001336	7.716166	7.528129	6.901504	5.945271	4.812500
        0.001382	7.584817	7.407753	6.812568	5.899540	4.810701
        0.001431	7.455701	7.289040	6.723822	5.852292	4.806177

        SCT:
        >>> T = 300
        >>> beta_grid = gen_beta(T)
        >>> alpha_grid = gen_alpha(T, 26)
        >>> Al.get_Sab(alpha_grid, beta_grid, T, model="sct")["Al27"].data.iloc[:10, :5].round(6)
        beta      0.000000  0.012894  0.025788  0.038682  0.051576
        alpha
        0.001050  8.342190  8.092079  7.298835  6.121534  4.773978
        0.001087  8.200211  7.964121  7.209904  6.084151  4.785744
        0.001125  8.060646  7.837859  7.120876  6.044744  4.794361
        0.001164  7.923454  7.713288  7.031821  6.003428  4.799921
        0.001205  7.788595  7.590401  6.942804  5.960320  4.802517
        0.001247  7.656028  7.469191  6.853888  5.915532  4.802243
        0.001291  7.525715  7.349649  6.765132  5.869173  4.799196
        0.001336  7.397618  7.231765  6.676593  5.821349  4.793476
        0.001382  7.271698  7.115530  6.588322  5.772162  4.785181
        0.001431  7.147919  7.000933  6.500370  5.721713  4.774412

        Phonon Expansion:
        >>> T = 800
        >>> Al.get_Sab(alpha0_, beta0_, T, scale=True, model="phonon expansion", threshold=1.0e-14)["Al27"].data.iloc[:10, :5].round(6)
        beta      0.000000  0.009175  0.018350  0.027524  0.036699
        alpha
        0.001835  0.038004  0.038171  0.038333  0.038492  0.038645
        0.003670  0.074701  0.075013  0.075307  0.075590  0.075857
        0.005505  0.110103  0.110542  0.110941  0.111315  0.111663
        0.007340  0.144226  0.144776  0.145255  0.145693  0.146093
        0.009175  0.177088  0.177733  0.178272  0.178749  0.179174
        0.011010  0.208709  0.209435  0.210015  0.210509  0.210937
        0.012845  0.239108  0.239904  0.240509  0.241002  0.241412
        0.014680  0.268310  0.269164  0.269779  0.270255  0.270631
        0.016515  0.296336  0.297239  0.297853  0.298297  0.298625
        0.018350  0.323212  0.324156  0.324758  0.325158  0.325425
        """
        if isinstance(args[0], dict):
            Sab = self._get_Sab_multiple(*args, model=model, **kwargs)
        else:
            Sab = self._get_Sab_single(*args, model=model, **kwargs)
        return Sab


def numba_hkl_data(d_min, hkl_max, rec_vecs, Bfac, pos, csl,
                   preferred_orientation, precision) -> pd.DataFrame:
    """
    Obtain hkl data for the solid in a certain temperature and for a neutron
    certain energy.
    2 atoms test in test folder.

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
    >>> unit_pos_Al27 = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.25, 0.25,0.75, 0.25])
    >>> Al = Target_mat(preferred_orientation_Al27, unit_pos_Al27, dir_vec_length_Al27, dir_vec_angles_Al27, A_Al27, Z_Al27, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, rho_in_energy_Al27, interv_in_energy_Al27)
    >>> UO2 = Target_mat(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atom_mass, b_coh, b_incoh, rho_in_energy, interv_in_energy)

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
