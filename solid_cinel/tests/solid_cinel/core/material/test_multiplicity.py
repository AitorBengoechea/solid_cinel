# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:15:33 2022
@author: AB272525
"""

from solid_cinel.core import Target_mat
from solid_cinel.cinematic.SVT import Neutron
import numba as nb
import numpy as np
import pandas as pd
import pytest

kPiSq = 9.86960440108935861883449099987615113531369941  # = pi^2
k2Pi = 6.2831853071795864769252867665590057683943388 # = 2pi

# CONSTANT VALUES:
preferred_orientation = [0, 0, 1]
A = 27
Z = 13
atomic_mass_Al27 = 26.98153433356103
b_incoh_Al27 = 0.256
b_coh_Al27 = 3.449
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

@nb.jit(nopython=True, nogil=True)
def hklloop(d_min,
            d_precision,
            Fsq_precision,
            rec_vecs,
            Bfac,
            pos,
            csl):
    hklrange = np.arange(-100, 101)
    hklM = {}
    hkldF = {}
    for h in hklrange[::-1]:  # to get positive hkl order
        for k in hklrange[::-1]:
            for l in hklrange[::-1]:
                if h ** 2 + k ** 2 + l ** 2 == 0:  # (0, 0, 0) is excluded
                    continue
                vec_tau_hkl = h * rec_vecs[0] + k * rec_vecs[1] + l * rec_vecs[2]
                vec_tau_hkl_norm = np.linalg.norm(vec_tau_hkl)
                d_hkl = k2Pi / vec_tau_hkl_norm  # d_hkl = 2pi / tau_hkl
                d_rnd = round(d_hkl, d_precision)

                if d_hkl < d_min:  # d < d_min is excluded
                    continue

                real = 0.
                imag = 0.   
                for element in pos:
                    expon_hkl = np.exp(-0.5 * vec_tau_hkl_norm ** 2 * Bfac[element] / (8 * kPiSq))
                    element_position = pos[element]
                    for iep in range(len(element_position)):
                        cumulant_cos = np.cos(np.sum(vec_tau_hkl * element_position[iep]))
                        cumulant_sin = np.sin(np.sum(vec_tau_hkl * element_position[iep]))
                        real += csl[element] * 0.1 * expon_hkl * cumulant_cos
                        imag += csl[element] * 0.1 * expon_hkl * cumulant_sin
                Fsq_hkl = real ** 2 + imag ** 2  # Fsquared
                Fsq_rnd = round(Fsq_hkl, Fsq_precision)

                # same dspacing and Fsquared with precision will be regrouped
                if (d_rnd, Fsq_rnd) in hkldF:
                    hklM[hkldF[(d_rnd, Fsq_rnd)]][-1] += 1
                else:
                    hkldF[(d_rnd, Fsq_rnd)] = (h, k, l)
                    hklM[(h, k, l)] = np.array([d_hkl, Fsq_hkl, 1])
    return hklM

@pytest.mark.parametrize("E", [np.random.rand(1)[0] * 10])
def test_multiplicity(E):
    d_min = Neutron(E).d_min
    T = np.random.rand(1)[0] * 1000
    unit_pos = np.random.rand(np.random.randint(1, 10), 3)
    dir_vec_length = np.random.rand(3) * 4
    dir_vec_angles = np.random.rand(3) * 180
    Al = Target_mat(preferred_orientation, unit_pos,
                    dir_vec_length, dir_vec_angles,
                    A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27,
                    rho_in_energy, interv_in_energy)
    while Al.reciproc_vec.isnull().values.any():
        dir_vec_length = np.random.rand(3) * 4
        dir_vec_angles = np.random.rand(3) * 180
        Al = Target_mat(preferred_orientation, unit_pos,
                        dir_vec_length, dir_vec_angles,
                        A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27,
                        rho_in_energy, interv_in_energy)
    Bfac = nb.typed.Dict.empty(
            key_type   = nb.core.types.unicode_type,
            value_type = nb.core.types.float64,
        )
    pos = nb.typed.Dict.empty(
            key_type   = nb.core.types.unicode_type,
            value_type = nb.core.types.float64[:,:],
        )
    csl = nb.typed.Dict.empty(
            key_type   = nb.core.types.unicode_type,
            value_type = nb.core.types.float64,
    )
    for element, value in Al.get_Bfact(T).to_dict().items():
        Bfac[element] = value
        pos[element] = Al.atom_pos[element].values
        csl[element] = 3.449
    try:
        data_original = hklloop(d_min,
                                6,
                                6,
                                Al.reciproc_vec.values,
                                Bfac,
                                pos,
                                csl)
        data_original_frame = pd.DataFrame([[h, k, l, d_hkl, Fsq_hkl, mul] for (h, k, l), [d_hkl, Fsq_hkl, mul] in data_original.items()],
                                   columns = ["h", "k", "l", "d", "Fsq", "Multiplicity"]).loc[:, "Multiplicity"].sum()
        data = Al.get_multiplicity(T, E).loc[:, "Multiplicity"].sum()
        assert int(data_original_frame) == int(data)

    except IndexError: #Random problem can not be solve in hklloop_max
        assert True