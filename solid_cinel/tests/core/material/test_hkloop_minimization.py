# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:13:08 2022

@author: AB272525
"""

from solid_cinel.core.material.crystal_symmetry import dir_vector_operator
from solid_cinel.core.material.solid import Solid, hkl_minimize
import numba as nb
import numpy as np
import pandas as pd
import pytest


@nb.jit(nopython=True, nogil=True)
def hklloop_max(rec_vecs, d_min):
    hklM = []
    hklrange = np.arange(-100, 100 + 1)
    hklrange = hklrange[hklrange != 0]  # (0,0,0) excluded

    for h in hklrange[::-1]:  # to get positive hkl order
        for k in hklrange[::-1]:
            for l in hklrange[::-1]:
                vec_tau_hkl = h * rec_vecs[0] + k * rec_vecs[1] + l * rec_vecs[2]
                vec_tau_hkl_norm = np.linalg.norm(vec_tau_hkl)
                d_hkl = 2 * np.pi / vec_tau_hkl_norm  # d_hkl = 2pi / tau_hkl

                if d_hkl < d_min:  # d < d_min is excluded
                    continue
                else:
                    hklM.append([h, k, l])

    return np.array(hklM)


@pytest.mark.parametrize("d_min", [0.2360746677309732])
def test_hklloop_aprox(d_min):
    preferred_orientation = np.array([ 0, 1, 1 ])
    dir_vec_length = np.random.rand(3) * 4
    dir_vec_angles = np.random.rand(3) * 180 
    unit_pos = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75, 0.25, 0.25,0.75, 0.25])
    A = 27
    Z = 13
    atomic_mass_Al27 = 26.98153433356103
    b_coh_Al27  = 3.449
    b_incoh_Al27 = 0.256
    Al = Solid(dir_vec_length, dir_vec_angles, preferred_orientation, unit_pos, A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)
    rec_vecs = Al.reciproc_vec
    if rec_vecs.isnull().values.any():
        assert True == True
    else:
        rec_vecs = rec_vecs.values
        full_hkl = hklloop_max(rec_vecs, d_min).max(axis=0)
        hkl_minimization = hkl_minimize(rec_vecs, d_min)
        assert (hkl_minimization >= full_hkl).all() == True
