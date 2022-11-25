# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:52:18 2022

@author: AB272525
"""
import numpy as np
import pandas as pd
from solid_cinel import Target_mat
import pytest
import os
import sys



@pytest.mark.parametrize("T", [20, 80, 293.6, 400, 600, 800])
def test_Fe56_coherent_Xs(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_Fe56.py", ""))
    preferred_orientation = np.array([ 0, 0, 1 ])
    a = 2.476832
    dir_vec_length = [a, a, a]
    dir_vec_angles = [109.47122063449069, 109.47122063449069, 109.47122063449069]
    unit_pos = np.array([0., 0., 0.])
    A = 56
    Z = 26
    atomic_mass_Al27 = 55.934504245209524
    b_coh_Al27 = 10.131877333163509
    b_incoh_Al27 = 0.
    # Examples variables:
    rho_in_energy_str = '''
    0. .000111 .000443 .000996 .00177 .00277 .00398 .00542 .00708
    .00896 .0111 .0134 .0159 .0187 .0217 .0249 .0283 .0317 .0367
    .0400 .0450 .0500 .0550 .0617 .0667 .0733 .0817 .0867 .0950
    .105 .115 .127 .138 .152 .168 .183 .207 .235 .272 .308 .323
    .338 .352 .370 .392 .413 .438 .473 .445 .395 .385 .383 .398
    .427 .505 .495 .467 .442 .420 .393 .360 .333 .315 .300 .295
    .293 .297 .310 .335 .388 .553 .672 .970 .553 .400 .300 .222
    .145 0.
    '''
    rho_in_energy = np.fromstring(rho_in_energy_str, dtype = np.float64, sep = ' ')
    interv_in_energy = 0.0005
    Fe = Target_mat(preferred_orientation, unit_pos,
                    dir_vec_length, dir_vec_angles,
                    A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27,
                    rho_in_energy, interv_in_energy)
    E = 2.301
    file = os.path.abspath(f"Multiplicity_Fe_{T}K.dat")
    test_data = pd.DataFrame(np.loadtxt(file),
                             columns=["h", "k", "l", "d", "theta", "Orientation angle", "PDDF", "Fsq", "Multiplicity", "E", "Xs"])\
        .set_index(["h", "k", "l"]).iloc[::, :5:-1]
    data = Fe.get_coherent_XS(T, E).iloc[::, :4:-1].drop(columns="theta")
    comp = test_data/data - 1
    comp_inv = data/test_data - 1  # Avoid one 0 and the other a number
    assert (comp.fillna(0) < 1.0e-4).all().all()
    assert (comp_inv.fillna(0) < 1.0e-4).all().all()
    os.chdir(wd)
