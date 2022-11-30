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


preferred_orientation = np.array([0, 0, 1])
a = 2.856710674519725
dir_vec_length = [a, a, a]
dir_vec_angles = [60, 60, 60]
unit_pos = np.array([0., 0., 0.])
A = 27
Z = 13
atomic_mass_Al27 = 26.98153433356103
b_coh_Al27 = 3.449
b_incoh_Al27 = 0.256
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
Al = Target_mat(preferred_orientation, unit_pos,
                dir_vec_length, dir_vec_angles,
                A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27,
                rho_in_energy, interv_in_energy)
energy_cut = 2.301
energy_sup = 10.0


@pytest.mark.parametrize("T", [20, 80, 293.6, 400, 600, 800])
def test_Al27_BraggEddges(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_Al27.py", ""))
    file = os.path.abspath(f"Multiplicity_Al_{T}K.dat")
    test_data = pd.DataFrame(np.loadtxt(file),
                             columns=["h", "k", "l", "d", "theta", "Orientation angle", "PDDF", "Fsq", "Multiplicity", "E", "Xs"])\
        .set_index(["h", "k", "l"]).iloc[::, :5:-1]
    data = Al.get_BraggEdges(T, energy_cut)\
             .iloc[::, :4:-1]\
             .drop(columns="theta")
    comp = test_data/data - 1
    comp_inv = data/test_data - 1  # Avoid one 0 and the other a number
    assert (comp.fillna(0) < 1.0e-4).all().all()
    assert (comp_inv.fillna(0) < 1.0e-4).all().all()
    os.chdir(wd)


@pytest.mark.parametrize("T", [20, 80, 293.6, 400, 600, 800])
def test_Fe56_coherent_Xs(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_Al27.py", ""))
    file = os.path.abspath(f"Al27_Al_{T}K_coh_XS")
    test_data = pd.DataFrame(np.loadtxt(file),
                             columns=["E", "Xs"])\
        .set_index(["E"])
    test_data.columns = pd.MultiIndex.from_product(
        [Al.atoms.apply(lambda x: x.zam).values, [2]],
        names=["ZAM", "MT"])
    data = Al.get_coherent_Xs(energy_cut, energy_sup, T)
    comp = test_data/data - 1
    comp_inv = data/test_data - 1  # Avoid one 0 and the other a number
    assert (comp.fillna(0) < 1.0e-4).all().all()
    assert (comp_inv.fillna(0) < 1.0e-4).all().all()
    os.chdir(wd)
