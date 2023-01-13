# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:52:18 2022

@author: AB272525
"""
import numpy as np
import pandas as pd
from solid_cinel import Target_mat
from solid_cinel.core.s import Alpha, Beta
from scipy.integrate import trapezoid
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
def test_Al27_coherent_Xs(T):
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


@pytest.mark.parametrize("T", [20, 80, 293.6, 400, 600, 800])
def test_Al27_Sab(T):
    wd = os.getcwd()
    beta_grid = Beta(beta0_).scale(T).data
    alpha_grid = Alpha(alpha0_).scale(T).data
    os.chdir(__file__.replace("test_Al27.py", ""))
    file = os.path.abspath(f"Al27_Al_{T}K_SSab")
    test_data = pd.DataFrame(np.loadtxt(file).T * np.exp(beta_grid/2),
                             columns=beta_grid,
                             index=alpha_grid)
    test_data.index.name, test_data.columns.name = "alpha", "beta"
    threshold = 0.0 if T < 200 else 1.0e-14
    data = Al.get_Sab(alpha_grid, beta_grid, T, model="phonon expansion",
                      threshold=threshold)["Al27"].data
    Sab_normalize = []
    for ia in range(len(test_data.index)):
        Sab_normalize_original = trapezoid(beta_grid * data.iloc[ia] * (1 + np.exp(-beta_grid)), beta_grid)
        Sab_normalize_Aitor = trapezoid(beta_grid * test_data.iloc[ia] * (1 + np.exp(-beta_grid)), beta_grid)
        Sab_normalize.append([Sab_normalize_original/Sab_normalize_Aitor - 1,
                              Sab_normalize_Aitor/Sab_normalize_original - 1])
    comp = pd.DataFrame(Sab_normalize, index=test_data.index)
    assert (abs(comp) < 1.0e-3).all().all()
    os.chdir(wd)
