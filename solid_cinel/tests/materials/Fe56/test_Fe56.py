# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:52:18 2022

@author: AB272525from solid_cinel.core.s import S
from scipy.integrate import trapezoid
"""
import pandas as pd
from solid_cinel import Solid, Alpha, Beta, Pdos, Sab
from scipy.integrate import trapezoid
import pytest
import os
# Material information variables:
from solid_cinel.data.materials.Fe56 import *


# Example variables:
from examples import *

# Target material
pdosFe56 = Pdos.from_dE(rho_in_energy, interv_in_energy)
Fe = Solid(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles,
               A, Z, atomic_mass, b_coh, b_incoh, pdosFe56)


@pytest.mark.parametrize("T", [20, 80, 293.6, 400, 600, 800])
def test_Fe56_BraggEddges(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_Fe56.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"Multiplicity_Fe_{T}K.dat"),
                            key="test")
    data = Fe.get_BraggEdges(energy_cut, T).iloc[::, :4:-1].drop(columns="theta")
    comp = test_data/data - 1
    comp_inv = data/test_data - 1  # Avoid one 0 and the other a number
    assert (comp.fillna(0) < 1.0e-4).all().all()
    assert (comp_inv.fillna(0) < 1.0e-4).all().all()
    os.chdir(wd)


@pytest.mark.parametrize("T", [20, 80, 293.6, 400, 600, 800])
def test_Fe56_coherent_Xs(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_Fe56.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"Fe56_Fe_{T}K_coh_XS"), key="test")
    data = Fe.get_XsCoh(energy_cut, T)
    comp = test_data/data - 1
    comp_inv = data/test_data - 1  # Avoid one 0 and the other a number
    assert (comp.fillna(0) < 1.0e-4).all().all()
    assert (comp_inv.fillna(0) < 1.0e-4).all().all()
    os.chdir(wd)


@pytest.mark.parametrize("T", [20, 80, 293.6, 400, 600, 800])
def test_Fe56_Sab(T):
    wd = os.getcwd()
    # Get the grid
    beta_grid = Beta(beta0_).scale(T).data
    alpha_grid = Alpha(alpha0_).scale(T).data
    # Get the test data
    os.chdir(__file__.replace("test_Fe56.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"Fe56_Fe_{T}K_SSab"), key="test").T
    test_data *= np.exp(beta_grid/2)
    test_data.index = pd.Index(alpha_grid, name="alpha")
    test_data.columns = pd.Index(beta_grid, name="beta")
    # Calculate the data
    threshold = 0.0 if T < 200 else 1.0e-14
    data = Sab.from_pdos(alpha_grid, beta_grid, T, pdosFe56, threshold=threshold).data
    Sab_normalize = []
    for ia in range(len(test_data.index)):
        Sab_normalize_original = trapezoid(beta_grid * data.iloc[ia] * (1 + np.exp(-beta_grid)), beta_grid)
        Sab_normalize_Aitor = trapezoid(beta_grid * test_data.iloc[ia] * (1 + np.exp(-beta_grid)), beta_grid)
        Sab_normalize.append([Sab_normalize_original/Sab_normalize_Aitor - 1,
                              Sab_normalize_Aitor/Sab_normalize_original - 1])
    comp = pd.DataFrame(Sab_normalize, index=test_data.index)
    assert (abs(comp) < 1.0e-3).all().all()
    os.chdir(wd)
