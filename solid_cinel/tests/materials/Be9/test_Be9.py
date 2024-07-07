# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:09:21 2022

@author: AB272525
"""
import pandas as pd
from solid_cinel.core import Solid, Alpha, Beta, Pdos, Sab
from scipy.integrate import trapezoid
import pytest
import os

# Example variables:
from examples import *

# Target material:
file_dir = os.path.dirname(os.path.abspath(__file__))
compositon_file = os.path.join(file_dir,
                                    '../../../data/materials/Be9/Be9Composition')
structure_file = os.path.join(file_dir,
                                   '../../../data/materials/Be9/Be9Structure')
atomPos_file = os.path.join(file_dir,
                                 '../../../data/materials/Be9/Be9AtomPos')
Be = Solid.from_files(compositon_file, structure_file, atomPos_file)
pdosBe9 = Pdos.from_dE(rho_in_energy, interv_in_energy)
Be.set_pdos(pdosBe9)


@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_Be9_BraggEddges(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_Be9.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"Multiplicity_Be_{T}K.dat"),
                            key="test").loc[:, "Multiplicity"].sum()
    data = Be.get_BraggEdges(energy_cut, T).loc[:, "Multiplicity"].sum()
    assert int(test_data) == int(data)
    os.chdir(wd)

@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_Be9_coherent_Xs(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_Be9.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"Be9_Be_{T}K_coh_XS"), key="test")
    data = Be.get_XsCoh(energy_cut,  T)
    comp = test_data / data - 1
    comp_inv = data / test_data - 1  # Avoid one 0 and the other a number
    assert (comp.fillna(0) < 1.0e-4).all().all()
    assert (comp_inv.fillna(0) < 1.0e-4).all().all()
    os.chdir(wd)


@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_Be9_Sab(T):
    wd = os.getcwd()
    # Get the grid
    beta_grid = Beta(beta0_).scale(T).data
    alpha_grid = Alpha(alpha0_).scale(T).data
    # Get the test data
    os.chdir(__file__.replace("test_Be9.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"Be9_Be_{T}K_SSab"), key="test").T
    test_data *= np.exp(beta_grid/2)
    test_data.index = pd.Index(alpha_grid, name="alpha")
    test_data.columns = pd.Index(beta_grid, name="beta")
    # Calculate the data
    threshold = 0.0 if T < 300 else 1.0e-14
    data = Sab.from_pdos(alpha_grid, beta_grid, T, pdosBe9, threshold=threshold).data
    Sab_normalize = []
    for ia in range(len(test_data.index)):
        Sab_normalize_original = trapezoid(beta_grid * data.iloc[ia] * (1 + np.exp(-beta_grid)), beta_grid)
        Sab_normalize_Aitor = trapezoid(beta_grid * test_data.iloc[ia] * (1 + np.exp(-beta_grid)), beta_grid)
        Sab_normalize.append([Sab_normalize_original/Sab_normalize_Aitor - 1,
                              Sab_normalize_Aitor/Sab_normalize_original - 1])
    comp = pd.DataFrame(Sab_normalize, index=test_data.index)
    assert (abs(comp) < 1.0e-3).all().all()
    os.chdir(wd)
