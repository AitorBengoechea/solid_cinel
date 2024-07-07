# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:00:47 2022

@author: AB272525
"""
import pandas as pd
from solid_cinel.core import Solid, Alpha, Beta, Pdos, Sab
from scipy.integrate import trapezoid
import pytest
import os

# Example variables:
from examples import *

# Target Material
file_dir = os.path.dirname(os.path.abspath(__file__))
compositon_file = os.path.join(file_dir,
                                    '../../../data/materials/UO2/UO2Composition')
structure_file = os.path.join(file_dir,
                                   '../../../data/materials/UO2/UO2Structure')
atomPos_file = os.path.join(file_dir,
                                 '../../../data/materials/UO2/UO2AtomPos')
UO2 = Solid.from_files(compositon_file, structure_file, atomPos_file)
pdosUO2 = {"O16": Pdos.from_dE(rho_in_energy[0], interv_in_energy[0]),
           "U238": Pdos.from_dE(rho_in_energy[1], interv_in_energy[1])}
UO2.set_pdos(pdosUO2)

@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_UO2_BraggEddges(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_UO2.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"Multiplicity_UO2_{T}K.dat"),
                            key="test").loc[:, "Multiplicity"].sum()
    data = UO2.get_BraggEdges(energy_cut, T).loc[:, "Multiplicity"].sum()
    assert int(test_data) == int(data)
    os.chdir(wd)


@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_UO2_coherent_Xs(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_UO2.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"O16_UO2_{T}K_coh_XS"), key="test")
    data = UO2.get_XsCoh(energy_cut, T)
    comp = test_data / data - 1
    comp_inv = data / test_data - 1  # Avoid one 0 and the other a number
    assert (comp.fillna(0) < 5.0e-4).all().all()
    assert (comp_inv.fillna(0) < 5.0e-4).all().all()
    os.chdir(wd)


@pytest.mark.parametrize("T", [296, 400, 800, 1200])
def test_UO2_Sab(T):
    wd = os.getcwd()
    beta_grid = {"O16": Beta(beta0_O16).scale(T).data,
                 "U238": Beta(beta0_U238).scale(T).data}
    alpha_grid = {"O16": Alpha(alpha0_O16).scale(T).data,
                  "U238": Alpha(alpha0_U238).scale(T).data}
    os.chdir(__file__.replace("test_UO2.py", ""))

    # Test O16:
    test_data_O16 = Sab.from_pdos(alpha_grid["O16"], beta_grid["O16"], T, pdosUO2["O16"]).data
    data_O16 = pd.read_hdf(os.path.abspath(f"O16_UO2_{T}K_SSab"), key="test").T
    data_O16 *= np.exp(beta_grid["O16"]/2)
    data_O16.index = pd.Index(alpha_grid["O16"], name="alpha")
    data_O16.columns = pd.Index(beta_grid["O16"], name="beta")

    Sab_normalize = []
    for ia in range(len(test_data_O16.index)):
        Sab_normalize_original = trapezoid(beta_grid["O16"] * data_O16.iloc[ia] * (1 + np.exp(-beta_grid["O16"])), beta_grid["O16"])
        Sab_normalize_Aitor = trapezoid(beta_grid["O16"] * test_data_O16.iloc[ia] * (1 + np.exp(-beta_grid["O16"])), beta_grid["O16"])
        Sab_normalize.append([Sab_normalize_original/Sab_normalize_Aitor - 1,
                              Sab_normalize_Aitor/Sab_normalize_original - 1])
    comp = pd.DataFrame(Sab_normalize, index=test_data_O16.index)
    assert (abs(comp) < 1.0e-3).all().all()

    # Test U238:
    test_data_U238 = Sab.from_pdos(alpha_grid["U238"], beta_grid["U238"], T, pdosUO2["U238"]).data
    data_U238 = pd.read_hdf(os.path.abspath(f"U238_UO2_{T}K_SSab"), key="test").T
    data_U238 *= np.exp(beta_grid["U238"]/2)
    data_U238.index = pd.Index(alpha_grid["U238"], name="alpha")
    data_U238.columns = pd.Index(beta_grid["U238"], name="beta")
    Sab_normalize = []
    for ia in range(len(test_data_U238.index)):
        Sab_normalize_original = trapezoid(beta_grid["U238"] * data_U238.iloc[ia] * (1 + np.exp(-beta_grid["U238"])), beta_grid["U238"])
        Sab_normalize_Aitor = trapezoid(beta_grid["U238"] * test_data_U238.iloc[ia] * (1 + np.exp(-beta_grid["U238"])), beta_grid["U238"])
        Sab_normalize.append([Sab_normalize_original/Sab_normalize_Aitor - 1,
                              Sab_normalize_Aitor/Sab_normalize_original - 1])
    comp = pd.DataFrame(Sab_normalize, index=test_data_U238.index)
    assert (abs(comp) < 1.0e-3).all().all()
    os.chdir(wd)
