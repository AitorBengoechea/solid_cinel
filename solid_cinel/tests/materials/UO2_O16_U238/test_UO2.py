# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:00:47 2022

@author: AB272525
"""
import pandas as pd
from solid_cinel.core.material.target_material import Target_mat
from solid_cinel.core.scattering_function.alpha import Alpha
from solid_cinel.core.scattering_function.beta import Beta
from scipy.integrate import trapezoid
import pytest
import os
# Material information variables:
from solid_cinel.data.materials.UO2 import *

# Examples variables:
from examples import *

# Target Material
UO2 = Target_mat(preferred_orientation, unit_pos,
                 dir_vec_length, dir_vec_angles,
                 A, Z, atom_mass, b_coh, b_incoh,
                 rho_in_energy, interv_in_energy)


@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_UO2_BraggEddges(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_UO2.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"Multiplicity_UO2_{T}K.dat"),
                            key="test").loc[:, "Multiplicity"].sum()
    data = UO2.get_BraggEdges(T, energy_cut).loc[:, "Multiplicity"].sum()
    assert int(test_data) == int(data)
    os.chdir(wd)


@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_UO2_coherent_Xs(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_UO2.py", ""))
    test_dat = pd.read_hdf(os.path.abspath(f"O16_UO2_{T}K_coh_XS"), key="test")
    test_data = pd.concat([test_dat] * len(UO2.atoms), axis=1)
    test_data.columns = pd.MultiIndex.from_product(
        [UO2.atoms.apply(lambda x: x.zam).values, [2]],
        names=["ZAM", "MT"])
    data = UO2.get_coherent_Xs(energy_cut, energy_sup, T)
    comp = pd.DataFrame(data.values / test_data.values - 1,
                        index=data.index).fillna(0)
    assert (comp < 5.0e-4).all().all()
    os.chdir(wd)


@pytest.mark.parametrize("T", [296, 400, 800, 1200])
def test_UO2_Sab(T):
    wd = os.getcwd()
    beta_grid = {"O16": Beta(beta0_O16).scale(T).data,
                 "U238": Beta(beta0_U238).scale(T).data}
    alpha_grid = {"O16": Alpha(alpha0_O16).scale(T).data,
                  "U238": Alpha(alpha0_U238).scale(T).data}
    nphonon = {"O16": 200 if T != 1200 else 900,
               "U238": 300 if T != 1200 else 900}
    os.chdir(__file__.replace("test_UO2.py", ""))
    Sab = UO2.get_Sab(alpha_grid, beta_grid, T, nphonon=nphonon)

    # Test O16:
    test_data_O16 = Sab["O16"].data
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
    test_data_U238 = Sab["U238"].data
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
