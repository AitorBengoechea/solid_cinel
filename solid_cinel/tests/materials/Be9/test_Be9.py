# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:09:21 2022

@author: AB272525
"""
import pandas as pd
from solid_cinel.core.material.target_material import TargetMat
from solid_cinel.core.scattering_function.alpha import Alpha
from solid_cinel.core.scattering_function.beta import Beta
from scipy.integrate import trapezoid
import pytest
import os
# Material information variables:
from solid_cinel.data.materials.Be9 import *

# Examples variables:
from examples import *

# Target material:
Be = TargetMat(preferred_orientation, unit_pos,
                dir_vec_length, dir_vec_angles,
                A, Z, atomic_mass, b_coh, b_incoh,
                rho_in_energy, interv_in_energy)


@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_Be9_BraggEddges(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_Be9.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"Multiplicity_Be_{T}K.dat"),
                            key="test").loc[:, "Multiplicity"].sum()
    data = Be.get_BraggEdges(T, energy_cut).loc[:, "Multiplicity"].sum()
    assert int(test_data) == int(data)
    os.chdir(wd)

@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_Be9_coherent_Xs(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_Be9.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"Be9_Be_{T}K_coh_XS"), key="test")
    test_data.columns = pd.MultiIndex.from_product(
        [Be.atoms.apply(lambda x: x.zam).values, [2]],
        names=["ZAM", "MT"])
    data = Be.get_coherent_Xs(energy_cut, energy_sup, T)
    data = data.iloc[:len(data)-1]
    comp = pd.DataFrame(data.values / test_data.values - 1,
                        index=data.index).fillna(0)
    assert (comp < 1e-4).all().all()
    os.chdir(wd)


@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_Be9_Sab(T):
    wd = os.getcwd()
    beta_grid = Beta(beta0_).scale(T).data
    alpha_grid = Alpha(alpha0_).scale(T).data
    os.chdir(__file__.replace("test_Be9.py", ""))
    test_data = pd.read_hdf(os.path.abspath(f"Be9_Be_{T}K_SSab"), key="test").T
    test_data *= np.exp(beta_grid/2)
    test_data.index = pd.Index(alpha_grid, name="alpha")
    test_data.columns = pd.Index(beta_grid, name="beta")
    threshold = 0.0 if T < 300 else 1.0e-14
    data = Be.get_Sab(alpha_grid, beta_grid, T, model="phonon expansion",
                      threshold=threshold)["Be9"].data
    Sab_normalize = []
    for ia in range(len(test_data.index)):
        Sab_normalize_original = trapezoid(beta_grid * data.iloc[ia] * (1 + np.exp(-beta_grid)), beta_grid)
        Sab_normalize_Aitor = trapezoid(beta_grid * test_data.iloc[ia] * (1 + np.exp(-beta_grid)), beta_grid)
        Sab_normalize.append([Sab_normalize_original/Sab_normalize_Aitor - 1,
                              Sab_normalize_Aitor/Sab_normalize_original - 1])
    comp = pd.DataFrame(Sab_normalize, index=test_data.index)
    assert (abs(comp) < 1.0e-3).all().all()
    os.chdir(wd)
