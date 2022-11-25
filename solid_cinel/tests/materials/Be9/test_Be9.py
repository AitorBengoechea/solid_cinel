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



@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_Be9_coherent_Xs(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_Be9.py", ""))
    preferred_orientation = np.array([ 0, 0, 1 ])
    a = 2.271566
    c = 3.545935
    dir_vec_length = [a, a, c]
    dir_vec_angles = [90, 90, 120]
    unit_pos = np.array([[0.3333, 0.6667, 0.25], [0.6667, 0.3333, 0.75]])
    A = 9
    Z = 4
    atomic_mass_Be9 = 9.012199117106308
    b_coh_Be9 = 7.780229639986676
    b_incoh_Be9 = 0.1196826841204298
    # Examples variables:
    rho_in_energy_str = '''
    0.0000E+00 7.2477E-04 3.7084E-03 8.0087E-03
    1.0642E-02 1.5897E-02 2.7372E-02 4.1843E-02
    5.0214E-02 6.5036E-02 8.3674E-02 9.9329E-02
    1.1977E-01 1.4296E-01 1.6484E-01 1.8945E-01
    2.1887E-01 2.3537E-01 2.6166E-01 3.0003E-01
    3.4054E-01 3.8728E-01 4.2481E-01 4.7598E-01
    5.1890E-01 5.7400E-01 6.2970E-01 6.5754E-01
    7.2042E-01 7.9118E-01 8.6756E-01 9.2948E-01
    1.0030E+00 1.1163E+00 1.2048E+00 1.2870E+00
    1.4139E+00 1.5249E+00 1.6221E+00 1.7638E+00
    1.8924E+00 2.0388E+00 2.2056E+00 2.3709E+00
    2.5558E+00 2.7595E+00 3.0108E+00 3.2603E+00
    3.5066E+00 3.7442E+00 4.0067E+00 4.3677E+00
    4.7164E+00 5.0820E+00 5.5881E+00 6.0898E+00
    6.5510E+00 7.0877E+00 7.5931E+00 8.0736E+00
    8.6232E+00 9.2283E+00 9.9334E+00 1.0613E+01
    1.1278E+01 1.1973E+01 1.2784E+01 1.3744E+01
    1.4739E+01 1.5918E+01 1.7654E+01 1.9834E+01
    2.1455E+01 2.2574E+01 2.3744E+01 2.4900E+01
    2.6227E+01 2.7931E+01 2.9747E+01 2.9884E+01
    2.7358E+01 2.4817E+01 2.3690E+01 2.3242E+01
    2.3624E+01 2.3473E+01 2.2368E+01 2.1447E+01
    2.0724E+01 2.1121E+01 2.4240E+01 2.7607E+01
    2.7643E+01 2.5431E+01 2.3755E+01 2.3377E+01
    2.3410E+01 2.3504E+01 2.3647E+01 2.3681E+01
    2.3805E+01 2.3714E+01 2.3385E+01 2.3050E+01
    2.2244E+01 2.1008E+01 1.9536E+01 1.8341E+01
    1.8075E+01 1.8606E+01 1.9599E+01 2.1037E+01
    2.3193E+01 2.4016E+01 2.3573E+01 2.5664E+01
    3.0187E+01 3.1256E+01 2.7257E+01 2.2765E+01
    1.4893E+01 6.8192E+00 3.8444E+00 2.4718E+00
    1.3358E+00 3.5968E-01 0.0000E+00
    '''
    rho_in_energy = np.fromstring(rho_in_energy_str, dtype = np.float64, sep = ' ')
    interv_in_energy = 0.00069552
    Be = Target_mat(preferred_orientation, unit_pos,
                    dir_vec_length, dir_vec_angles,
                    A, Z, atomic_mass_Be9, b_coh_Be9, b_incoh_Be9,
                    rho_in_energy, interv_in_energy)
    E = 0.33118
    file = os.path.abspath(f"Multiplicity_Be_{T}K.dat")
    test_data = pd.DataFrame(np.loadtxt(file),
                             columns=["h", "k", "l", "d", "theta", "Orientation angle", "PDDF", "Fsq", "Multiplicity", "E", "Xs"])\
        .set_index(["h", "k", "l"]).iloc[::, :5:-1]
    data = Be.get_coherent_XS(T, E).iloc[::, :4:-1].drop(columns="theta")
    comp = test_data/data - 1
    comp_inv = data/test_data - 1  # Avoid one 0 and the other a number
    assert (comp.fillna(0) < 1.0e-4).all().all()
    assert (comp_inv.fillna(0) < 1.0e-4).all().all()
    os.chdir(wd)
