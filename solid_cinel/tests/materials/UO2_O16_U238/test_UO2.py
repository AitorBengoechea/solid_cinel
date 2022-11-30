# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:00:47 2022

@author: AB272525
"""

import numpy as np
import pandas as pd
from solid_cinel import Target_mat
import pytest
import os
import sys

rho_in_energy_O16_str = '''
0.000000E+00 6.923874E-03 2.497670E-02 5.488348E-02
9.504920E-02 1.479389E-01 2.139513E-01 2.889902E-01
3.722217E-01 4.694096E-01 5.797566E-01 7.142103E-01
8.648680E-01 1.037820E+00 1.270585E+00 1.816629E+00
1.726038E+00 1.064790E+00 8.615329E-01 8.017556E-01
7.808027E-01 7.815640E-01 7.991818E-01 8.341637E-01
8.819783E-01 9.379855E-01 1.239156E+00 2.544143E+00
5.493169E+00 7.542890E+00 6.200347E+00 3.899265E+00
3.176137E+00 3.982750E+00 5.293972E+00 5.997778E+00
6.310875E+00 6.750886E+00 7.348876E+00 9.615086E+00
1.342006E+01 1.785015E+01 2.514254E+01 3.308876E+01
3.394329E+01 3.018225E+01 2.677197E+01 2.401069E+01
2.264379E+01 2.170413E+01 2.108204E+01 2.061835E+01
2.034441E+01 2.020180E+01 2.021090E+01 2.032422E+01
2.048390E+01 2.089081E+01 2.149127E+01 2.224180E+01
2.335361E+01 2.456841E+01 2.400029E+01 2.378412E+01
2.057435E+01 1.340375E+01 1.410426E+01 1.986035E+01
2.335756E+01 2.336811E+01 2.139441E+01 2.074719E+01
2.063757E+01 2.095150E+01 2.158940E+01 2.254504E+01
2.422993E+01 2.691331E+01 3.179124E+01 4.563659E+01
5.752641E+01 5.609342E+01 4.689648E+01 2.438494E+01
4.135546E+00 0.000000E+00 0.000000E+00 0.000000E+00
0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
0.000000E+00 5.255256E-01 2.237281E+00 4.016499E+00
5.186199E+00 6.762413E+00 8.981823E+00 1.161543E+01
1.599614E+01 3.174364E+01 3.582525E+01 2.322394E+01
1.668530E+01 1.262719E+01 1.004853E+01 8.184454E+00
6.946893E+00 6.009886E+00 5.284656E+00 4.766380E+00
4.441574E+00 4.272284E+00 3.664723E+00 2.362237E+00
8.264423E-01 1.493527E-02 0.000000E+00'''
rho_in_energy_O16 = np.fromstring(rho_in_energy_O16_str, dtype=np.float64,
                                  sep=' ')
interv_in_energy_O16 = interv_in_energy_U238 = 6.956193E-04
rho_in_energy_U238_str = '''
0.000000E+00 1.041128E-01 3.759952E-01 8.354039E-01
1.469796E+00 2.335578E+00 3.467660E+00 4.841392E+00
6.492841E+00 8.608376E+00 1.131303E+01 1.504441E+01
2.006807E+01 2.750471E+01 4.171597E+01 1.585670E+02
1.978483E+02 1.144621E+02 7.555927E+01 4.831100E+01
4.389081E+01 4.246484E+01 4.103699E+01 3.986249E+01
3.827959E+01 3.592088E+01 3.272170E+01 3.914602E+01
8.144694E+01 9.693959E+01 5.503795E+01 2.619253E+01
1.763331E+01 1.475875E+01 1.522465E+01 1.213117E+01
6.175029E+00 2.483519E+00 1.445581E+00 1.423177E+00
1.502350E+00 1.718768E+00 2.211346E+00 3.061686E+00
3.550530E+00 3.349917E+00 2.768379E+00 2.177488E+00
1.856123E+00 1.622775E+00 1.445254E+00 1.300794E+00
1.180078E+00 1.075748E+00 9.928057E-01 9.238564E-01
8.577708E-01 8.073819E-01 7.634820E-01 7.172257E-01
6.728183E-01 6.251482E-01 5.496737E-01 4.992486E-01
3.945195E-01 2.206960E-01 1.452214E-01 1.246671E-01
9.863893E-02 7.855588E-02 6.536053E-02 6.568678E-02
7.308199E-02 8.388478E-02 1.026265E-01 1.245221E-01
1.487740E-01 1.757085E-01 2.055793E-01 2.473042E-01
3.128097E-01 3.455081E-01 3.048708E-01 1.621507E-01
2.653572E-02 0.000000E+00 0.000000E+00 0.000000E+00
0.000000E+00 0.000000E+00 0.000000E+00 0.000000E+00
0.000000E+00 7.105193E-03 5.274518E-02 1.324974E-01
2.310275E-01 4.042710E-01 6.421137E-01 8.073457E-01
9.162074E-01 1.077923E+00 1.142595E+00 1.092532E+00
1.060668E+00 1.000020E+00 8.769838E-01 7.610532E-01
6.898200E-01 6.324347E-01 5.857072E-01 5.563076E-01
5.468099E-01 5.515587E-01 4.871045E-01 3.198787E-01
1.132118E-01 2.066306E-03 0.000000E+00
'''
rho_in_energy_U238 = np.fromstring(rho_in_energy_U238_str, dtype=np.float64,
                                   sep=' ')
rho_in_energy = [rho_in_energy_O16, rho_in_energy_U238]
interv_in_energy = [interv_in_energy_O16, interv_in_energy_U238]
preferred_orientation = np.array([0, 0, 1])
unit_pos_U_str = '''
0.500000  0.000000  0.000000
0.500000  0.500000  0.500000
0.000000  0.000000  0.500000
0.000000  0.500000  0.000000'''
unit_pos_U = np.fromstring(unit_pos_U_str, dtype=np.float64, sep=' ')\
               .reshape(-1, 3)
unit_pos_O_str = '''
0.250000  0.250000  0.250000
0.750000  0.250000  0.250000
0.250000  0.750000  0.750000
0.750000  0.750000  0.750000
0.750000  0.250000  0.750000
0.250000  0.250000  0.750000
0.750000  0.750000  0.250000
0.250000  0.750000  0.250000'''
unit_pos_O = np.fromstring(unit_pos_O_str, dtype=np.float64, sep=' ')\
               .reshape(-1, 3)
unit_pos = {"O16": unit_pos_O, "U238": unit_pos_U}
a = 5.54781
dir_vec_length = [a, a, a]
dir_vec_angles = [90, 90, 90]
energy_sup = 5.  # eV
energy_cut = 6.85e-1
A = [16, 238]
Z = [8, 92]
atom_mass = [15.99491399021626, 238.05077040419212]
b_coh = [5.878374042670532, 8.62912188811068]
b_incoh = [0.0, 0.19947114020071632]
UO2 = Target_mat(preferred_orientation, unit_pos,
                 dir_vec_length, dir_vec_angles,
                 A, Z, atom_mass, b_coh, b_incoh,
                 rho_in_energy, interv_in_energy)


@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_UO2_BraggEddges(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_UO2.py", ""))
    file = os.path.abspath(f"Multiplicity_UO2_{T}K.dat")
    test_data = pd.DataFrame(np.loadtxt(file),
                             columns=["h", "k", "l", "d", "theta", "Orientation angle", "PDDF", "Fsq", "Multiplicity", "E", "Xs"])\
                  .set_index(["h", "k", "l"]).loc[:, "Multiplicity"].sum()
    data = UO2.get_BraggEdges(T, energy_cut).loc[:, "Multiplicity"].sum()
    assert int(test_data) == int(data)
    os.chdir(wd)


@pytest.mark.parametrize("T", [296, 400, 500, 600, 700, 800, 1000, 1200])
def test_UO2_coherent_Xs(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_UO2.py", ""))
    file = f"O16_UO2_{T}K_coh_XS"
    test_dat = pd.DataFrame(np.loadtxt(file),
                            columns=["E", "Xs"])\
                 .set_index(["E"])
    test_data = pd.concat([test_dat] * len(UO2.atoms), axis=1)
    test_data.columns = pd.MultiIndex.from_product(
        [UO2.atoms.apply(lambda x: x.zam).values, [2]],
        names=["ZAM", "MT"])
    data = UO2.get_coherent_Xs(energy_cut, energy_sup, T)
    comp = pd.DataFrame(data.values / test_data.values - 1,
                        index=data.index).fillna(0)
    assert (comp < 5.0e-4).all().all()
    os.chdir(wd)
