import pytest
import os
import numpy as np
import pandas as pd
from solid_cinel.core.material.pdos import Pdos
from solid_cinel.core.xs.ddxs import DDxs
from solid_cinel.core.xs.xs import Xs


# Global variables for the tests:
M = 238.0
Ein = 6.67
theta = np.arange(0, 181, 1)[1::]
T = 1474.2
interv_in_energy_U238 = 6.956193E-04
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
pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)


# Coercelle model + sigma1 for xs matrix:
@pytest.mark.parametrize("model", ["sigma1", "fgm", "sct"])
def test_coercelle(model):
    wd = os.getcwd()
    # Get test data:
    os.chdir(__file__.replace("test_coercelle.py", ""))
    ddxs_test = pd.read_hdf("ddxs_arno_{}".format(model), 'test')

    # Get 0K data:
    os.chdir("../../../../data/xs/U238/")
    xs0K = pd.read_csv("u238.0.2", sep='\s+', header=None, index_col=0, usecols=[0, 1], engine="python").iloc[::, 0]
    xs0K.index.name = "E"
    xs0K = xs0K.reset_index().drop_duplicates(subset='E', keep='first').set_index('E').iloc[:, 0]
    os.chdir(wd)

    # Get the Xs object:
    # Get the Xs object:
    M = 238.05077040419212
    xs = Xs(M, 0, xs0K)

    # Get doppler broadening ddxs:
    Eout = ddxs_test.columns.values
    if model == "sigma1":
        ddxs = DDxs.from_4PCF(xs, Ein, T, Eout, theta)
    elif model == "fgm":
        ddxs = DDxs.from_4PCF(xs, Ein, T, Eout, theta, model=model)
    elif model == "sct":
        ddxs = DDxs.from_4PCF(xs, Ein, T, Eout, theta, pdos, model=model)

    # round to 6 decimals:
    ddxs_round = ddxs.data.round(6).values
    ddxs_test_round = ddxs_test.round(6).values

    # Compare the results:
    diff = (1 - ddxs_round / ddxs_test_round) * 100
    diff = np.nan_to_num(diff)
    assert abs(diff).max() < 1e-5