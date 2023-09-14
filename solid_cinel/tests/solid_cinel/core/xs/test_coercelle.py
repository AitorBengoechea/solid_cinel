import pytest
import os
import numpy as np
import pandas as pd
from solid_cinel.core.xs.doppler_broad import get_DB
from solid_cinel.core.generic import integrate


# Global variables for the tests:
M = 238.0
Ein = 6.67
theta = np.arange(0, 181, 1)[1::]
T = 1474.2


# Coercelle model + sigma1 for xs matrix:
@pytest.mark.parametrize("model", ["sigma1", "fgm"])
def test_coercelle(model):
    wd = os.getcwd()
    # Get test data:
    os.chdir(__file__.replace("test_coercelle.py", ""))
    ddxs_test = pd.read_hdf("ddxs_arno_{}".format(model), 'test')
    os.chdir(wd)

    # Get 0K data:
    os.chdir("../../../../data/xs/U238/")
    xs_0K = pd.read_csv("u238.0.2", sep="    ", header=None, engine="python")\
              .set_index(0).drop([2], axis=1).iloc[::, 0]
    xs_0K = xs_0K[~xs_0K.index.duplicated()]
    os.chdir(wd)

    # Get doppler broadening ddxs:
    Eout = ddxs_test.columns.values
    if model == "sigma1":
        ddxs = get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm="courcelle")
    else:
        ddxs = get_DB(xs_0K, Ein, M, T, Eout, theta, algorithm="courcelle",
                      model=model)

    # Check integral value:
    test_integral = integrate(ddxs_test.apply(integrate, axis=1))
    ddxs_integral = integrate(ddxs.apply(integrate, axis=1))
    assert abs(1 - ddxs_integral / test_integral) <= 0.03

    # Check differential value:
    assert abs(1 - ddxs / ddxs_test).max().max() <= 0.03

    # Check angular distribution:
    test_angular_distr = ddxs_test.apply(integrate, axis=1)
    ddxs_angular_distr = ddxs.apply(integrate, axis=1)
    assert abs(1 - ddxs_angular_distr / test_angular_distr).max() <= 0.03
