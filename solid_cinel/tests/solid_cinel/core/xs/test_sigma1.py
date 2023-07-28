import pytest
import os
import numpy as np
import pandas as pd
from solid_cinel.core.xs.doppler_broad import get_DB
from solid_cinel.core.generic import integrate

M = 238.05077040419212
@pytest.mark.parametrize("T", [300, 1000])
def test_sigma1(T):
    wd = os.getcwd()
    os.chdir(__file__.replace("test_sigma1.py", ""))
    # Get test data:
    xs_test = pd.read_csv(f"u238.{T}.2", sep="    ", header=None, engine="python")\
                .set_index(0).drop([2], axis=1)
    # Get 0K data:
    os.chdir("../../../../data/xs/U238/")
    xs_0K = pd.read_csv("u238.0.2", sep="    ", header=None, engine="python")\
              .set_index(0).drop([2], axis=1)
    os.chdir(wd)
    # Remove duplicated index:
    xs_0K = xs_0K[~xs_0K.index.duplicated(keep='first')]
    xs_test = xs_test[~xs_test.index.duplicated(keep='first')]
    for Ein in xs_test.index[xs_test.index <= 100]:
        Eout_small = np.linspace(0,
                                  0.99 * Ein,
                                  num=1000, endpoint=False)
        Eout_middle = np.linspace(0.99 * Ein,
                                  Ein * 1.01,
                                  num=1000,
                                  endpoint=False)
        if Ein * 2 < 5.0:
            Eout_great = np.logspace(np.log10(Ein * 1.01),
                                     np.log10(5.0),
                                     num=1500, endpoint=True)
        else:
            Eout_great = np.logspace(np.log10(Ein * 1.01),
                                     np.log10(2 * Ein),
                                     num=1500, endpoint=True)
        E_out = np.sort(np.concatenate((Eout_great, Eout_small, Eout_middle)))
        xs_broad = integrate(get_DB(xs_0K, Ein, E_out, M, T, algorithm="sigma1"))
        assert abs(1 - xs_broad/xs_test.loc[Ein].values) < 0.8


