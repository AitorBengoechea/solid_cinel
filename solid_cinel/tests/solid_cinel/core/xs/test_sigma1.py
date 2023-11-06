import pytest
import os
import pandas as pd
from solid_cinel import default_Eout, Dxs


@pytest.mark.parametrize("T", [300, 1000])
def test_sigma1(T):
    M = 238.05077040419212
    wd = os.getcwd()
    os.chdir(__file__.replace("test_sigma1.py", ""))
    # Get test data:
    xs_test = pd.read_hdf(f"u238.{T}.2", "test")
    # Get 0K data:
    os.chdir("../../../../data/xs/U238/")
    xs_0K = pd.read_hdf("u238.0.2", key="elastic")
    os.chdir(wd)
    # Remove duplicated index:
    xs_test = xs_test[~xs_test.index.duplicated(keep='first')]
    for Ein in xs_test.index[xs_test.index <= 100]:
        Eout = default_Eout(Ein)
        xs_broad = Dxs.from_sigma1(xs_0K, Ein, M, T, Eout).integral
        assert abs(1 - xs_broad/xs_test.loc[Ein]) < 0.8


