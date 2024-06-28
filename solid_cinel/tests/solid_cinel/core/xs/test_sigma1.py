import pytest
import os
import pandas as pd
from solid_cinel.core.xs.xs import default_Eout
from solid_cinel.core.xs.dxs import Dxs


@pytest.mark.parametrize("T", [300, 1000])
def test_sigma1(T):
    M = 238.05077040419212
    wd = os.getcwd()
    os.chdir(__file__.replace("test_sigma1.py", ""))
    # Get test data:
    xs_test = pd.read_hdf(f"u238.{T}.2", "test")
    xsTest = pd.read_csv(f"u238.{T}.2", sep='\s+', header=None, index_col=0, usecols=[0, 1], engine="python").iloc[::, 0]
    xsTest.index.name = "E"
    xsTest = xsTest.reset_index().drop_duplicates(subset='E', keep='first').set_index('E').iloc[:, 0]
    # Get 0K data:
    os.chdir("../../../../data/xs/U238/")
    xs0K = pd.read_csv("u238.0.2", sep='\s+', header=None, index_col=0, usecols=[0, 1], engine="python").iloc[::, 0]
    xs0K.index.name = "E"
    xs0K = xs0K.reset_index().drop_duplicates(subset='E', keep='first').set_index('E').iloc[:, 0]
    os.chdir(wd)
    # Remove duplicated index:
    for Ein in xsTest.index[xsTest.index <= 100]:
        Eout = default_Eout(Ein)
        xs_broad = Dxs.from_sigma1(xs0K, Ein, M, T, Eout).integral
        assert abs(1 - xs_broad/xs_test.loc[Ein]) < 0.8


