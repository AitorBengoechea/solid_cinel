import pytest
import numpy as np
import os
import pandas as pd
from solid_cinel.core.material.scattering_function.scatfunc import ScatFunc
from solid_cinel.core.generic import integrate


@pytest.mark.parametrize("Ein, M, T", [(0.1000000E-01, 238.0, 1000)])
def test_sab_fgm_scatfunc(Ein, M, T):
    # Get test data:
    wd = os.getcwd()
    os.chdir(__file__.replace("test_Sab.py", ""))
    ddxs_test = pd.read_hdf('Sab_scatt_constant', 'test')
    os.chdir(wd)
    # Generate data:
    theta = np.arange(0, 181, 1)[1::]
    ddxs = ScatFunc.from_model(Ein, M, T, ddxs_test.columns.values, theta).data

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
