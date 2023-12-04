import numpy as np
import pandas as pd
from solid_cinel.core.xs.ddxs import DDxs

# Example variables:
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


class Xs:
    def __init__(self, xs_0K: pd.Series, M: float):
        self.M = M
        self.xs_0K = xs_0K[~xs_0K.index.duplicated(keep='first')]

    def Doppler_broad(self, T_new: float, *args,
                      Ein_grid: [float, np.ndarray] = None,
                      num_Eout: int = 3000,
                      theta_diff: float = 1.0,
                      prob: bool = False,
                      **kwargs) -> [pd.Series, pd.DataFrame]:
        """
        Doppler broadening of cross section using 4PCF

        Parameters
        ----------
        T_new: float
            New temperature
        Ein_grid: [float, np.ndarray]
            Energy grid for output cross section
        num_Eout: int
            Number of energy grid for outgoing energy grid. Default is 3000.
        theta_diff: float
            Step size of scattering angle in degree. Default is 1.0.
        prob: bool
            If True, return probability of upscattering and downscattering.
            Default is False.
        model: str
            Model for 4PCF. Default is None, so sigma1 algorith is going to be
            use. The available models are:
                - "fgm": Free Gas Model
                - "sct": Short Collision Time model
                - "pdos": Phonon Density of State model

        4PCF parameters for model = "sct"
        ---------------------------------
        pdos: "Pdos"
            Phonon Density of State

        4PCF parameters for model = "pdos"
        ----------------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object.
        threshold : 'float', optional
            Minimun value to take into account in the creation of tau_n
            functions. For T>200 is convenient to set into 1.0e-14 to speed up
            the calculations. The default is 0.0.
        nphonon : 'int', optional
            Phonon expansion order. The default is 1000.

        Returns
        -------
        pd.Series or pd.DataFrame
            Cross section or cross section and probability of upscattering and
            downscattering.

        Examples
        --------
        Examples
        --------
        # 0K xs data for U238:
        >>> import os
        >>> wd = os.getcwd()
        >>> os.chdir(__file__.replace("xs.py", ""))
        >>> os.chdir("../../data/xs/U238/")
        >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
        >>> os.chdir(wd)

        # Generate DDXS test variables:
        >>> T = 1000
        >>> M = 238.05077040419212
        >>> xs = Xs(xs_0K, M)

        # Doppler broadening using 4PCF(SIGMA1) algorithm:
        >>> Ein = 2.0
        >>> xs.Doppler_broad(T, Ein_grid=Ein, num_Eout=1000).round(6)
                   xs
        Ein
        2.0  8.313812

        >>> xs.Doppler_broad(T, Ein_grid=Ein, num_Eout=1000, prob=True).round(6)
             Ein=Eout  downscattering  upscattering        xs
        Ein
        2.0  0.002214        0.590631      0.407155  8.313812

        >>> Ein = [2.0, 6.67]
        >>> xs.Doppler_broad(T, Ein_grid=Ein, num_Eout=1000, prob=True).round(6)
                      xs  upscattering  downscattering  Ein=Eout
        Ein
        2.00    8.313812      0.407155        0.590631  0.002214
        6.67  282.747095      0.259304        0.736903  0.003793

        # Doppler broadening using 4PCF(SCT) algorithm:
        >>> from solid_cinel.core.material.vibration.pdos import Pdos
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> xs.Doppler_broad(T, pdos, Ein_grid=Ein, num_Eout=1000, prob=True, model="sct").round(6)
                      xs  upscattering  downscattering  Ein=Eout
        Ein
        2.00    8.310310      0.407398        0.590390  0.002211
        6.67  281.437747      0.262325        0.733844  0.003831

        # Doppler broadening using 4PCF(PDOS) algorithm (not very accurate):
#        >>> xs.Doppler_broad(T, pdos, nphonon=10, Ein_grid=Ein, num_Eout=1000, theta_diff= 15, prob=True, model="pdos").round(6)
#                     xs  upscattering  downscattering  Ein=Eout
#        Ein
#        2.00   2.179519      0.439105        0.557491  0.003404
#        6.67  33.519221      0.458024        0.529928  0.012048
        """
        if Ein_grid:
            Ein_grid_ = Ein_grid if hasattr(Ein_grid, '__len__') else [Ein_grid]
        else:
            Ein_grid_ = self.xs_0K.index.values
        theta = np.arange(1, 180 + theta_diff, theta_diff)
        xs_db = {}
        for Ein in Ein_grid_:
            Eout = np.linspace(Ein * 0.95, Ein * 1.05, num_Eout)
            ddxs = DDxs.from_4PCF(self.xs_0K, Ein, self.M, T_new, Eout, theta, *args, **kwargs)
            xs_db[Ein] = {"xs": ddxs.integral}
            if prob:
                xs_db[Ein].update(ddxs.E_prob)
        xs_db = pd.DataFrame(xs_db).T.sort_index()
        xs_db.index.name = "Ein"
        return xs_db
