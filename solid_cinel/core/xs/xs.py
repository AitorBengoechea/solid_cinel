import numpy as np
import pandas as pd
import numba as nb
from math import pi
from numba import prange
from solid_cinel.core.material.vibration.tau import tau_n_functions
from solid_cinel.core.material.scattering_function.scatfunc import ScatFunc, get_scatfunc_pdos_row
from solid_cinel.core.xs import XsMat, Ein_arno_row, Db
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

        # Doppler broadening using 4PCF(SCT) algorithm:
        >>> Ein = np.array([2.0, 6.67])
        >>> from solid_cinel.core.material.vibration.pdos import Pdos
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> xs.Doppler_broad(T, pdos, Ein_grid=Ein, num_Eout=1000, prob=True, model="sct").round(6)
                      xs  downscattering  upscattering  Ein=Eout
        Ein
        2.00    8.310310        0.590390      0.407398  0.002211
        6.67  281.437747        0.733844      0.262325  0.003831
        """
        if Ein_grid is None:
            Ein_grid_ = self.xs_0K.index.values
        else:
            Ein_grid_ = Ein_grid if hasattr(Ein_grid, '__len__') else [Ein_grid]
        theta = np.arange(1, 180 + theta_diff, theta_diff)
        if kwargs.get('model') == "pdos":
            nphonon = kwargs.get('nphonon', 1000)
            threshold = kwargs.get('threshold', 0.0)
            return self.clm_db(self.xs_0K, Ein_grid_, self.M, T_new, theta,
                               num_Eout, prob, *args, nphonon=nphonon,
                               threshold=threshold)
        else:
            return self.sct_db(self.xs_0K, Ein_grid_, self.M, T_new, theta,
                                num_Eout, prob, *args, **kwargs)

    @staticmethod
    def clm_db(xs_0K: pd.Series, Ein_grid: np.ndarray, M: float, T: float,
               theta: np.ndarray, num_Eout: int, prob: bool, pdos,
               nphonon: int = 1000, threshold: float = 0.0):
        """

        Parameters
        ----------
        xs_0K: pd.Series
            Cross section at 0K.
        Ein_grid: np.ndarray
            Incident energy grid for doppler broadened cross section.
        M: float
            Mass of the target in amu.
        T: float
            Temperature in Kelvin.
        theta: np.ndarray
            Scattering angle grid.
        num_Eout: int
            Number of energy grid for outgoing energy grid in each Ein grid.
        prob: bool
            If True, return probability of upscattering and downscattering.
            Default is False.
        pdos: 'solid_cinel.core.material.Pdos'
            Pdos object.
        nphonon: int
            Phonon expansion order. Default is 1000.
        threshold: float
            Minimun value to take into account in the creation of tau_n functions.

        Returns
        -------
        pd.DataFrame
            Cross section or cross section and probability of upscattering and
            downscattering.

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
        >>> theta = np.array([10, 46, 90, 180])
        >>> Ein_grid = np.array([2.0, 6.67, 36.6])
        >>> from solid_cinel.core.material.vibration.pdos import Pdos
        >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
        >>> Xs.clm_db(xs_0K, Ein_grid, M, T, theta, 1000, True, pdos, nphonon=10, threshold=0.0).round(6)
                       xs  downscattering  upscattering  Ein=Eout
        Ein
        2.00     9.909883        0.727215      0.271357  0.001428
        6.67    40.636358        0.633586      0.360698  0.005716
        36.60  746.548920        0.521897      0.452571  0.025532
        """
        # Get common variables:
        xs_0K_values, xs_0K_E = xs_0K.values, xs_0K.index.values
        mu = np.sort(np.cos(np.deg2rad(theta)))
        T_arno = T * (1 + mu) / 2
        # Get Scattering function data:
        tau_n_scatt, delta_beta_scatt, debye_waller_coeff_scatt = pdos.get_clm_param(T, nphonon=nphonon, threshold=threshold)
        # 1 Scatfunct for getting mu_fit:
        Eout = np.linspace(Ein_grid[0] * 0.9, Ein_grid[0] * 1.1, num_Eout)
        mu_fit = ScatFunc.from_tau(Ein_grid[0], M, T, Eout, theta, tau_n_scatt,
                                   delta_beta_scatt,
                                   debye_waller_coeff_scatt).get_angle

        # Create xs_mat creation data:
        tau1, DebyeWallerCoeff, delta_beta = XsMat.get_pdos_variables(pdos, T_arno)

        # Create a list to hold the results
        if mu[0] == np.cos(pi):
            result = ddxs_clm_0K(Ein_grid, num_Eout, M, T,
                        tau_n_scatt, delta_beta_scatt, debye_waller_coeff_scatt,
                        xs_0K_values, xs_0K_E, prob)
            start = 1
        else:
            result = []
            start = 0
        for i in range(start, len(theta)):
            # Create angle tau_n function:
            tau_n_angle = tau_n_functions(tau1[i], delta_beta[i], nphonon,
                                          threshold)
            # Select the especific data for the next function:
            for Ein in Ein_grid:
                # Gen Eout grid:
                Eout = np.linspace(Ein * 0.9, Ein * 1.1, num_Eout)
                row_results = ddxs_clm_row(Ein, M, T, Eout, mu[i], tau_n_scatt,
                                           delta_beta_scatt,
                                           debye_waller_coeff_scatt,
                                           tau_n_angle, delta_beta[i],
                                           DebyeWallerCoeff[i], xs_0K_values,
                                           xs_0K_E, mu_fit, T_arno[i])

                Ein_results = [mu[i], Ein, np.trapz(row_results, x=Eout)]

                # Get probability of upscattering and downscattering:
                if prob:
                    mask_up, mask_down = Eout > Ein, Eout < Ein
                    Ein_results.append(np.trapz(row_results[mask_up], x=Eout[mask_up]))
                    Ein_results.append(np.trapz(row_results[mask_down], x=Eout[mask_down]))

                # Update results:
                result.append(Ein_results)

        if prob:
            df = pd.DataFrame(result, columns=["mu", "Ein", "xs", "xs_up", "xs_down"])
            df_grouped = df.groupby("Ein")
            xs_db = df_grouped.apply(lambda group: pd.Series({
                'xs': np.trapz(group['xs'], x=group['mu']),
                'upscattering': np.trapz(group['xs_up'], x=group['mu']),
                'downscattering': np.trapz(group['xs_down'], x=group['mu'])
            }))
            xs_db['upscattering'] /= xs_db['xs']
            xs_db['downscattering'] /= xs_db['xs']
            xs_db['Ein=Eout'] = 1.0 - xs_db['upscattering'] - xs_db['downscattering']
            xs_db = xs_db[["xs", "downscattering", "upscattering", "Ein=Eout"]]
        else:
            df = pd.DataFrame(result, columns=["mu", "Ein", "xs"])
            df_grouped = df.groupby("Ein")
            xs_db = df_grouped.apply(lambda group: pd.Series({
                'xs': np.trapz(group['xs'], x=group['mu'])}))
        return xs_db

    @staticmethod
    def sct_db(xs_0K: pd.Series, Ein_grid: np.ndarray, M: float, T: float,
               theta: np.ndarray, num_Eout: int, prob: bool, *args, **kwargs) -> pd.DataFrame:
        """
        Doppler broadening of cross section using 4PCF (SCT) algorithm

        Parameters
        ----------
        xs_0K: pd.Series
            Cross section at 0K.
        Ein_grid: np.ndarray
            Incident energy grid for output cross section.
        M: float
            Mass of the target.
        T: float
            Temperature in Kelvin.
        theta: np.ndarray
            Scattering angle grid.
        num_Eout: int
            Number of energy grid for outgoing energy grid
        prob: bool
            If True, return probability of upscattering and downscattering.
            Default is False.

        Parameters for 4PCF (FGM or SCT) algorithm
        ------------------------------------------
        model: str
            Model for 4PCF. Default is None, so sigma1 algorith is going to be used. The available models are:
                - "fgm": Free Gas Model
                - "sct": Short Collision Time model

        Parameters for 4PCF (SCT) algorithm
        -----------------------------------
        pdos : 'solid_cinel.core.material.Pdos'
            Pdos object

        Returns
        -------
        pd.DataFrame
            Cross section or cross section and probability of upscattering and
            downscattering.

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
        >>> Ein_grid = [2.0]
        >>> theta = np.arange(1, 181, 1)
        >>> num_Eout = 1000
        >>> Xs.sct_db(xs_0K, Ein_grid, M, T, theta, num_Eout, prob=True).round(6)
                   xs  downscattering  upscattering  Ein=Eout
        Ein
        2.0  8.313812        0.590631      0.407155  0.002214
        """
        results = {}
        for Ein in Ein_grid:
            Eout = np.linspace(Ein * 0.95, Ein * 1.05, num_Eout)
            ddxs = DDxs.from_4PCF(xs_0K, Ein, M, T, Eout, theta, *args,
                                  **kwargs)
            result = {"xs": ddxs.integral}
            if prob:
                result.update(ddxs.E_prob)
            results[Ein] = result
        xs_db = pd.DataFrame(results).T.sort_index()
        xs_db.index.name = "Ein"
        return xs_db[["xs", "downscattering", "upscattering", "Ein=Eout"]] if prob else xs_db


def ddxs_clm_0K(Ein_grid: np.ndarray, num_Eout: int, M: float, T: float,
                tau_n_scatt: np.ndarray, delta_beta_scatt: float, debye_waller_coeff_scatt: float,
                xs_0K_values: np.ndarray, xs_0K_E: np.ndarray, prob: bool) -> list:
    """
    Compute the ddxs for 180 degree in clm model using

    Parameters
    ----------
    Ein_grid : np.ndarray
        Incoming energy grid.
    num_Eout : int
        Number of energy grid for outgoing energy grid.
    M : float
        Mass of the target in amu.
    T : float
        Temperature in kelvin.
    tau_n_scatt : np.ndarray
        Tau(-beta) function for n expansion for calculation of the scattering function.
    delta_beta_scatt : float
        Interval of beta for the scattering function.
    debye_waller_coeff_scatt : float
        Debye-Waller coefficient for the scattering function.
    xs_0K_values : np.ndarray
        Cross section values at 0K.
    xs_0K_E : np.ndarray
        Cross section energy grid.
    prob : bool
        If True, return probability of upscattering and downscattering.

    Returns
    -------
    list
        Cross section or cross section and probability of upscattering and
        downscattering.

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
    >>> Ein_grid = np.array([2.0, 6.67, 36.6])
    >>> num_Eout = 1000
    >>> xs_0K_values, xs_0K_E = xs_0K.values, xs_0K.index.values
    >>> from solid_cinel.core.material.vibration.pdos import Pdos
    >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
    >>> tau_n_scatt, delta_beta_scatt, debye_waller_coeff_scatt = pdos.get_clm_param(T, nphonon=10, threshold=0.0)
    >>> result = ddxs_clm_0K(Ein_grid, num_Eout, M, T, tau_n_scatt, delta_beta_scatt, debye_waller_coeff_scatt, xs_0K_values, xs_0K_E, True)
    >>> pd.DataFrame(result, columns=["mu", "Ein", "xs", "xs_up", "xs_down"]).round(6)
        mu    Ein   xs  xs_up  xs_down
    0 -1.0   2.00  0.0    0.0      0.0
    1 -1.0   6.67  0.0    0.0      0.0
    2 -1.0  36.60  0.0    0.0      0.0
    """
    result = []
    tau_n_beta_scatt = np.arange(tau_n_scatt.shape[1]) * delta_beta_scatt
    for Ein in Ein_grid:
        # Gen Eout grid:
        Eout = np.linspace(Ein * 0.9, Ein * 1.1, num_Eout)
        Ein_row = Ein_arno_row(Ein, Eout, -1.0, M)
        scattfunc_row = get_scatfunc_pdos_row(Ein, M, T, Eout, -1.0,
                                              tau_n_scatt,
                                              tau_n_beta_scatt,
                                              debye_waller_coeff_scatt)
        row_results = scattfunc_row * np.interp(Ein_row, xs_0K_E, xs_0K_values)
        Ein_results = [-1.0, Ein, np.trapz(row_results, x=Eout)]

        # Get probability of upscattering and downscattering:
        if prob:
            mask_up, mask_down = Eout > Ein, Eout < Ein
            Ein_results.append(np.trapz(row_results[mask_up], x=Eout[mask_up]))
            Ein_results.append(np.trapz(row_results[mask_down], x=Eout[mask_down]))

        # Update results:
        result.append(Ein_results)
    return result


@nb.jit(nopython=True, nogil=True, cache=True)
def ddxs_clm_row(Ein: float, M: float, T: float, Eout: np.ndarray, mu: float, tau_n_scatt: np.ndarray,
                 delta_beta_scatt: float, debye_waller_coeff_scatt: float, tau_n_mu: np.ndarray, delta_beta_mu: float,
                 debye_waller_coeff_mu: float, xs_0K_values: np.ndarray, xs_0K_E: np.ndarray, mu_fit: float,
                 T_mu: float) -> np.ndarray:
    """
    Compute the ddxs for a given angle and energy.

    Parameters
    ----------
    Ein : float
        Incoming energy.
    M : float
        Mass of the target.
    T : float
        Temperature.
    Eout : np.ndarray
        Outgoing energy grid.
    mu : float
        Cosine of the scattering angle.
    tau_n_scatt : np.ndarray
        Tau(-beta) function for n expansion for calculation of the scattering function.
    delta_beta_scatt : float
        Interval of beta for the scattering function.
    debye_waller_coeff_scatt : float
        Debye-Waller coefficient for the scattering function.
    tau_n_mu : np.ndarray
        Tau(-beta) function for n expansion for calculation of the xs_mat in the selected mu.
    delta_beta_mu : float
        Interval of beta for the xs_mat in the selected mu.
    debye_waller_coeff_mu : float
        Debye-Waller coefficient for the xs_mat in the selected mu.
    xs_values : np.ndarray
        Cross section values at 0K.
    xs_E : np.ndarray
        Cross section energy grid.
    mu_fit : float
        Cosine of the scattering angle for the scattering function.
    T_mu : float
        Temperature for the xs_mat in the selected mu.

    Returns
    -------
    np.ndarray, (len(Eout),)
        ddxs for a given angle and energy.
    """
    # Scattering function for selected angle and Ein:
    tau_n_beta_scatt = np.arange(tau_n_scatt.shape[1]) * delta_beta_scatt
    scattfunc_row = get_scatfunc_pdos_row(Ein, M, T, Eout, mu,
                                          tau_n_scatt,
                                          tau_n_beta_scatt,
                                          debye_waller_coeff_scatt)

    # xs_mat row for selected angle and Ein:
    Ein_row = Ein_arno_row(Ein, Eout, mu, M)
    xs_mat_row = db_pdos_row(tau_n_mu, delta_beta_mu,
                             debye_waller_coeff_mu,
                             Ein_row, T_mu,
                             xs_0K_values, xs_0K_E, mu_fit, M)

    return scattfunc_row * xs_mat_row


@nb.jit(nopython=True, nogil=True, parallel=True, cache=True)
def db_pdos_row(tau_n: np.ndarray, delta_beta: float, debyewallercoeff: float, Ein_row: np.ndarray, T: float,
                xs_0K_values: np.ndarray, xs_0K_E: np.ndarray, mu_fit: float, M: float):
    """
    Compute the xs_mat for a given angle and energy

    Parameters
    ----------
    tau_n : np.ndarray
        Tau(-beta) function for n expansion for calculation of the xs_mat in the selected mu.
    delta_beta : float
        Interval of beta for the xs_mat in the selected mu.
    debyewallercoeff : float
        Debye-Waller coefficient for the xs_mat in the selected mu.
    Ein_row : np.ndarray
        Incoming energy grid for the xs_mat.
    T : float
        Temperature for the xs_mat in the selected mu.
    xs_0K_values : np.ndarray
        Cross section values at 0K.
    xs_0K_E : np.ndarray
        Cross section energy grid.
    """
    tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    xs_mat = np.empty(len(Ein_row))
    for j in prange(len(Ein_row)):
        Eout_db = np.linspace(Ein_row[j] * 0.9, Ein_row[j] * 1.1, 3000)
        pdf = get_scatfunc_pdos_row(Ein_row[j], M, T,
                                    Eout_db, mu_fit, tau_n, tau_n_beta,
                                    debyewallercoeff)
        xs_mat[j] = Db(xs_0K_values, xs_0K_E, Ein_row[j], Eout_db, pdf)
    return xs_mat
