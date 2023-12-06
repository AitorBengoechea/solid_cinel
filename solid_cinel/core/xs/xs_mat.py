"""
Python for working with the Xs matrix from 4PCF.

@author: AB272525
"""
import numpy as np
import pandas as pd
import numba as nb
import h5py
import re
from numba import prange
from scipy.constants import physical_constants as const
from solid_cinel.core.material.scattering_function.scatfunc import sigma1, get_scat_sct_angular, get_ScatFunc_pdos_angle
from solid_cinel.core.material.vibration.tau import tau_n_functions
import dask
from solid_cinel.core.xs.dxs import Dxs
import os
from math import pi

from typing import Iterable

# constants
kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]

# Avoid numba fast math:
nb.config.FASTMATH_DEFAULT = False

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

class XsMat:
    """
    Xs matrix class
    """
    def __init__(self, xs_0K, Ein, M, T, *args, **kwargs):
        self.xs_0K = xs_0K
        self.Ein = Ein
        self.M = M
        self.T = T
        self.data = pd.DataFrame(*args, **kwargs)

    @property
    def data(self) -> pd.DataFrame:
        """
        DDXS data.

        Returns
        -------
        pd.DataFrame
            DDXS data
        """
        return self._data

    @data.setter
    def data(self, dd_pdf: Iterable):
        """
        Set the diferential data.

        Parameters
        ----------
        dd_pdf : pd.DataFrame
            Double differential scattering function data
        """
        dd_pdf_ = pd.DataFrame(dd_pdf).sort_index(axis=0).sort_index(axis=1)
        dd_pdf_.index.name = "mu"
        dd_pdf_.columns.name = "Eout"
        self._data = dd_pdf_
    @classmethod
    def from_model(cls, *args, **kwargs):
        """
         Calculate the cross section matrix for a given incident energy, target mass,
         target temperature, outgoing energy grid and outgoing angle grid using arno
         model with the most similar S(alpha, -beta) distribution with sigma1
         .. math::
             \sigma^{T(1+\mu)/2}\left( \frac{E + E^\prime}{2} - E\frac{\mu m}{M}\right)

         Parameters
         ----------
         mu_fit : float
             The cosine of the outgoing angle to fit the S(alpha, -beta) distribution
             with sigma1

         Parameters for fgm, sct and pdos models
         ---------------------------------------
         xs_0K : pd.Series
             Cross section at 0K in barns
         Ein : float
             The incident energy of the neutron in eV
         M : float
             Mass of the material in amu
         T : float
             Temperature of the material in K
         Eout : np.array, (N,)
             The neutron outgoing energy grid in eV
         theta : np.array, (M,)
             The neutron outgoing angle grid in degrees (0, 180]

         Extra parameters for sct
         ------------------------
         Teff : float
             Effective temperature of the material in K

         Extra parameters for pdos
         -------------------------
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
         np.ndarray, (M, N)
             Cross section matrix in barns

         Examples
         --------
         # 0K xs data for U238:
         >>> wd = os.getcwd()
         >>> os.chdir(__file__.replace("xs_mat.py", ""))
         >>> os.chdir("../../data/xs/U238/")
         >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
         >>> os.chdir(wd)

         >>> T = 1000
         >>> Ein = 2.0
         >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 7)
         >>> M = 238.05077040419212
         >>> theta = np.arange(10, 190, 10)
         >>> mu_fit = np.cos(np.deg2rad(60))

         # sigma1 model:
         >>> xs_values = XsMat.from_model(xs_0K, Ein, M, T, Eout, theta)
         >>> pd.DataFrame(xs_values.data.values, index=theta[::-1], columns=Eout).round(6)
             1.800000	1.866667	1.933333	2.000000	2.066667	2.133333	2.200000
         180 9.102355	9.095532	9.088710	9.081758	9.074679	9.067600	9.060521
         170 9.102381	9.095558	9.088736	9.081785	9.074706	9.067627	9.060548
         160 9.102454	9.095632	9.088810	9.081861	9.074782	9.067703	9.060625
         150 9.102577	9.095755	9.088932	9.081987	9.074910	9.067831	9.060753
         140 9.102746	9.095924	9.089098	9.082158	9.075085	9.068007	9.060928
         130 9.102952	9.096130	9.089299	9.082363	9.075297	9.068219	9.061139
         120 9.103190	9.096369	9.089534	9.082602	9.075545	9.068466	9.061386
         110 9.103451	9.096632	9.089797	9.082865	9.075817	9.068740	9.061657
         100 9.103729	9.096912	9.090074	9.083149	9.076110	9.069031	9.061947
         90	 9.104017	9.097203	9.090360	9.083438	9.076408	9.069334	9.062245
         80	 9.104301	9.097490	9.090649	9.083730	9.076705	9.069633	9.062545
         70	 9.104579	9.097769	9.090927	9.084011	9.076995	9.069924	9.062834
         60	 9.104837	9.098033	9.091189	9.084274	9.077265	9.070196	9.063105
         50	 9.105070	9.098270	9.091426	9.084513	9.077508	9.070442	9.063350
         40	 9.105269	9.098471	9.091631	9.084720	9.077716	9.070655	9.063557
         30	 9.105425	9.098635	9.091795	9.084887	9.077888	9.070823	9.063725
         20	 9.105525	9.098748	9.091915	9.085010	9.078011	9.070941	9.063833
         10	 9.105489	9.098775	9.091979	9.085087	9.078074	9.070973	9.063803

         # fgm model:
         >>> xs_values = XsMat.from_model(xs_0K, Ein, M, T, Eout, theta, mu_fit, model="fgm")
         >>> pd.DataFrame(xs_values.data.values, index=theta[::-1], columns=Eout).round(6)
             1.800000	1.866667	1.933333	2.000000	2.066667	2.133333	2.200000
         180 9.102355	9.095532	9.088710	9.081758	9.074679	9.067600	9.060521
         170 9.102381	9.095559	9.088737	9.081785	9.074706	9.067627	9.060549
         160 9.102456	9.095634	9.088811	9.081863	9.074784	9.067705	9.060627
         150 9.102577	9.095755	9.088932	9.081987	9.074910	9.067831	9.060752
         140 9.102745	9.095923	9.089097	9.082157	9.075084	9.068005	9.060926
         130 9.102951	9.096129	9.089300	9.082364	9.075298	9.068219	9.061140
         120 9.103189	9.096367	9.089535	9.082603	9.075546	9.068467	9.061386
         110 9.103451	9.096631	9.089796	9.082867	9.075819	9.068741	9.061658
         100 9.103729	9.096911	9.090073	9.083148	9.076109	9.069033	9.061948
         90	 9.104018	9.097203	9.090360	9.083438	9.076407	9.069333	9.062246
         80	 9.104303	9.097492	9.090650	9.083728	9.076705	9.069633	9.062545
         70	 9.104579	9.097771	9.090929	9.084012	9.076996	9.069924	9.062834
         60	 9.104837	9.098033	9.091191	9.084276	9.077266	9.070196	9.063105
         50	 9.105070	9.098269	9.091428	9.084515	9.077509	9.070443	9.063350
         40	 9.105269	9.098473	9.091632	9.084722	9.077717	9.070654	9.063560
         30	 9.105427	9.098637	9.091796	9.084887	9.077888	9.070825	9.063727
         20	 9.105526	9.098748	9.091917	9.085012	9.078012	9.070942	9.063833
         10	 9.105490	9.098776	9.091979	9.085087	9.078076	9.070974	9.063804

         # sct model:
         >>> from solid_cinel.core.material.vibration.pdos import Pdos
         >>> pdos = Pdos.from_dE(rho_in_energy_U238, interv_in_energy_U238)
         >>> xs_values = XsMat.from_model(xs_0K, Ein, M, T, Eout, theta, mu_fit, pdos, model="sct")
         >>> pd.DataFrame(xs_values.data.values, index=theta[::-1], columns=Eout).round(6)
              1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
         180  9.102355  9.095532  9.088710  9.081758  9.074679  9.067600  9.060521
         170  9.102371  9.095549  9.088724  9.081772  9.074696  9.067617  9.060539
         160  9.102449  9.095627  9.088802  9.081853  9.074777  9.067698  9.060619
         150  9.102574  9.095752  9.088926  9.081981  9.074906  9.067828  9.060749
         140  9.102744  9.095919  9.089091  9.082151  9.075080  9.068001  9.060922
         130  9.102948  9.096126  9.089296  9.082360  9.075295  9.068217  9.061137
         120  9.103187  9.096365  9.089532  9.082600  9.075544  9.068465  9.061384
         110  9.103449  9.096629  9.089793  9.082865  9.075817  9.068739  9.061657
         100  9.103730  9.096910  9.090072  9.083146  9.076107  9.069031  9.061946
         90   9.104016  9.097202  9.090362  9.083436  9.076406  9.069332  9.062245
         80   9.104302  9.097491  9.090649  9.083729  9.076704  9.069632  9.062544
         70   9.104578  9.097770  9.090928  9.084011  9.076995  9.069923  9.062833
         60   9.104837  9.098032  9.091190  9.084275  9.077265  9.070196  9.063104
         50   9.105069  9.098269  9.091427  9.084514  9.077509  9.070442  9.063349
         40   9.105269  9.098472  9.091632  9.084721  9.077716  9.070654  9.063559
         30   9.105426  9.098636  9.091798  9.084886  9.077887  9.070824  9.063726
         20   9.105528  9.098748  9.091917  9.085011  9.078012  9.070941  9.063832
         10   9.105490  9.098775  9.091979  9.085086  9.078076  9.070973  9.063803

         # pdos model:
         >>> nphonon = 100
         >>> threshold = 1.0e-14
         >>> xs_values = XsMat.from_model(xs_0K, Ein, M, T, Eout, theta, mu_fit, pdos, nphonon=nphonon, threshold=threshold, model="pdos")
         >>> pd.DataFrame(xs_values.data.values, index=theta[::-1], columns=Eout).round(6)
              1.800000  1.866667  1.933333  2.000000  2.066667  2.133333  2.200000
         180  9.102355  9.095532  9.088710  9.081758  9.074679  9.067600  9.060521
         170  9.103715  9.096910  9.090104  9.083212  9.076165  9.069103  9.062042
         160  9.103626  9.096821  9.090015  9.083121  9.076074  9.069013  9.061952
         150  9.103167  9.096364  9.089558  9.082651  9.075602  9.068542  9.061485
         140  9.102780  9.095979  9.089172  9.082254  9.075208  9.068149  9.061094
         130  9.102666  9.095866  9.089056  9.082135  9.075097  9.068042  9.060984
         120  9.102740  9.095944  9.089135  9.082217  9.075183  9.068131  9.061076
         110  9.102926  9.096135  9.089323  9.082409  9.075386  9.068336  9.061282
         100  9.103175  9.096384  9.089570  9.082664  9.075646  9.068599  9.061543
         90   9.103449  9.096661  9.089848  9.082944  9.075937  9.068892  9.061832
         80   9.103734  9.096949  9.090132  9.083232  9.076233  9.069190  9.062131
         70   9.104011  9.097229  9.090414  9.083518  9.076522  9.069484  9.062421
         60   9.104271  9.097492  9.090677  9.083783  9.076796  9.069758  9.062696
         50   9.104647  9.097872  9.091058  9.084168  9.077187  9.070151  9.063088
         40   9.104847  9.098078  9.091263  9.084378  9.077398  9.070365  9.063300
         30   9.105006  9.098241  9.091431  9.084545  9.077570  9.070538  9.063469
         20   9.105109  9.098356  9.091553  9.084671  9.077696  9.070658  9.063578
         10   9.105072  9.098383  9.091617  9.084748  9.077762  9.070689  9.063551

         Dirac delta test for Teff calculation and pdos model:
         >>> Ein = 36.68
         >>> M = 238.05077040419212
         >>> T = 300
         >>> theta = np.arange(1, 11, 1)
         >>> Eout = np.linspace(Ein * 0.9 , Ein * 1.1, 7)[:6]
         >>> xs_values = XsMat.from_model(xs_0K, Ein, M, T, Eout, theta, mu_fit, pdos, model="sct")
         >>> pd.DataFrame(xs_values.data.values, index=theta[::-1], columns=Eout).round(6)
             33.012000  34.234667  35.457333    36.680000   37.902667  39.125333
         10   0.781011   0.212474  18.943993  7832.112249  116.900933  48.371957
         9    0.774634   0.214952  18.935478  7827.100451  116.942587  48.337172
         8    0.765202   0.218967  18.947906  7822.491592  116.933518  48.278974
         7    0.750937   0.225468  18.991055  7818.332543  116.851287  48.184526
         6    0.728496   0.236354  19.083733  7814.650622  116.653779  48.029743
         5    0.690994   0.255950  19.265986  7811.484114  116.253915  47.765243
         4    0.622453   0.296031  19.636420  7808.858742  115.447043  47.276543
         3    0.479921   0.399744  20.498496  7806.792841  113.658175  46.249378
         2    0.151264   0.848392  23.250511  7805.305048  108.749223  43.580322
         1   39.411876  15.646885  48.317676  7804.407359   88.153494  33.909257

         """
        # Common arguments:
        if len(args) == 6:
            xs_0K, Ein, M, T, Eout, theta = args
        elif len(args) == 7:
            xs_0K, Ein, M, T, Eout, theta, mu_fit = args
        else:
            xs_0K, Ein, M, T, Eout, theta, mu_fit, pdos = args

        # Common variables:
        xs_values, xs_E, Ein_arno, mu, T_arno = cls.common_variables(xs_0K, Ein,
                                                                     M, T, Eout,
                                                                     theta)
        # Calculate the cross-section matrix:
        model = kwargs.pop("model", "sigma1")
        xs_mat, start = get_input_data(xs_values, xs_E, Ein_arno, mu[0])
        if model == "sigma1":
            update_xs_mat_sigma1(xs_mat, Ein_arno, start, xs_values, xs_E, M,
                                 T_arno)
        elif model == "fgm":
            update_xs_mat_sct(xs_mat, Ein_arno, start, xs_values, xs_E, M,
                              T_arno, mu_fit, T_arno)
        elif model == "sct":
            Teff = cls.get_Teff(pdos, T_arno)
            update_xs_mat_sct(xs_mat, Ein_arno, start, xs_values, xs_E, M,
                              T_arno, mu_fit, Teff)
        elif model == "pdos":
            tau1, DebyeWallerCoeff, delta_beta = cls.get_pdos_variables(pdos, T_arno)
            threshold = kwargs.pop("threshold", 0.0)
            nphonon = kwargs.pop("nphonon", 1000)
            tau_to_file = kwargs.pop("tau_to_file", False)
            binary = kwargs.pop("binary", False)
            if tau_to_file:
                os.makedirs("tau", exist_ok=True)
            if binary:
                os.makedirs("tau/binary", exist_ok=True)
            update_xs_mat_pdos(xs_mat, Ein_arno, start, xs_values, xs_E, M,
                               T_arno, mu_fit, delta_beta, DebyeWallerCoeff,
                               tau1, nphonon, threshold,
                               tau_to_file=tau_to_file, binary=binary)
        else:
            raise ValueError("Model not implemented")
        return cls(xs_0K, Ein, M, T, xs_mat, index=mu, columns=Eout)

    @classmethod
    def from_tau(cls, xs_0K, Ein, M, T, Eout, theta, mu_fit, tau_folder, delta_beta,
                 DebyeWallerCoeff, check=True, key=None):

        xs_values, xs_E, Ein_arno, mu, T_arno = cls.common_variables(xs_0K, Ein,
                                                                     M, T, Eout,
                                                                     theta)
        # Check tau_n files:
        if check:
            tau_n_list = cls.check_data(tau_folder, delta_beta, DebyeWallerCoeff,
                                        T_arno)
        else:
            tau_n_list = cls.check_tau_folder(tau_folder)
        key_ = key if key is not None else "tau"

        # Begin the calculation:
        xs_mat, start = get_input_data(xs_values, xs_E, Ein_arno, mu[0])
        update_xs_mat_pdos(xs_mat, Ein_arno, start, xs_values, xs_E, M,
                           T_arno, mu_fit, delta_beta, DebyeWallerCoeff,
                           tau_n_list, key_)
        return cls(xs_0K, Ein, M, T, xs_mat, index=mu, columns=Eout)

    @staticmethod
    def check_tau_folder(tau_folder):
        tau_n_list = [f"{tau_folder}/{f}" for f in os.listdir(tau_folder) if f.endswith(".h5")]
        if len(tau_n_list) == 0:
            tau_n_text = [f for f in os.listdir(tau_folder) if f.endswith(".txt")]
            if len(tau_n_text) == 0:
                raise ValueError("No tau_n files found")
            else:
                print("tau_n files are in txt format. It will be use hdf5 format")
                for tau_n_text_name in tau_n_text:
                    array = np.loadtxt(tau_folder + "/" + tau_n_text_name)
                    name = f"{tau_folder}/binary/{tau_n_text_name.replace('.txt', '.h5')}"
                    tau_n_list.append(name)
                with h5py.File(name, "w") as f:
                    f.create_dataset("tau", data=array)
        return sorted(tau_n_list, key=extract_number)

    @staticmethod
    def check_data(tau_folder, delta_beta, DebyeWallerCoeff, T_arno):
        tau_n_list = XsMat.check_tau_folder(tau_folder)
        if len(tau_n_list) != len(delta_beta):
            raise ValueError("The number of tau_n files is not equal to the number of delta_beta values")
        if len(tau_n_list) != len(DebyeWallerCoeff):
            raise ValueError("The number of tau_n files is not equal to the number of DebyeWallerCoeff values")
        T_doc = [extract_number(f) for f in tau_n_list]
        if (T_doc == T_arno[1:] if T_arno[0] == 0 else T_doc == T_arno).all():
            return tau_n_list
        else:
            raise ValueError("The tau_n files are not in the correct order or the temperature grid is not correct")

    @staticmethod
    def common_variables(xs_0K, Ein, M, T, Eout, theta):
        mu = np.sort(np.cos(np.deg2rad(theta)))
        T_arno = T * (1 + mu) / 2
        Ein_arno = get_Ein_arno(Ein, Eout, mu, M)
        xs_values, xs_E = xs_0K.values, xs_0K.index.values
        return xs_values, xs_E, Ein_arno, mu, T_arno

    @staticmethod
    def get_pdos_variables(pdos, T_arno):
        """
        Get the tau1, DebyeWallerCoeff and delta_beta variables for the pdos model.
        If Teff can't be calculated, the values are nan, so we replace them with the
        next values.

        Parameters
        ----------
        pdos: 'solid_cinel.core.material.Pdos'
            Pdos object.
        T_arno: np.ndarray, (M,)
            Target arno temperature grid in K

        Returns
        -------
        tau1: np.ndarray, (M, T)
            tau1 values for all the T_arno values
        DebyeWallerCoeff: np.ndarray, (M,)
            DebyeWallerCoeff values for all the T_arno values
        delta_beta: np.ndarray, (M,)
            delta_beta values for all the T_arno values
        """
        # Create variables:
        tau1 = np.zeros((len(T_arno), len(pdos.rho.values)))
        DebyeWallerCoeff = np.zeros(len(T_arno))
        delta_beta = np.zeros(len(T_arno))

        # Fill variables:
        for i in range(len(T_arno)):
            if T_arno[i] > 0.0:
                tau1[i, :] += pdos.get_tau_1(T_arno[i]).values
                DebyeWallerCoeff[i] += pdos.DebyeWallerCoeff(T_arno[i])
                delta_beta[i] += pdos.to_beta_grid(T_arno[i]).grid

        # Some values are nan, so we replace them with the next values:
        nan_indices = np.where(np.isnan(DebyeWallerCoeff))
        if nan_indices[0].size > 0:
            new_value_index = nan_indices[0].max() + 1
            delta_beta[nan_indices] = delta_beta[new_value_index]
            DebyeWallerCoeff[nan_indices] = DebyeWallerCoeff[new_value_index]
            tau1[nan_indices] = tau1[new_value_index]
        return tau1, DebyeWallerCoeff, delta_beta

    @staticmethod
    def get_Teff(pdos, T_arno):
        """
        Get the effective temperature for the sct model. If Teff can't be calculated,
        the values are nan, so we replace them with the next values.

        Parameters
        ----------
        pdos: 'solid_cinel.core.material.Pdos'
            Pdos object.
        T_arno: np.ndarray, (M,)
            Target arno temperature grid in K

        Returns
        -------
        Teff: np.ndarray, (M,)
            Effective temperature values for all the T_arno values
        """
        Teff = np.array(
            [pdos.Teff(T_aprox) if T_aprox > 0.0 else 0 for T_aprox in
             T_arno]
        )
        # Aproximation: if Teff is nan, take the next value of Teff
        nan_indices = np.where(np.isnan(Teff))
        if nan_indices[0].size > 0:
            Teff[nan_indices] = Teff[nan_indices[0].max() + 1]
        return Teff


@nb.jit(nopython=True, nogil=True, cache=True)
def default_Eout(Ein: float) -> np.ndarray:
    """
    Generate the default Eout grid for the convolution. The grid is tested with
    NJOY values to ensure a relative difference smaller than 0.4%

    Parameters
    ----------
    Ein : float
        Incident energy in eV

    Returns
    -------
    Eout : ndarray
        Outgoing energy grid in eV

    Examples
    --------
    Test the default Eout with NJOY values:
    # 0K xs data for U238:
    >>> wd = os.getcwd()
    >>> os.chdir(__file__.replace("xs_mat.py", ""))
    >>> os.chdir("../../data/xs/U238/")
    >>> xs_0K = pd.read_hdf("u238.0.2", key="elastic")
    >>> os.chdir(wd)

    # Generate Broadening test results:
    >>> T = 1000
    >>> Ein = 2.0
    >>> Eout = default_Eout(Ein)
    >>> M = 238.05077040419212
    >>> round(Dxs.from_sigma1(xs_0K, Ein, M, T, Eout).integral, 2)
    9.09
    """
    Eout_small = np.linspace(0,
                             0.99 * Ein,
                             2000)
    Eout_middle = np.linspace(0.99 * Ein,
                              Ein * 1.01,
                              3000)
    if Ein * 2 < 5.0:
        Eout_great = np.logspace(np.log10(Ein * 1.01),
                                 np.log10(5.0),
                                 2000)
    else:
        Eout_great = np.logspace(np.log10(Ein * 1.01),
                                 np.log10(2 * Ein),
                                 2000)
    return np.sort(np.concatenate((Eout_great, Eout_small, Eout_middle)))


@nb.jit("float64[:, :](float64, float64[:], float64[:], float64)",
    nopython=True, nogil=True, cache=True)
def get_Ein_arno(Ein: float, Eout: np.ndarray, mu: np.ndarray,
                 M: float) -> np.ndarray:
    """
    Get the incident energy matrix for the arno model.

    Parameters
    ----------
    Ein: float
        The incident energy of the neutron in eV
    Eout: np.ndarray, (Z,)
        The neutron outgoing energy grid in eV
    mu: np.ndarray, (M,)
        The neutron outgoing angle grid in degrees (0, 180]
    M: float
        Mass of the material in amu

    Returns
    -------
    Ein_arno: np.ndarray, (M, Z)
        Incident energy matrix for the arno model
    """
    Ein_arno = np.empty((len(mu), len(Eout)))
    for i in range(len(mu)):
        alpha = (Ein + Eout - 2 * mu[i] * np.sqrt(Ein * Eout)) * m / M
        Ein_arno[i, :] = (Eout + Ein) / 2 - Ein * mu[i] * m / M
        Ein_arno[i, :] += 0.5 * alpha / (1 - mu[i])
    return Ein_arno


@nb.jit("Tuple((float64[:, :], int8))(float64[:], float64[:], float64[:, :], float64)", nopython=True)
def get_input_data(xs_values: np.ndarray, xs_E: np.ndarray,
                   Ein_arno: np.ndarray, mu_min: float) -> (np.ndarray, int):
    """
    Get the input data for the convolution for the xs matrix calculation.

    Parameters
    ----------
    xs_values: np.ndarray, (N,)
        Cross section values in barns at 0K
    xs_E: np.ndarray, (N,)
        Cross section energy grid in eV
    Ein_arno: np.ndarray, (M, N)
        Incident energy matrix for the arno model
    mu_min: float
        The minimum cosine. If mu_min == -1.0, the first row of the matrix is
        filled with the cross section values at 0K.

    Returns
    -------
    xs_mat: np.empty, (M, N)
        Cross section matrix in barns
    start: int
        Start index for the loop in theta. If mu_min == -1.0, start = 1, else
        start = 0
    """
    xs_mat = np.zeros(Ein_arno.shape)
    if mu_min == np.cos(pi):
        xs_mat[0, :] += np.interp(Ein_arno[0, :], xs_E, xs_values)
        start = 1
    else:
        start = 0
    return xs_mat, start


@nb.jit(nopython=True, nogil=True, cache=True)
def Db(xs_values, xs_E, Ein, Eout, pdf):
    """
    Calculate the doppler broadening of a cross section for a pdf

    Parameters
    ----------
    xs_values: np.ndarray, (N,)
        Cross section values in barns
    xs_E: np.ndarray, (N,)
        Cross section energy grid in eV
    Ein: float
        The incident energy of the neutron in eV
    Eout: np.ndarray, (Z,)
        The neutron outgoing energy grid in eV
    pdf: np.ndarray, (Z,)
        Probability density function

    Returns
    -------
    Db_xs: float
        Doppler broadened cross section in barns
    """
    max_pos = np.argmax(pdf)
    if pdf[max_pos] > 1.0e308:  # Overflow found in pdf_val
        Db_xs = np.interp(Eout[max_pos], xs_E, xs_values)
    else:
        norm = np.trapz(pdf, x=Eout)
        # Recoil:
        recoil = Ein - Eout[max_pos]
        # xs:
        xs_Eout_arno = np.interp(Eout + recoil, xs_E, xs_values)
        Db_xs = np.trapz(xs_Eout_arno * pdf, x=Eout) / norm
    return Db_xs


def update_xs_mat_pdos(xs_mat: np.ndarray, Ein_arno: np.ndarray, start: int,
                       xs_values: np.ndarray, xs_E: np.ndarray,
                       M: float, T_arno: np.ndarray,
                       mu_fit: float, delta_beta: np.ndarray,
                       DebyeWallerCoeff: np.ndarray, *args,
                       tau_to_file = False, binary= False):
    """
    Calculate the cross section matrix for a given incident energy, target mass,
    target temperature, outgoing energy grid and outgoing angle grid using arno
    model with different pdf.

    Parameters
    ----------
    xs_values: np.ndarray, (N,)
        Cross section values in barns
    xs_E: np.ndarray, (N,)
        Cross section energy grid in eV
    Ein: float
        The incident energy of the neutron in eV
    M: float
        Mass of the material in amu
    T_arno: np.ndarray, (M,)
        Target temperature grid in K
    Eout: np.ndarray, (Z,)
        The neutron outgoing energy grid in eV
    mu: np.ndarray, (M,)
        The neutron outgoing angle grid in degrees (0, 180]
    mu_fit: float
        The cosine of the outgoing angle to fit the S(alpha, -beta) distribution
        with sigma1
    nphonon: int
        Phonon expansion order
    tau1: np.ndarray, (M, T)
        tau1 values for all the T_arno values
    delta_beta: np.ndarray, (M,)
        delta_beta values for all the T_arno values
    threshold: float
        Minimun value to take into account in the creation of tau_n
    DebyeWallerCoeff: np.ndarray, (M,)
        DebyeWallerCoeff values for all the T_arno values

    Returns
    -------
    np.ndarray, (M, N)
        Cross section matrix in barns
    """
    def gen_xs_mat_mu(i, tau1, nphonon, threshold):
        tau_n = tau_n_functions(tau1[i], delta_beta[i], nphonon, threshold)
        save_data(tau_n, nphonon, T_arno[i], tau_to_file, binary)
        return dask.delayed(update_xs_mat_pdos_row)(xs_mat, i, tau_n,
                                                    delta_beta[i], DebyeWallerCoeff[i],
                                                    Ein_arno[i], T_arno[i],
                                                    xs_values, xs_E, mu_fit, M)

    def xs_mat_mu_from_tau(i, tau_n_list, key):
        i_ = i - start
        tau_n = h5py.File(tau_n_list[i_], "r")[key][:]
        return dask.delayed(update_xs_mat_pdos_row)(xs_mat, i, tau_n,
                                                    delta_beta[i_], DebyeWallerCoeff[i_],
                                                    Ein_arno[i], T_arno[i],
                                                    xs_values, xs_E, mu_fit, M)
    if len(args) == 1:
        calculation = xs_mat_mu_from_tau
    else:
        calculation = gen_xs_mat_mu
    delayed_tasks = [calculation(i, *args) for i in range(start, xs_mat.shape[0])]
    dask.compute(*delayed_tasks)


@nb.jit(nopython=True, nogil=True, parallel=True)
def update_xs_mat_pdos_row(xs_mat, i, tau_n, delta_beta, debyewallercoeff,
                           Ein, T, xs_values, xs_E, mu_fit, M):
    tau_n_beta = np.arange(tau_n.shape[1]) * delta_beta
    for j in prange(xs_mat.shape[1]):
        Eout_db = default_Eout(Ein[j])
        pdf = get_ScatFunc_pdos_angle(Ein[j], M, T,
                                      Eout_db, mu_fit, tau_n, tau_n_beta,
                                      debyewallercoeff)
        xs_mat[i, j] += Db(xs_values, xs_E, Ein[j], Eout_db, pdf)


def save_data(tau_n, nphonon, T, tau_to_file, binary):
    name = f"tau_{nphonon}_{T}"
    if tau_to_file:
        np.savetxt(f"tau/{name}.txt", tau_n, delimiter="\t", fmt="%.14f")
    if binary:
        with h5py.File(f"tau/binary/{name}.h5", "w") as f:
            f.create_dataset("tau", data=tau_n)


@nb.jit("void(float64[:, :], float64[:, :], int8, float64[:], float64[:], float64, float64[:])",
    nopython=True, nogil=True, cache=True, parallel=True)
def update_xs_mat_sigma1(xs_mat: np.ndarray, Ein_arno: np.ndarray, start: int,
                         xs_values: np.ndarray, xs_E: np.ndarray,
                         M: float, T_arno: np.ndarray) -> np.ndarray:
    """
    Calculate the cross section matrix for a given incident energy, target mass,
    target temperature, outgoing energy grid and outgoing angle grid using arno
    model with the pdf calculated with SIGMA1 algorithm.

    Parameters
    ----------
    xs_mat: np.ndarray, (M, N)
        Cross section matrix in barns
    Ein_arno: np.ndarray, (M, N)
        Incident energy matrix for the arno model
    start: int
        Start index for the loop in theta. If mu_min == -1.0, start = 1, else
        start = 0
    xs_values: np.ndarray, (N,)
        Cross section values in barns at 0K
    xs_E: np.ndarray, (N,)
        Cross section energy grid in eV
    M: float
        Mass of the material in amu
    T_arno: np.ndarray, (M,)
        Target temperature grid in K

    Returns
    -------
    np.ndarray, (M, N)
        Cross section matrix in barns
    """
    for i in range(start, Ein_arno.shape[0], 1):
        for j in prange(Ein_arno.shape[1]):
            Eout_db = default_Eout(Ein_arno[i, j])
            pdf = sigma1(Eout_db, Ein_arno[i, j], T_arno[i], M)
            xs_mat[i, j] += Db(xs_values, xs_E, Ein_arno[i, j], Eout_db,
                                pdf)

@nb.jit("void(float64[:, :], float64[:, :], int8, float64[:], float64[:], float64, float64[:], float64, float64[:])",
    nopython=True, nogil=True, cache=True, parallel=True)
def update_xs_mat_sct(xs_mat: np.ndarray, Ein_arno: np.ndarray, start: int,
                      xs_values: np.ndarray, xs_E: np.ndarray,
                      M: float, Tarno: np.ndarray, mu_fit: float,
                      Tarno_eff: np.ndarray) -> np.ndarray:
    """
    Calculate the cross section matrix for a given incident energy, target mass,
    target temperature, outgoing energy grid and outgoing angle grid using arno
    model with the pdf calculated with S(alpha, -beta) algorithm of sct.

    Parameters
    ----------
    xs_mat: np.ndarray, (M, N)
        Cross section matrix in barns
    Ein_arno: np.ndarray, (M, N)
        Incident energy matrix for the arno model
    start: int
        Start index for the loop in theta. If mu_min == -1.0, start = 1, else
        start = 0
    xs_values: np.ndarray, (N,)
        Cross section values in barns at 0K
    xs_E: np.ndarray, (N,)
        Cross section energy grid in eV
    M: float
        Mass of the material in amu
    T_arno: np.ndarray, (M,)
        Target temperature grid in K
    mu_fit: float
        The cosine of the outgoing angle to fit the S(alpha, -beta) distribution
        with SIGMA1 algoritm
    Tarno_eff: np.ndarray, (M,)
        Effective temperature grid in K

    Returns
    -------
    np.ndarray, (M, N)
        Cross section matrix in barns
    """
    for i in range(start, Ein_arno.shape[0], 1):
        for j in prange(Ein_arno.shape[1]):
            Eout_db = default_Eout(Ein_arno[i, j])
            pdf = get_scat_sct_angular(Eout_db, mu_fit, Ein_arno[i, j],
                                       Tarno[i], M, Tarno_eff[i], 1.0)
            xs_mat[i, j] += Db(xs_values, xs_E, Ein_arno[i, j], Eout_db, pdf)


def extract_number(s):
    return float(re.findall("\d+\.\d+", s)[-1])
