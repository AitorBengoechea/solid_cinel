import numpy as np
cimport numpy as np
from scipy.constants import physical_constants as const
from libc.math cimport sqrt, exp, pi, fmax

cdef double m = const["neutron mass in u"][0]  # Define the value of m
cdef double kb = const["Boltzmann constant in eV/K"][0]  # Define the value of kb

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef double[:] sigma1(np.ndarray[double, ndim=1] Eout, double Ein, double T, double M):
    """
    Sigma1 function for Energy differential scattering function
    ..math::
           S(E, E^\prime, M, T) = \frac{1}{2}\sqrt{\frac{M}{m\pi k_BT}}\frac{\sqrt{E^\prime}}{E}\left(exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} - \sqrt{E^\prime}\right)^2 \right) - exp\left(\frac{-M}{m k_B T}\left(\sqrt{E} + \sqrt{E^\prime}\right)^2 \right)\right)

    Parameters
    ----------
    Eout : np.array
        Outgoing energy grid in eV
    Ein : float
        Incoming energy in eV
    T : float
        Temperature in K
    M :
        Mass of the target in amu

    Returns
    -------
    scattfunc : np.array
        Scattering function based on sigma1 model
    """
    cdef:
        int n = Eout.shape[0]
        np.ndarray[double, ndim=1] exp_negative = np.exp(- M / (m * kb * T) * (sqrt(Ein) - np.sqrt(Eout)) ** 2)
        np.ndarray[double, ndim=1] exp_positive = np.exp(- M / (m * kb * T) * (sqrt(Ein) + np.sqrt(Eout)) ** 2)
        np.ndarray[double, ndim=1] scattfunc = 0.5 * (exp_negative - exp_positive) * np.sqrt(Eout) / Ein
    scattfunc *= sqrt(M / (pi * m * kb * T))
    return scattfunc

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef double[:] get_scat_sct_angular(np.ndarray[double, ndim=1] Eout, double mu, double Ein, double T, double M, double Teff, double ws):
    """
    Calculate the scattering function from the Short Collision Time model using
    a single angle.
    ..math::
        S(\theta, E^\prime, E, M, T) = \frac{1}{2 * k_B * T}\sqrt{\frac{E^\prime}{E}} \frac{1}{\sqrt{4 \pi w_s \alpha T_{eff} / T}} exp\left(\frac{(w_s\alpha +\beta)^2}{4 \alpha w_s T_{eff}/T}\right)

    Parameters
    ----------
    Eout : np.ndarray, (N,)
        The neutron outgoing energy grid in eV
    mu : float
        Cosine of the angle between the incident neutron direction and
        the outgoing neutron direction
    Ein : float
        The incident energy of the neutron in eV
    T : float
        Temperature of the material in K
    M : float
        The mass of the target material in amu
    Teff : float
        Effective temperature of the material in K
    ws : float
        Normalization for continuous (vibrational) part. For solid is 1.

    Returns
    -------
    np.array, (N,)
        The scattering function values for a single angle
    """
    cdef:
        double awr = ((M / m + 1) / (M / m)) ** 2
        np.ndarray[double, ndim=1] beta = (Eout - Ein) / (kb * T)
        np.ndarray[double, ndim=1] alpha = Eout + Ein - 2 * mu * np.sqrt(Eout * Ein) / (M * kb * T / m)
        np.ndarray[double, ndim=1] scattfunc = np.exp(-(ws * alpha + beta) ** 2 / (4 * alpha * Teff / T * ws))

    scattfunc /= np.sqrt(4 * pi * ws * alpha * Teff / T)
    scattfunc *= awr * np.sqrt(Eout / Ein) / (2 * kb * T)
    return scattfunc


#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
cpdef np.ndarray[double, ndim=1] default_Eout(double Ein):
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
    """
    cdef:
        np.ndarray[double, ndim=1] Eout_small = np.linspace(0, 0.99 * Ein, 2000)
        np.ndarray[double, ndim=1] Eout_middle = np.linspace(0.99 * Ein, Ein * 1.01, 3000)
        np.ndarray[double, ndim=1] Eout_great

    if Ein * 2 < 5.0:
        Eout_great = np.logspace(np.log10(Ein * 1.01), np.log10(5.0), 2000)
    else:
        Eout_great = np.logspace(np.log10(Ein * 1.01), np.log10(2 * Ein), 2000)

    cdef np.ndarray[double, ndim=1] result = np.sort(np.concatenate((Eout_great, Eout_small, Eout_middle)))
    return result


cpdef double Db(np.ndarray[double, ndim=1] xs_values, np.ndarray[double, ndim=1] xs_E, double Ein, np.ndarray[double, ndim=1] Eout, np.ndarray[double, ndim=1] pdf):
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
    cdef:
        int max_pos
        double norm, recoil, Db_xs

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