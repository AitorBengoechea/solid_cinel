import argparse
import numpy as np
import pandas as pd
import numba as nb
from numba import prange
from scipy.constants import physical_constants as const

# Import from solid_cinel:
from solid_cinel import Pdos, AlphaBase, Beta, Xs0K, Sab
from solid_cinel.core.generic import integrate, reshape_differential
from solid_cinel import default_Eout, get_ScatFuncClm, get_tauNbeta

# Import from application:
from solid_cinel.application.pdosApp import get_Pdos


kb = const["Boltzmann constant in eV/K"][0]
m = const["neutron mass in u"][0]


def add_XsArgs(parser: argparse.ArgumentParser):
    """
    Add arguments to the parser for the calculation of double differential
    scattering cross section.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to which the arguments should be added.
    """
    parser.add_argument('model', type=str.lower,
                        choices=['sta', 'alpha0'],
                        help='Model to use for the calculation of the algorithm')
    parser.add_argument('xs0K', type=str,
                        help='Cross section at 0 K')
    parser.add_argument('Ein', type=str,
                        help='Incident energy in eV')
    parser.add_argument('M', type=float,
                        help='Mass of the target atom in a.m.u.')
    parser.add_argument('T', type=float,
                        help='Temperature in K')
    parser.add_argument('--nphonon', type=int,
                        default=None,
                        help='Number of phonon')
    parser.add_argument('--theta', type=str,
                        default=None,
                        help='Grid for the scattering angle in degrees')
    parser.add_argument('--p0', type=bool, default=True,
                        help='Add p0 to the calculation. Default is True.')

def calc_alpha0(xs0K: pd.Series, Ein: np.ndarray, M: float, T: float,
                pdos: Pdos, nphonon: int = None, p0: bool = True) -> np.ndarray:
    """
    Calculate the XS using alpha0 model asymtotic value.

    Parameters
    ----------
    xs0K : pd.Series
        The cross section at 0 K.
    Ein : np.ndarray
        The incident energy in eV.
    M : float
        The mass of the target atom in a.m.u.
    T : float
        The temperature in K.
    pdos :  Pdos
        The phonon density of states object.
    nphonon : int, optional
        The number of phonon modes to consider. If None, it will be calculated
        based on the maximum alpha value. Default is None.
    p0 : bool, optional
        Whether to add p0 to the calculation. Default is True.

    Returns
    -------
    np.ndarray
        The Xs values.
    """
    # Get the temperature dependent pdos and the Debye Waller coefficient:
    pdos_ = pdos.get_Tpdos(T)
    DebyeWallerCoeff = pdos_.DebyeWallerCoeff

    # Get the Expansion order:
    if nphonon is None:
        alphaMax = m / (m + M) * (Ein[-1]) / (kb * T)
        nphonon = AlphaBase(alphaMax).expansionOrder(DebyeWallerCoeff, 1.0e-6, 7000)
        print("Number of phonons:", nphonon)

    # Calculate the recoil and the alpha values:
    recoil = m / (m + M) * (Ein)
    alpha = recoil / (kb * T)

    # Get the maximun order get the tauN functions:
    tauNdf = pdos_.tauN(nphonon, 0.0)
    tauN, tauNbeta = tauNdf.values, tauNdf.columns.values

    # Get the scattering function and calculate the cross section:
    beta = Beta.from_default(T).data
    scatfunc = Sab.from_tau(alpha, beta, tauN, tauNbeta, DebyeWallerCoeff).full

    # Get the outgoing energy grid:
    EinCalc = Ein + recoil
    EoutCalc = scatfunc.columns.values * kb * T + EinCalc[::, np.newaxis]

    # Integrate the cross section:
    xsDb = (scatfunc * reshape_differential(xs0K, EoutCalc)).apply(integrate,
                                                                   axis=1)

    # Add p0 to the calculation:
    if p0:
        xsDb += np.exp(- DebyeWallerCoeff * alpha) * reshape_differential(xs0K, EinCalc)
    return xsDb.values


@nb.jit(nopython=True, cache=True)
def get_recoil(Eout, Ein, M, mu):
    """
    Calculate the recoil energy for the given outgoing and incoming energies.

    Parameters
    ----------
    Eout : np.ndarray
        The outgoing energy grid.
    Ein : float
        The incoming energy.
    M : float
        The mass of the target atom in a.m.u.
    mu : np.ndarray
        The chemical potential values.

    Returns
    -------
    np.ndarray
        The recoil energy.
    """
    return (Eout + Ein - 2 * mu[::, np.newaxis] * np.sqrt(Eout * Ein)) / (M / m)


@nb.jit(nopython=True, cache=True)
def get_alphaMat(Eout, Ein, T, M, mu):
    """
    Calculate the alpha matrix for the given outgoing and incoming energies.

    Parameters
    ----------
    Eout : np.ndarray
        The outgoing energy grid.
    Ein : float
        The incoming energy.
    T : float
        The temperature in K.
    M : float
        The mass of the target atom in a.m.u.
    mu_ : np.ndarray
        The chemical potential values.

    Returns
    -------
    np.ndarray
        The alpha matrix.
    """
    return get_recoil(Eout, Ein, M, mu) / (kb * T)


@nb.jit(nopython=True, parallel=True, nogil=True, cache=True)
def EinLoopOpt(EinGrid: np.ndarray, M: float, T: float, mu: np.ndarray,
               tauNcut: np.ndarray, beta: np.ndarray,
               DebyeWallerCoeff: float, xsMatrix: np.ndarray,
               xsMatrixEin: np.ndarray) -> np.ndarray:
    """
    Calculate the scattering Xs using the sta method.

    Parameters
    ----------
    EinGrid : np.ndarray, (N,)
        The incident energy grid in eV.
    M : float
        The mass of the target atom in a.m.u.
    T : float
        The temperature in K.
    mu : np.ndarray, (Nmu,)
        The chemical potential values.
    tauNcut : np.ndarray, (Z, Nbeta)
        The tauN functions cut to the beta grid.
    beta : np.ndarray, (Nbeta,)
        The beta grid for the tauN functions.
    DebyeWallerCoeff : float
        The Debye Waller coefficient.
    xsMatrix : np.ndarray, (Nmu, NEin_0K)
        The cross section matrix for the given mu values and incoming energies.
    xsMatrixEin : np.ndarray, (NEin_0K,)
        The incoming energy grid for the cross section matrix.

    Returns
    -------
    np.ndarray, (N,)
        The Xs for the given incident energy grid.
    """
    Nmu = len(mu)
    resultCLM = np.zeros((len(EinGrid)), dtype=np.float64)
    for i in prange(len(EinGrid)):
        # Get the outgoing energy
        if EinGrid[i] >= 10.0:
            # If the incoming energy is greater than 10 eV, use a linspace around it
            Eout = np.linspace(EinGrid[i] * 0.9, EinGrid[i] * 1.1, 3000)
        else:
            Eout = default_Eout(EinGrid[i])

        # Get the phonon dynamics for CLM:
        PhononDyn = get_ScatFuncClm(EinGrid[i], M, T, Eout, mu, tauNcut, beta,
                                    DebyeWallerCoeff, 0.0)

        # Get the interaction energy matrix:
        EinMat = 0.5 * (EinGrid[i] + Eout + get_recoil(Eout, EinGrid[i], M, mu))

        # Interpolate the db data to the interaction energy matrix:
        xsInterp = np.zeros(EinMat.shape)
        for j in prange(Nmu):
            xsInterp[j, :] = np.interp(EinMat[j, ::], xsMatrixEin, xsMatrix[j])

        # Integrate:
        resultCLM[i] = np.trapz(np.trapz(PhononDyn * xsInterp, x=Eout), x=mu)
    return resultCLM

@nb.jit(nopython=True, cache=True)
def calc_sta_p0(EinGrid: np.ndarray, M: float, mu: np.ndarray, T: float,
                DebyeWallerCoeff: float, xsMatrix: np.ndarray,
                xsMatrixEin: np.ndarray):
    """
    Calculate p0 contribution for the scattering cross section using the
    sta method.

    Parameters
    ----------
    EinGrid : np.ndarray, (N,)
        The incident energy grid in eV.
    M : float
        The mass of the target atom in a.m.u.
    mu : np.ndarray, (Nmu,)
        The chemical potential values.
    T : float
        The temperature in K.
    DebyeWallerCoeff : float
        The Debye Waller coefficient.
    xsMatrix : np.ndarray, (Nmu, NEin_0K)
        The cross section matrix for the given mu values and incoming energies.
    xsMatrixEin : np.ndarray, (NEin_0K,)
        The incoming energy grid for the cross section matrix.

    Returns
    -------
    np.ndarray, (N,)
        The p0 term for the given incident energy grid.
    """
    A = m / (m + M)

    # Calculate the interaction energy matrix:
    EinMat = EinGrid * (A + 1 - mu[::, np.newaxis]) / A

    # Calculate the ALPHA matrix:
    alpha = 2 * EinGrid * (1 - mu[::, np.newaxis]) / (A * kb * T)

    # Initialize the xsInterp matrix:
    xsInterp = np.zeros(EinMat.shape)
    for j in prange(len(mu)):
        # Interpolate the xsMatrix to the EinMat:
        xsInterp[j, :] = np.interp(EinMat[j, ::], xsMatrixEin, xsMatrix[j])

    # Calculate the phonon dynamics for CLM:
    XsMat_mu = 0.5 * np.exp(- DebyeWallerCoeff * alpha) * xsInterp

    return np.trapz(XsMat_mu.T, x=mu)

def calc_sta(xs0K: Xs0K, EinGrid: np.ndarray, M: float, T: float,
             pdos: Pdos, nphonon: int = None, theta: bool = None,
             p0: bool = True) -> np.ndarray:
    """
    Calculate the double differential scattering cross section using the
    sta method.
    Parameters
    ----------
    xs0K : Xs0K
        The cross section at 0 K.
    EinGrid : np.ndarray
        The incident energy grid in eV.
    M : float
        The mass of the target atom in a.m.u.
    T : float
        The temperature in K.
    pdos : Pdos
        The phonon density of states object.
    nphonon : int, optional
        The number of phonon modes to consider. If None, it will be calculated
        based on the maximum alpha value. Default is None.
    theta : str or None, optional
        The path to the theta grid file. If None, a default grid will be used.
    p0 : bool, optional
        Whether to add p0 to the calculation. Default is True.

    Returns
    -------
    np.ndarray
        Xs values.
    """
    if theta is None:
        # Default theta grid if not provided
        theta = np.arange(1, 181, 1)[::-1]
    else:
        theta = np.loadtxt(theta)
    mu = np.cos(np.deg2rad(theta))


    # Get the temperature dependent pdos and the Debye Waller coefficient:
    pdos_ = pdos.get_Tpdos(T)
    DebyeWallerCoeff = pdos_.DebyeWallerCoeff
    Teff = pdos_.Teff

    # Change the index to mu values instead of Temperatures:
    Tarno = Teff * (1 + mu) / 2
    EinGrid_0K = xs0K.EinGrid[xs0K.EinGrid <= 2 * EinGrid[-1]]
    xsData = xs0K.sigma1(Tarno, EinGrid_0K, values=True)

    # Get the Expansion order:
    if nphonon is None:
        EoutMax = default_Eout(EinGrid[-1])
        alphaMat = get_alphaMat(EoutMax, EinGrid[-1], T, M, mu).max()
        nphonon = AlphaBase(alphaMat).expansionOrder(DebyeWallerCoeff,1.0e-6, 8000)
        print("Number of phonons:", nphonon)

    # Calculate the necessary phonon expansion and the tau N functions:
    tauN = pdos_.tauN(nphonon, 0.0, values=True)
    tauNbeta = get_tauNbeta(pdos_.beta.data, tauN.shape[1])

    # Cut the tauN functions to the defaults beta grid:
    betaMax = EinGrid[-1] / (kb * T) * 1.1
    betaMask = tauNbeta <= betaMax
    tauNcut = tauN[::, betaMask]
    beta = tauNbeta[betaMask]

    xsDb = EinLoopOpt(EinGrid, M, T, mu, tauNcut, beta, DebyeWallerCoeff,
                      xsData, EinGrid_0K)

    # If p0 is True, add the p0 contribution:
    if p0:
        xsDb += calc_sta_p0(EinGrid, M, mu, T, DebyeWallerCoeff,
                            xsData, EinGrid_0K)

    # Save the result:
    return xsDb


def get_Xs(args: argparse.Namespace) -> np.ndarray:
    """
    Handle the arguments for the calculation of the double differential scattering
    cross section.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.ndarray
        The Xs values.
    """
    # Get the extra arguments for Pdos
    argsPdos = get_Pdos(args)

    # Initialize the Xs class with 0K cross section data
    xs = Xs0K.from_file(args.xs0K, args.M)

    Ein = np.loadtxt(args.Ein)

    if args.model == "sta":
        return calc_sta(xs, Ein, args.M, args.T, argsPdos,
                        nphonon=args.nphonon, theta=args.theta, p0=args.p0)
    elif args.model == "alpha0":
        # Calculate the Xs using alpha0 model asymtotic value:
        return calc_alpha0(xs.data, Ein, args.M, args.T, argsPdos,
                           nphonon=args.nphonon, p0=args.p0)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def handle_XsArgs(args: argparse.Namespace) -> dict:
    """
    Handle the arguments for the calculation of the double differential scattering
    cross section.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed arguments.

    Returns
    -------
    np.ndarray
        An array containing the double differential scattering cross section
        values.
    """
    # Return the results as a dictionary:
    return {"values": get_Xs(args)}
