
import numpy as np
import numba as nb
import h5py
import os
from numba import cuda
from solid_cinel.core.scattering_function.beta import Beta
gpu_available = True if cuda.is_available() else False


@nb.jit(nopython=True)
def first_all_zero_column(tauN, threshold):
    for i in range(tauN.shape[1]):
        if np.all(tauN[:, i] <= threshold):
            return i
    return -1


def optional_jit(func):
    """
    Decorator to use numba.jit or cuda.jit depending on the gpu availability.

    Parameters
    ----------
    func: function
        Function to decorate.

    Returns
    -------
    function
        Decorated function.
    """
    if gpu_available:
        return cuda.jit(func)
    else:
        return nb.jit(func, nopython=True, nogil=True, cache=True, parallel=True)


@optional_jit
def tauNconvol(expBeta: np.ndarray, deltaBeta: np.ndarray, tau1: np.ndarray,
               Ntau1: int, tauNminus1: np.ndarray, NtauNminus1: int, i: int):
    """
    Calculate the convolution in the tauN[i] between  tau1 and tauNminus1.

    Parameters
    ----------
    expBeta: np.ndarray, (N,)
        Minus beta exponential.
    deltaBeta: np.ndarray, (N,)
        beta interval between two consecutive values.
    tau1: np.ndarray, (N,)
        Tau(-beta) function values for n = 1 expansion.
    Ntau1: int
        Length of tau1.
    tauNminus1: np.ndarray, (M,)
        Tau(-beta) function values for n-1 expansion.
    NtauNminus1: int
        Length of tauNminus1.
    i: int
        Index of the tauN[i] value to calculate.

    Returns
    -------
    convol: float
        Convolution value.
    """
    trapz_limit = Ntau1 - 1
    convol = 0.
    for j in range(1, Ntau1):
        convol_j = 0.

        k = i - j  # tauNminus1(-(beta-beta^prime))
        if abs(k) < NtauNminus1:
            if k >= 0:
                convol_j += tauNminus1[k]
            else:
                convol_j += tauNminus1[-k] * expBeta[-k]

        l = i + j  # tauNminus1(-(beta+beta^prime))
        if l < NtauNminus1:
            convol_j += tauNminus1[l] * expBeta[j]

        if j == trapz_limit:
            convol_j *= 0.5  # trapz integrate

        convol += tau1[j] * deltaBeta[j] * convol_j
    return convol
    


@optional_jit
def calc_tauN(expBeta: np.ndarray, deltaBeta: np.ndarray, tau1: np.ndarray,
              Ntau1: int, tauNminus1: np.ndarray, tauN: np.ndarray,
              start: int, NtauN: int, stride: int):
    """
    Calculate the tauN(-beta) function values for all n. The values are
    calculated until the last value of tauN is zero.

    Parameters
    ----------
    expBeta: np.ndarray, (N,)
        Minus beta exponential.
    deltaBeta: np.ndarray, (N,)
        beta interval between two consecutive values.
    tau1: np.ndarray, (N,)
        Tau(-beta) function values for n = 1 expansion.
    tauNminus1: np.ndarray, (M,)
        Tau(-beta) function values for n-1 expansion.
    tauN: np.ndarray, (M,)
        Tau(-beta) function values for n expansion.
    start:  int
        position to start the calculation
    final: int
        position to end the calculation
    stride: int
        step of the calculation

    Returns
    -------
    tauN: np.ndarray, (M,)
        Tau(-beta) function values for n expansion.
    """
    NtauNminus1 = NtauN - Ntau1 + 1
    # tauN(-beta) loop:
    for i in range(start, NtauN, stride):
        # 1 iteration: j = 0
        tauN[i] += tau1[0] * deltaBeta[0] * tauNminus1[i] if i < NtauNminus1 else 0.
        # rest of iterations:
        tauN[i] += tauNconvol(expBeta, deltaBeta, tau1, Ntau1, tauNminus1, NtauNminus1,  i)


@cuda.jit
def tauN_calculation_threads(expBeta, deltaBeta, tau1, Ntau1, tauNminus1, tauNdevice,
                             NtauN):
    """
    Calculate the tauN(-beta) function values for all n. The values are
    calculated until the last value of tauN is zero.

    Parameters
    ----------
    expBeta: np.ndarray
        Minus exponential of beta grid.
    deltaBeta: np.ndarray
        beta interval between two consecutive values.
    tau1: np.ndarray, (N,)
        Tau(-beta) function values for n = 1 expansion.
    tauNminus1: np.ndarray, (M,)
        Tau(-beta) function values for n-1 expansion.
    tauNdevice: np.ndarray, (M,)
        Tau(-beta) function values for n expansion.
    final: int
        position to end the calculation

    Returns
    -------
    tauNdevice: np.ndarray, (M,)
        Tau(-beta) function values for n expansion.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    calc_tauN(expBeta, deltaBeta, tau1, Ntau1, tauNminus1, tauNdevice,
              start, NtauN, stride)


@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def calc_tauNfunc_cpu(tauNfunc: np.ndarray, tau1: np.ndarray,
                      Ntau1: int, beta: np.ndarray,
                      deltaBeta: np.ndarray, nphonon: int, NtauN: int,):
    """
    Get the tau_{n}(-beta) function values for all n.

    Parameters
    ----------
    tauNfunc: 'np.ndarray', (nphonon, N * nphonon)
        All Tau(-beta) function values for n = 1 expansion.
    tau1: 'np.ndarray', (N,)
        Tau(-beta) function values for n = 1 expansion.
    Ntau1: int
        Length of tau1.
    beta: 'np.ndarray', (N,)
        Beta array of tau1 function.
    deltaBeta: np.ndarray
        Interval of beta for the PDOS.
    nphonon: 'int'
        Number of phonon to calculate the tau functions.
    NtauN: 'int'
        Length of the tau_2 function: 2 * N - 1.
    threshold: 'float'
        Minimun value to take into account.

    Returns
    -------
    tauNfunc: 'np.ndarray', (N * nphonon, nphonon)
        All Tau(-beta) function values for n expansion.
    """
    tauNminus1 = tau1.copy()
    expBeta = np.exp(- beta)
    for n in range(1, nphonon):
        tauN = np.zeros(NtauN)
        calc_tauN(expBeta, deltaBeta, tau1, Ntau1, tauNminus1, tauN,
                  0, NtauN, 1)
        # Copy thet data into the array:
        tauNfunc[n, :NtauN] += tauN
        # If the last N values are zero, the next tauN will have the same length
        # because the convolution will be zero for the following values
        NtauN = NtauN if np.all(tauN[-Ntau1:] == 0.0) else NtauN + Ntau1 - 1
        # Next tauN:
        tauNminus1 = tauN

    # Erase the zeros in the last part of the array
    return tauNfunc[::, :NtauN]


def calc_tauNfunc_gpu(tauNfunc: np.ndarray, tau1: np.ndarray,
                      Ntau1: int, beta: np.ndarray,
                      deltaBeta: np.ndarray, nphonon: int, NtauN: int,
                      threadsperblock: int = 128) -> np.ndarray:

    """
    Get the tau_{n}(-beta) function values for all n.

    Parameters
    ----------
    tauNfunc: 'np.ndarray', (nphonon, N * nphonon)
        All Tau(-beta) function values for n = 1 expansion.
    tau1: 'np.ndarray', (N,)
        Tau(-beta) function values for n = 1 expansion.
    Ntau1: int
        Length of tau1.
    beta: 'np.ndarray', (N,)
        Beta array of tau1 function.
    deltaBeta: np.ndarray
        Interval of beta for the PDOS.
    nphonon: 'int'
        Number of phonon to calculate the tau functions.
    NtauN: 'int'
        Length of the tau_2 function: 2 * N - 1.
    threshold: 'float'
        Minimun value to take into account.
    threadsperblock: 'int'
         How many parallel threads are grouped into a single block. The default
          is 128.

    Returns
    -------
    tauNfunc: 'np.ndarray', (N * nphonon, nphonon)
        All Tau(-beta) function values for n expansion.
    """
    # Copy the data to the device
    tau1 = cuda.to_device(tau1)
    tauNminus1 = cuda.to_device(tau1)
    expBeta = cuda.to_device(np.exp(- beta))
    deltaBeta = cuda.to_device(deltaBeta)
    for n in range(1, nphonon):
        # Perform the calculation on the device:
        tauN = cuda.to_device(np.zeros(NtauN))
        blockspergrid = NtauN + threadsperblock - 1
        blockspergrid //= threadsperblock
        tauN_calculation_threads[blockspergrid, threadsperblock](expBeta, deltaBeta, tau1, Ntau1, tauNminus1,
                                                                 tauN, NtauN)

        # Copy the data back to the host
        tauNfunc[n, :NtauN] += tauN.copy_to_host()

        # If the last N values are zero, the next tauN will have the same length
        # because the convolution will be zero for the following values
        NtauN = NtauN if np.all(tauN[-Ntau1:] == 0.0) else NtauN + Ntau1 - 1
        # Next tauN:
        tauNminus1 = tauN
    return tauNfunc[::, :NtauN]


def get_tauNfunc(tau1: np.ndarray, beta: np.ndarray,
                 nphonon: int, threshold: float) -> np.ndarray:
    """
    Get the tau_{n}(-beta) function values for all n.

    Parameters
    ----------
    tau1: 'np.ndarray', (N,)
        Tau(-beta) function values for n = 1 expansion.
    beta: 'np.ndarray', (N,)
        Beta array of tau1 function.
    nphonon: 'int'
        Number of phonon to calculate the tau functions.
    threshold: 'float'
        Minimun value to take into account.

    Returns
    -------
    tauNfunc: 'np.ndarray', (nphonon, N * nphonon)
        All Tau(-beta) function values for n expansion.
    """
    Ntau1 = len(tau1)
    column_max = Ntau1 * nphonon
    tauNfunc = np.zeros((nphonon, column_max))
    tauNfunc[0, :Ntau1] += tau1
    calc_tauNfunc = calc_tauNfunc_gpu if gpu_available else calc_tauNfunc_cpu
    deltaBeta = Beta(beta).grid
    tauNfunc = calc_tauNfunc(tauNfunc, tau1, Ntau1, beta,
                                    deltaBeta, nphonon, 2 * Ntau1 - 1)
    # Erase the zeros in the last part of the array
    if threshold > 0.0:
        column_max = first_all_zero_column(tauNfunc, threshold)
    return tauNfunc[::, :column_max]


def save_tau(tauN: np.ndarray, nphonon: int, T: float, tauToFile: bool,
             binary: bool) -> None:
    """
    Save the tauN values in a file or in a binary file.

    Parameters
    ----------
    tauN: np.ndarray, (Z, T)
        tauN values for all the row T. Z is the number of the phonon expansion
        order and T is the number of the beta grid
    nphonon: int
        Phonon expansion order
    T: float
        Target temperature value for the caculation of tauN
    tauToFile: bool
        If True, save the tauN values in a file. If False, don't save the tauN
        values in a file. Default is False
    binary: bool
        If True, save the tauN values in a binary file. If False, save the tauN
        values in a txt file. Default is False.
    """
    name = f"tau_{nphonon}_{T}"
    if tauToFile:
        os.makedirs("tau", exist_ok=True)
        np.savetxt(f"tau/{name}.txt", tauN, delimiter="\t", fmt="%.14f")
    if binary:
        os.makedirs("tau/binary", exist_ok=True)
        with h5py.File(f"tau/binary/{name}.h5", "w") as f:
            f.create_dataset("tau", data=tauN)

@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def get_tauNbeta(tau1beta: np.ndarray, Nbeta=int):
    """
    Create the tauN_beta grid based on the tau1beta grid. The tauN_beta grid
    for beta is the same as the tau1beta grid for the first values and the rest
    of the values are the same as the tau1beta grid but with a step of the last
    value of the tau1beta grid.

    Parameters
    ----------
    tau1beta: np.ndarray
        Tau(-beta) function values for n = 1 expansion.
    Nbeta: int
        Length of the beta grid.

    Returns
    -------
    tauNbetaGrid: np.ndarray
        Tau(-beta) function values for n expansion.

    Examples
    --------
    >>> tau1beta = np.array([0.0, 0.05, 0.1, 0.2, 0.4])
    >>> get_tauNbeta(tau1beta, 8).round(2)
    array([0.  , 0.05, 0.1 , 0.2 , 0.4 , 0.6 , 0.8 , 1.  ])
    """
    # Get the length of tau1beta
    N = len(tau1beta)

    # If the length of tau1 beta is equal to Nbeta, return tau1beta as is
    if N == Nbeta:
        return tau1beta
    else:
        # Initialize an array of zeros with length Nbeta
        tauNbetaGrid = np.empty(Nbeta)

        # Add the values of tau1beta to the beginning of tauNbetaGrid
        tauNbetaGrid[:N] = tau1beta

        # Add the last value of tau1beta to the rest of tauNbetaGrid
        tauNbetaGrid[N:] = tau1beta[-1]

        # Calculate the difference between the last two values of tau1beta
        deltaBeta = tau1beta[-1] - tau1beta[-2]

        # Add a sequence of multiples of deltaBeta to the rest of tauNbetaGrid
        tauNbetaGrid[N:] += np.arange(1, Nbeta + 1 - N) * deltaBeta

        return tauNbetaGrid

