
import numpy as np
import numba as nb
from numba import cuda, float64, int32
from solid_cinel.core.dynamic_structure.beta import Beta
gpu_available = True if cuda.is_available() else False


@nb.jit(int32(float64[:, :], float64),
        nopython=True, cache=True)
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
    # Get the integration limit position
    trapz_limit = Ntau1 - 1

    # Initialize the convolution:
    convol = 0.
    for j in range(1, Ntau1):
        convol_j = 0.

        # tauNminus1(-(beta-beta^prime))
        k = i - j
        if abs(k) < NtauNminus1:
            convol_j += tauNminus1[k] if k >= 0 else tauNminus1[-k] * expBeta[-k]

        # tauNminus1(-(beta+beta^prime))
        l = i + j
        if l < NtauNminus1:
            convol_j += tauNminus1[l] * expBeta[j]

        # trapezoidal integration in the limit values of the convultion
        if j == trapz_limit:
            convol_j *= 0.5

        # trapezoidal integration
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
    # Get the position of the thread
    start = cuda.grid(1)

    # Get the stride of the threads
    stride = cuda.gridsize(1)

    # Call the kernel
    calc_tauN(expBeta, deltaBeta, tau1, Ntau1, tauNminus1, tauNdevice,
              start, NtauN, stride)


@nb.jit(nopython=True, cache=True)
def calc_tauNfunc_cpu(tauNfunc: np.ndarray, tau1: np.ndarray, Ntau1: int,
                      beta: np.ndarray, deltaBeta: np.ndarray, nphonon: int,
                      NtauN: int):
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
    # Copy the tau1 data for the firts iteraction:
    tauNminus1 = tau1.copy()

    # Get -beta exponential values:
    expBeta = np.exp(- beta)

    # Calculate the tauN(-beta) function values for all n > 1
    for n in range(1, nphonon):
        # Save the length of the tauN function:
        tauN = np.zeros(NtauN)

        # call the kernel
        calc_tauN(expBeta, deltaBeta, tau1, Ntau1, tauNminus1, tauN, 0, NtauN, 1)

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

    # Calculate the tauN(-beta) function values for all n > 1
    for n in range(1, nphonon):
        # Perform the calculation on the device:
        tauN = cuda.to_device(np.zeros(NtauN))

        # Calculate the number of blocks needed
        blockspergrid = NtauN + threadsperblock - 1
        blockspergrid //= threadsperblock

        # Call the kernel
        tauN_calculation_threads[blockspergrid, threadsperblock](expBeta, deltaBeta, tau1, Ntau1, tauNminus1,
                                                                 tauN, NtauN)

        # Copy the data back to the host
        tauNfunc[n, :NtauN] = tauN.copy_to_host()

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
    # Get the delta beta grid:
    deltaBeta = Beta(beta).grid

    # Get the length of tau1:
    Ntau1 = len(tau1)

    # Get the length of the columns in tauN function:
    column_max = Ntau1 * nphonon

    # Initialize the tauN function:
    tauNfunc = np.zeros((nphonon, column_max))

    # Calculate the tauN(-beta) function values for n = 0
    tauNfunc[0, :Ntau1] += tau1

    # Select the tauN function calculation procedure:
    calc_tauNfunc = calc_tauNfunc_gpu if gpu_available else calc_tauNfunc_cpu

    # Calculate the tauN(-beta) function values for all n > 1
    tauNfunc = calc_tauNfunc(tauNfunc, tau1, Ntau1, beta, deltaBeta, nphonon,
                             2 * Ntau1 - 1)

    # Erase the zeros in the last part of the array
    if threshold > 0.0:
        column_max = first_all_zero_column(tauNfunc, threshold)

    return tauNfunc[::, :column_max]


@nb.jit(float64[:](float64[:], int32),
    nopython=True, nogil=True, cache=True, parallel=False)
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

