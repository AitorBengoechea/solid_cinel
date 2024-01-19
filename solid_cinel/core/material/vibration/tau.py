
import numpy as np
import numba as nb
import h5py
import os
from math import exp
from numba import cuda
gpu_available = True if cuda.is_available() else False


@nb.jit(nopython=True)
def first_all_zero_column(tau_n, threshold):
    for i in range(tau_n.shape[1]):
        if np.all(tau_n[:, i] <= threshold):
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
def tau_n_convolution(delta_beta, tau1, Ntau1, tau_n_minus_1, Ntaunm1, i):
    """
    Calculate the convolution in the tau_n[i] between  tau1 and tau_n_minus_1.

    Parameters
    ----------
    delta_beta: float
        beta interval between two consecutive values.
    tau1: np.ndarray, (N,)
        Tau(-beta) function values for n = 1 expansion.
    Ntau1: int
        Length of tau1.
    tau_n_minus_1: np.ndarray, (M,)
        Tau(-beta) function values for n-1 expansion.
    Ntaunm1: int
        Length of tau_n_minus_1.
    i: int
        Index of the tau_n[i] value to calculate.

    Returns
    -------
    convol: float
        Convolution value.
    """
    trapz_limit = Ntau1 - 1
    convol = 0.
    for j in range(1, Ntau1):
        convol_j = 0.

        k = i - j  # tau_n_minus_1(-(beta-beta^prime))
        if abs(k) < Ntaunm1:
            if k >= 0:
                convol_j += tau_n_minus_1[k]
            else:
                convol_j += tau_n_minus_1[-k] * exp(k * delta_beta)

        l = i + j  # Tau_n_minus_1(-(beta+beta^prime))
        if l < Ntaunm1:
            convol_j += tau_n_minus_1[l] * exp(-j * delta_beta)

        if j == trapz_limit:
            convol_j *= 0.5  # trapz integrate

        convol += tau1[j] * convol_j * delta_beta
    return convol
    


@optional_jit
def calculate_tau_n(delta_beta, tau1, Ntau1, tau_n_minus_1, tau_n,
                    start, Ntaun, stride):
    """
    Calculate the tau_n(-beta) function values for all n. The values are
    calculated until the last value of tau_n is zero.

    Parameters
    ----------
    delta_beta: float
        beta interval between two consecutive values.
    tau1: np.ndarray, (N,)
        Tau(-beta) function values for n = 1 expansion.
    tau_n_minus_1: np.ndarray, (M,)
        Tau(-beta) function values for n-1 expansion.
    tau_n: np.ndarray, (M,)
        Tau(-beta) function values for n expansion.
    start:  int
        position to start the calculation
    final: int
        position to end the calculation
    stride: int
        step of the calculation

    Returns
    -------
    tau_n: np.ndarray, (M,)
        Tau(-beta) function values for n expansion.
    """
    Ntaunm1 = Ntaun - Ntau1 + 1
    # Tau_N(-beta) loop:
    for i in range(start, Ntaun, stride):
        # 1 iteration: j = 0
        tau_n[i] += tau1[0] * tau_n_minus_1[i] * delta_beta if i < Ntaunm1 else 0.
        # rest of iterations:
        tau_n[i] += tau_n_convolution(delta_beta, tau1, Ntau1, tau_n_minus_1, Ntaunm1,  i)


@cuda.jit
def tau_n_calculation_threads(delta_beta, tau1, Ntau1, tau_n_minus_1, tau_n_device,
                              Ntaun):
    """
    Calculate the tau_n(-beta) function values for all n. The values are
    calculated until the last value of tau_n is zero.

    Parameters
    ----------
    delta_beta: float
        beta interval between two consecutive values.
    tau1: np.ndarray, (N,)
        Tau(-beta) function values for n = 1 expansion.
    tau_n_minus_1: np.ndarray, (M,)
        Tau(-beta) function values for n-1 expansion.
    tau_n_device: np.ndarray, (M,)
        Tau(-beta) function values for n expansion.
    final: int
        position to end the calculation

    Returns
    -------
    tau_n_device: np.ndarray, (M,)
        Tau(-beta) function values for n expansion.
    """
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    calculate_tau_n(delta_beta, tau1, Ntau1, tau_n_minus_1, tau_n_device,
                    start, Ntaun, stride)


@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
def calculate_tau_n_functions_cpu(tau_n_func: np.ndarray, tau1: np.ndarray,
                                  Ntau1: int, delta_beta: float, nphonon: int, Ntaun: int,):
    """
    Get the tau_{n}(-beta) function values for all n.

    Parameters
    ----------
    tau_n_func: 'np.ndarray', (nphonon, N * nphonon)
        All Tau(-beta) function values for n = 1 expansion.
    tau1: 'np.ndarray', (N,)
        Tau(-beta) function values for n = 1 expansion.
    Ntau1: int
        Length of tau1.
    delta_beta: 'float'
        Interval of beta for the PDOS.
    nphonon: 'int'
        Number of phonon to calculate the tau functions.
    Ntaun: 'int'
        Length of the tau_2 function: 2 * N - 1.
    threshold: 'float'
        Minimun value to take into account.

    Returns
    -------
    tau_n_func: 'np.ndarray', (N * nphonon, nphonon)
        All Tau(-beta) function values for n expansion.
    """
    tau_n_minus_1 = tau1.copy()
    for n in range(1, nphonon):
        tau_n = np.zeros(Ntaun)
        calculate_tau_n(delta_beta, tau1, Ntau1, tau_n_minus_1, tau_n, 0, Ntaun, 1)
        # Copy thet data into the array:
        tau_n_func[n, :Ntaun] += tau_n
        # If the last N values are zero, the next tau_n will have the same length
        # because the convolution will be zero for the following values
        Ntaun = Ntaun if np.all(tau_n[-Ntau1:] == 0.0) else Ntaun + Ntau1 - 1
        # Next tau_n:
        tau_n_minus_1 = tau_n

    # Erase the zeros in the last part of the array
    return tau_n_func[::, :Ntaun]


def calculate_tau_n_functions_gpu(tau_n_func: np.ndarray, tau1: np.ndarray,
                                  Ntau1: int, delta_beta: float, nphonon: int, Ntaun: int,
                                  threadsperblock: int = 128) -> np.ndarray:

    """
    Get the tau_{n}(-beta) function values for all n.

    Parameters
    ----------
    tau_n_func: 'np.ndarray', (nphonon, N * nphonon)
        All Tau(-beta) function values for n = 1 expansion.
    tau1: 'np.ndarray', (N,)
        Tau(-beta) function values for n = 1 expansion.
    Ntau1: int
        Length of tau1.
    delta_beta: 'float'
        Interval of beta for the PDOS.
    nphonon: 'int'
        Number of phonon to calculate the tau functions.
    Ntaun: 'int'
        Length of the tau_2 function: 2 * N - 1.
    threshold: 'float'
        Minimun value to take into account.
    threadsperblock: 'int'
         How many parallel threads are grouped into a single block. The default
          is 128.

    Returns
    -------
    tau_n_func: 'np.ndarray', (N * nphonon, nphonon)
        All Tau(-beta) function values for n expansion.
    """
    # Copy the data to the device
    tau1 = cuda.to_device(tau1)
    tau_n_minus_1 = cuda.to_device(tau1)
    for n in range(1, nphonon):
        # Perform the calculation on the device:
        tau_n = cuda.to_device(np.zeros(Ntaun))
        blockspergrid = Ntaun + threadsperblock - 1
        blockspergrid //= threadsperblock
        tau_n_calculation_threads[blockspergrid, threadsperblock](delta_beta, tau1, Ntau1, tau_n_minus_1,
                                                                  tau_n, Ntaun)

        # Copy the data back to the host
        tau_n_func[n, :Ntaun] += tau_n.copy_to_host()

        # If the last N values are zero, the next tau_n will have the same length
        # because the convolution will be zero for the following values
        Ntaun = Ntaun if np.all(tau_n[-Ntau1:] == 0.0) else Ntaun + Ntau1 - 1
        # Next tau_n:
        tau_n_minus_1 = tau_n
    return tau_n_func[::, :Ntaun]


def tau_n_functions(tau1: np.ndarray, delta_beta: float,
                    nphonon: int, threshold: float) -> np.ndarray:
    """
    Get the tau_{n}(-beta) function values for all n.

    Parameters
    ----------
    tau1: 'np.ndarray', (N,)
        Tau(-beta) function values for n = 1 expansion.
    delta_beta: 'float'
        Interval of beta for the PDOS.
    nphonon: 'int'
        Number of phonon to calculate the tau functions.
    threshold: 'float'
        Minimun value to take into account.

    Returns
    -------
    tau_n_func: 'np.ndarray', (nphonon, N * nphonon)
        All Tau(-beta) function values for n expansion.
    """
    Ntau1 = len(tau1)
    column_max = Ntau1 * nphonon
    tau_n_func = np.zeros((nphonon, column_max))
    tau_n_func[0, :Ntau1] += tau1
    calculate_tau_n_functions = calculate_tau_n_functions_gpu if gpu_available else calculate_tau_n_functions_cpu
    tau_n_func = calculate_tau_n_functions(tau_n_func, tau1, Ntau1, delta_beta, nphonon, 2 * Ntau1 - 1)
    # Erase the zeros in the last part of the array
    if threshold > 0.0:
        column_max = first_all_zero_column(tau_n_func, threshold)
    return tau_n_func[::, :column_max]


def save_tau(tau_n: np.ndarray, nphonon: int, T: float, tau_to_file: bool,
              binary: bool) -> None:
    """
    Save the tau_n values in a file or in a binary file.

    Parameters
    ----------
    tau_n: np.ndarray, (Z, T)
        tau_n values for all the row T. Z is the number of the phonon expansion
        order and T is the number of the beta grid
    nphonon: int
        Phonon expansion order
    T: float
        Target temperature value for the caculation of tau_n
    tau_to_file: bool
        If True, save the tau_n values in a file. If False, don't save the tau_n
        values in a file. Default is False
    binary: bool
        If True, save the tau_n values in a binary file. If False, save the tau_n
        values in a txt file. Default is False.
    """
    name = f"tau_{nphonon}_{T}"
    if tau_to_file:
        os.makedirs("tau", exist_ok=True)
        np.savetxt(f"tau/{name}.txt", tau_n, delimiter="\t", fmt="%.14f")
    if binary:
        os.makedirs("tau/binary", exist_ok=True)
        with h5py.File(f"tau/binary/{name}.h5", "w") as f:
            f.create_dataset("tau", data=tau_n)
