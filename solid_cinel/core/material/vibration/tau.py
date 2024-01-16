
import numpy as np
import numba as nb
import h5py
import os
from math import exp
from numba import prange, cuda
from solid_cinel.core.generic import gpu_available
try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np


@nb.jit(nopython=True)
def find_first_all_zero_column_cpu(tau_n, threshold):
    for i in range(tau_n.shape[1]):
        if np.all(tau_n[:, i] <= threshold):
            return i
    return -1


def find_first_zero_column_gpu(tau_n, threshold, Ntau):
    for i in range(Ntau):
        if xp.all(tau_n[:, i] <= threshold):
            return i
    return -1  # Return -1 if no column with all zeros is found


@nb.jit("float64[:](float64, float64[:], float64[:])",
    nopython=True, nogil=True, cache=True, parallel=True)
def get_tau_n_cpu(delta_beta: float, tau1: np.ndarray,
                  tau_n_minus_1: np.ndarray) -> np.ndarray:
    """
    Get the tau_{n}(-beta) function values.

    Parameters
    ----------
    delta_beta : 'float'
        Interval of beta for the PDOS.
    tau1 : 'np.ndarray', (N,)
        Tau(-beta) function for n = 1 expansion.
    tau_n_minus_1 : 'np.ndarray', (M,)
        Tau(-beta) function for n - 1 expansion.
    threshold : 'float'
        Minimun value to take into account.

    Returns
    -------
    tau_n : 'np.ndarray', (N + M -1,)
        Tau(-beta) function for n expansion.
    """
    tau_n = np.zeros(len(tau1) + len(tau_n_minus_1) - 1)
    Nnm1 = len(tau_n_minus_1)  # length of tau_n_minus_1
    N = len(tau1)

    for i in prange(len(tau_n)):  # loop for tau_n
        # 1 iteration: j = 0
        tau_n[i] += tau1[0] * tau_n_minus_1[i] * delta_beta if i < Nnm1 else 0.

        # loop for tau1
        for j in range(1, N):
            convol = 0.

            k = i - j  # tau_n_minus_1(-(beta-beta^prime))
            if abs(k) < Nnm1:
                if k >= 0:
                    convol += tau_n_minus_1[k]
                else:
                    convol += tau_n_minus_1[-k] * exp(k * delta_beta)

            l = i + j  # Tau_n_minus_1(-(beta+beta^prime))
            if l < Nnm1:
                convol += tau_n_minus_1[l] * exp(-j * delta_beta)

            if j == N - 1:
                convol *= 0.5                      # trapz integrate

            tau_n[i] += tau1[j] * convol * delta_beta

    return tau_n


@nb.jit("float64[:, :](float64[:], float64, int32, float64)",
    nopython=True, nogil=True, cache=True, parallel=False)
def tau_n_functions_cpu(tau1: np.ndarray, delta_beta: float,
                        nphonon: int, threshold: float):
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
    tau_n_func: 'np.ndarray', (N * nphonon, nphonon)
        All Tau(-beta) function values for n expansion.
    """
    tau_n_func = np.zeros((nphonon, len(tau1) * nphonon))
    tau_n_func[0, :len(tau1)] += tau1
    tau_n_minus_1 = tau1.copy()
    for n in range(1, nphonon):
        tau_n = get_tau_n_cpu(delta_beta, tau1, tau_n_minus_1)
        tau_n_func[n, :len(tau_n)] += tau_n
        # Next tau_n
        tau_n_minus_1 = tau_n
    # Erase the zeros in the last part of the array
    max_column = find_first_all_zero_column_cpu(tau_n_func, threshold)
    return tau_n_func[::, :max_column]


@cuda.jit
def get_tau_n_gpu(delta_beta: float, tau1: np.ndarray, tau_n_minus_1: np.ndarray,
              tau_n: np.ndarray) -> np.ndarray:
    """
    Get the tau_{n}(-beta) function values.

    Parameters
    ----------
    delta_beta : 'float'
        Interval of beta for the PDOS.
    tau1 : 'np.ndarray', (N,)
        Tau(-beta) function for n = 1 expansion.
    tau_n_minus_1 : 'np.ndarray', (M,)
        Tau(-beta) function for n - 1 expansion.
    threshold : 'float'
        Minimun value to take into account.

    Returns
    -------
    tau_n : 'np.ndarray', (N + M - 1,)
        Tau(-beta) function for n expansion.
    """
    start = cuda.grid(1)      # 1 = one dimensional thread grid, returns a single value
    stride = cuda.gridsize(1)

    Nnm1 = len(tau_n_minus_1)  # length of tau_n_minus_1
    N = len(tau1)

    for i in range(start,  len(tau_n), stride):  # loop for tau_n
        # 1 iteration: j = 0
        tau_n[i] += tau1[0] * tau_n_minus_1[i] * delta_beta if i < Nnm1 else 0.

        # loop for tau1
        for j in range(1, N):
            convol = 0.

            k = i - j  # tau_n_minus_1(-(beta-beta^prime))
            if abs(k) < Nnm1:
                if k >= 0:
                    convol += tau_n_minus_1[k]
                else:
                    convol += tau_n_minus_1[-k] * exp(k * delta_beta)

            l = i + j  # Tau_n_minus_1(-(beta+beta^prime))
            if l < Nnm1:
                convol += tau_n_minus_1[l] * exp(-j * delta_beta)

            if j == N - 1:
                convol *= 0.5                      # trapz integrate

            tau_n[i] += tau1[j] * convol * delta_beta


def tau_n_functions_gpu(tau1: np.ndarray, delta_beta: float,
                        nphonon: int, threshold: float,
                        threadsperblock: int = 128) -> xp.ndarray:
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
    threadsperblock: 'int'
         How many parallel threads are grouped into a single block. The default
          is 128.

    Returns
    -------
    tau_n_func: 'np.ndarray', (N * nphonon, nphonon)
        All Tau(-beta) function values for n expansion.
    """
    N = len(tau1)
    Ntau = 2 * N - 1
    # Copy the data to the device if is posible:
    try:
        tau_n_func = xp.zeros((nphonon, len(tau1) * nphonon))
        tau1 = cuda.to_device(tau1)
        # First tau_n function:
        tau_n_func[0, :len(tau1)] += tau1
        gpu_memory_available = True
    except xp.cuda.memory.OutOfMemoryError:
        tau_n_func = np.zeros((nphonon, len(tau1) * nphonon))
        tau_n_func[0, :len(tau1)] += tau1
        tau1 = cuda.to_device(tau1)
        gpu_memory_available = False
    tau_n_minus_1 = cuda.to_device(tau1)

    # Next tau_n functions:
    for n in range(1, nphonon):
        # Perform the calculation on the device:
        tau_n_device = cuda.to_device(np.zeros(Ntau))
        blockspergrid = Ntau + threadsperblock - 1
        blockspergrid //= threadsperblock
        get_tau_n_gpu[blockspergrid, threadsperblock](delta_beta,
                                                      tau1,
                                                      tau_n_minus_1,
                                                      tau_n_device)
        # Copy the data back to the host if is needed:
        tau_n_func[n, :Ntau] += tau_n_device if gpu_memory_available else tau_n_device.copy_to_host()

        # Next tau_n
        tau_n_minus_1 = tau_n_device
        Ntau += N - 1
    if gpu_memory_available:
        max_column = find_first_zero_column_gpu(tau_n_func, threshold, Ntau)
    else:
        max_column = find_first_all_zero_column_cpu(tau_n_func, threshold)
    # copy the data back to the device :
    return xp.asarray(tau_n_func[::, :max_column])


if gpu_available:
    tau_n_functions = tau_n_functions_gpu
else:
    tau_n_functions = tau_n_functions_cpu


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
