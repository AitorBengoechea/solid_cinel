
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
def tau_n_convolution(delta_beta, tau1, tau_n_minus_1, i):
    """
    Calculate the convolution in the tau_n[i] between  tau1 and tau_n_minus_1.

    Parameters
    ----------
    delta_beta: float
        beta interval between two consecutive values.
    tau1: np.ndarray, (N,)
        Tau(-beta) function values for n = 1 expansion.
    tau_n_minus_1: np.ndarray, (M,)
        Tau(-beta) function values for n-1 expansion.
    i: int
        Index of the tau_n[i] value to calculate.

    Returns
    -------
    convol: float
        Convolution value.
    """
    N = len(tau1)
    Nnm1 = len(tau_n_minus_1)  # length of tau_n_minus_1
    convol = 0.
    for j in range(1, N):
        convol_j = 0.

        k = i - j  # tau_n_minus_1(-(beta-beta^prime))
        if abs(k) < Nnm1:
            if k >= 0:
                convol_j += tau_n_minus_1[k]
            else:
                convol_j += tau_n_minus_1[-k] * exp(k * delta_beta)

        l = i + j  # Tau_n_minus_1(-(beta+beta^prime))
        if l < Nnm1:
            convol_j += tau_n_minus_1[l] * exp(-j * delta_beta)

        if j == N - 1:
            convol_j *= 0.5  # trapz integrate

        convol += tau1[j] * convol_j * delta_beta
    return convol
    


@optional_jit
def calculate_tau_n(delta_beta, tau1, tau_n_minus_1, tau_n,
                    start, final, stride):
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
    Nnm1 = len(tau_n_minus_1)
    # Tau_N(-beta) loop:
    for i in range(start, final, stride):
        # 1 iteration: j = 0
        tau_n[i] += tau1[0] * tau_n_minus_1[i] * delta_beta if i < Nnm1 else 0.
        # rest of iterations:
        tau_n[i] += tau_n_convolution(delta_beta, tau1, tau_n_minus_1, i)


@cuda.jit
def tau_n_calculation_threads(delta_beta, tau1, tau_n_minus_1, tau_n_device,
                              final):
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
    calculate_tau_n(delta_beta, tau1, tau_n_minus_1, tau_n_device,
                    start, final, stride)


@nb.jit(nopython=True, nogil=True, cache=True, parallel=False)
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
    N = len(tau1)
    Ntau = 2 * N - 1
    for n in range(1, nphonon):
        tau_n = np.zeros(Ntau)
        calculate_tau_n(delta_beta, tau1, tau_n_minus_1, tau_n, 0, Ntau, 1)
        # Copy thet data into the array:
        tau_n_func[n, :Ntau] += tau_n
        # If the last N values are zero, the next tau_n will have the same length
        # because the convolution will be zero for the following values
        Ntau = Ntau if np.all(tau_n[-N:] == 0.0) else Ntau + N - 1
        # Next tau_n:
        tau_n_minus_1 = tau_n

    # Erase the zeros in the last part of the array
    if threshold == 0.0:
        return tau_n_func[::, :Ntau]
    else:
        return tau_n_func[::, :first_all_zero_column(tau_n_func, threshold)]


def tau_n_functions_gpu(tau1: np.ndarray, delta_beta: float,
                        nphonon: int, threshold: float,
                        threadsperblock: int = 128):
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
    tau_n_func = np.zeros((nphonon, len(tau1) * nphonon))
    tau_n_func[0, :len(tau1)] += tau1
    N = len(tau1)
    Ntau = 2 * N - 1
    # Copy the data to the device
    tau1 = cuda.to_device(tau1)
    tau_n_minus_1 = cuda.to_device(tau1)
    for n in range(1, nphonon):
        # Perform the calculation on the device:
        tau_n = cuda.to_device(np.zeros(Ntau))
        blockspergrid = Ntau + threadsperblock - 1
        blockspergrid //= threadsperblock
        tau_n_calculation_threads[blockspergrid, threadsperblock](delta_beta, tau1, tau_n_minus_1,
                                                                  tau_n, Ntau)

        # Copy the data back to the host
        tau_n_func[n, :Ntau] += tau_n.copy_to_host()

        # If the last N values are zero, the next tau_n will have the same length
        # because the convolution will be zero for the following values
        Ntau = Ntau if np.all(tau_n[-N:] == 0.0) else Ntau + N - 1
        # Next tau_n:
        tau_n_minus_1 = tau_n
    # Erase the zeros in the last part of the array
    if threshold == 0.0:
        return tau_n_func[::, :Ntau]
    else:
        return tau_n_func[::, :first_all_zero_column(tau_n_func, threshold)]


tau_n_functions = tau_n_functions_gpu if gpu_available else tau_n_functions_cpu


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
