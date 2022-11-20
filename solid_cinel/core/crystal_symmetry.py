import numpy as np
import pandas as pd

def dir_vector_operator(angles, symmetry="cubic") -> pd.DataFrame:
    """
    Generate direct lattice vectors accordinto to the symmetry of the solid.

    Parameters
    ----------
    angles : 'pd.Series'
        Angles of the direct lattice vectors
    symmetry : str, optional
        Symmetry of the solid, by default "cubic"
    """
    if symmetry.lower() == "cubic":
        operator = pd.DataFrame(cubic(angles))
    operator.index = pd.Index(["a1", "a2", "a3"])
    operator.columns = pd.Index(["x", "y", "z"])
    return operator

def cubic(angles) -> np.ndarray:
    """
    Generate the operator for obteining the direct lattice vectors in cubic
    symmetry based on the angles.

    Parameters
    ----------
    angles : 'pd.Series'
        Angles of the direct lattice vectors

    Example
    -------
    >>> angles = pd.Series([60, 60, 60], index=["alpha", "beta", "gamma"])
    >>> cubic_vec = cubic(angles)

    Test the results:
    >>> assert all(cubic_vec[0].round(6) == np.array([1.      , 0.      , 0.]))
    >>> assert all(cubic_vec[1].round(6) == np.array([0.5     , 0.866025, 0.      ]))
    >>> assert all(cubic_vec[2].round(6) == np.array([0.5     , 0.288675, 0.816497]))
    """
    a = np.array([1.,
                  0.,
                  0.])
    b = np.array([np.cos(angles["gamma"]),
                  np.sin(angles["gamma"]),
                  0.])
    c = np.array([np.cos(angles["beta"]),
                  np.cos(angles["alpha"]) - np.cos(angles["beta"]) * np.cos(angles["gamma"]),
                  1.0])
    c[1] /=  np.sin(angles["gamma"])
    c[2] *= np.sqrt(1. - c[0] ** 2 - c[1] ** 2)
    return [a, b, c]