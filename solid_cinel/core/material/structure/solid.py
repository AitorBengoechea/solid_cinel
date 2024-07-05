"""
Python file for working with the solid structure.

@author: AB272525
"""

from solid_cinel.core.material.structure.material_composition import Molecule
from solid_cinel.core.material.structure.crystal_symmetry import CrystalStructure
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import collections
from typing import Iterable, Union
collections.Callable = collections.abc.Callable


class Solid(CrystalStructure, Molecule):
    """
    Class for the solid structure. It is a combination of the crystal structure
    and the molecule class with the addition of the preferred orientation and
    the unit position of the atoms in the unit cell.
    """
    def __init__(self, preferred_orientation: Iterable,
                 unit_pos: Union[dict, Iterable], length: Iterable[float],
                 angles: Iterable[float], A: Iterable[int], Z: Iterable[int],
                 M: Iterable[float], b_coh: Iterable[float],
                 b_incoh: Iterable[float], name: str = None):
        """
        Initialize the solid class

        Parameters
        ----------
        preferred_orientation : Iterable
            Preferred orientation of the solid.
        unit_pos : Union[dict, Iterable]
            Position of the atoms in the unit cell.
        length : Iterable[float]
            direct vector lengths
        angles : Iterable[float]
            direct vector angles in degrees
        A : Iterable[int]
            Atomic number of the atoms in the unit cell.
        Z : Iterable[int]
            Number of atoms in the unit cell.
        M : Iterable[float]
            Atomic mass of the atoms in the unit cell.
        b_coh : Iterable[float]
            Coherent scattering length of the atoms in the unit cell.
        b_incoh : Iterable[float]
            Incoherent scattering length of the atoms in the unit cell.
        name : str, optional
            Name of the solid. The default is None.
        """
        CrystalStructure.__init__(self, length, angles)
        Molecule.__init__(self, A=A, Z=Z, M=M, b_coh=b_coh, b_incoh=b_incoh, name=name)
        self.preferred_orientation = pd.Series(preferred_orientation,
                                               index=["x", "y", "z"],
                                               name="preferred orientation")
        self.set_unit_pos(unit_pos)

    def set_unit_pos(self, unit_pos: Union[dict, Iterable]):
        """
        Set the position of the atoms in the unit cell.

        Parameters
        ----------
        unit_pos : Union[dict, Iterable]
            Position of the atoms in the unit cell.
        """
        col = pd.Index(["x", "y", "z"])
        if isinstance(unit_pos, dict):
            _unit_pos = {element: pd.DataFrame(single_unit_pos.reshape(-1, 3), columns=col)
                         for element, single_unit_pos in unit_pos.items()}
        else:
            _unit_pos = {self.name: pd.DataFrame(np.array(unit_pos).reshape(-1, 3), columns=col)}
        self.unit_pos = pd.Series(_unit_pos)

    @property
    def atom_pos(self) -> pd.Series:
        """
        Position of atoms in the direct lattice.

        Returns
        -------
        "pd.Series"
            Pandas series containnig the dataframe of the atom position in a
            direct lattice cell for each atom.

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.UO2 import *
        >>> UO2 = Solid(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atom_mass, b_coh, b_incoh)

        >>> UO2.atom_pos["O16"].round(6)
                  x         y         z
        0  1.386952  1.386952  1.386952
        1  4.160858  1.386952  1.386952
        2  1.386953  4.160858  4.160858
        3  4.160858  4.160858  4.160858
        4  4.160858  1.386953  4.160858
        5  1.386953  1.386953  4.160858
        6  4.160858  4.160858  1.386952
        7  1.386953  4.160858  1.386952

        >>> UO2.atom_pos["U238"].round(6)
                  x         y         z
        0  2.773905  0.000000  0.000000
        1  2.773905  2.773905  2.773905
        2  0.000000  0.000000  2.773905
        3  0.000000  2.773905  0.000000
        """
        return self.unit_pos.apply(lambda x: pd.DataFrame(x.values.dot(self.dir_vec),
                                                          columns=x.columns))

    @property
    def atom_number(self) -> int:
        """
        Number of atoms in the unit cell.

        Returns
        -------
        "int"
            Number of atom in a molecule

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> Al = Solid(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atomic_mass, b_coh, b_incoh)
        >>> from solid_cinel.data.materials.UO2 import *
        >>> UO2 = Solid(preferred_orientation, unit_pos, dir_vec_length, dir_vec_angles, A, Z, atom_mass, b_coh, b_incoh)

        Test the results:
        >>> assert Al.atom_number == 1
        >>> assert UO2.atom_number == 12
        """
        return sum(self.atom_pos.apply(lambda x: x.shape[0]).values)


def hkl_max_value(rec_vecs: np.ndarray, d_min: float,
                  precision: float = 1.0e-7) -> np.ndarray:
    """
    Get the maximun h, k and l integers for the constrain of d > d_min.

    Parameters
    ----------
    rec_vecs : 'np.ndarray', (3, 3)
        Reciprocal vectors.
    d_min : 'float'
        The minimun d space.
    precision: "float", optional
        Precision of the minimization problem. The default is 1.0e-7.

    Returns
    -------
    "np.ndarray", (3,)
        Array with the integers of the maximum hkl planes.

    Example
    -------
    Object initialization:
    >>> from solid_cinel.data.materials.Al27 import *
    >>> crys = CrystalStructure(dir_vec_length, dir_vec_angles)
    >>> rec_vecs = crys.reciproc_vec.values
    >>> d_min = 0.2360746677309732
    >>> hkl_max_value(rec_vecs, d_min)
    array([12, 12, 12])
    """
    def hkl_range_minimization(x, i):
        return x[i]

    def constrain(x, d_min):
        vec_tau_hkl = x[0] * rec_vecs[0] + x[1] * rec_vecs[1] + x[2] * rec_vecs[2]
        vec_tau_hkl_norm = np.linalg.norm(vec_tau_hkl)
        d_hkl = 2 * np.pi / vec_tau_hkl_norm
        return d_hkl - d_min
    result = []
    for i in range(3):
        result.append(minimize(lambda x: hkl_range_minimization(x, i),
                               [-100, -100, -100],
                               method='COBYLA',
                               constraints=({'type': 'ineq', 'fun': constrain, 'args': [d_min]})).x)
    confidance = np.array(list(map(lambda x: constrain(x, d_min), result)))
    hkl_max = abs(np.array(result).diagonal().astype(int))
    return np.where(abs(confidance) <= precision,  hkl_max, 100)
