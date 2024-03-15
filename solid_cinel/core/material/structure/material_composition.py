"""
Python file for working with material composition.

@author: AB272525
"""
from solid_cinel.data.elements import ELEMENTS
from scipy.constants import physical_constants as const
import numpy as np
import pandas as pd
import collections
from typing import Iterable, Union
collections.Callable = collections.abc.Callable


class Atom:
    """
    Class to store properties of the atoms.

    Attributes
    ----------
    A : 'int'
        Atomic number.
    Z : 'int'
        Number of protons.
    atom_mass : 'float'
        Atom mass, amu.
    b : 'dict'
        Dictionary with the bound coherent and incoherent scattering lengths
        (fm)

    Properties
    ----------
    name : 'str'
        Material name: element + A
    zam: 'int'
        (Z * 1000 + A) * 10
    boundXs: float
        Bound total scattering xs in barns
    boundIncXs: float
        Bound incoherent scattering xs in barns
    freeXs: float
        Free total scattering xs in barns

    """

    def __init__(self, A: int, Z: int, atom_mass: float,
                 b_coh: float, b_incoh: float):
        """
        Initialize the Atom class to describe a single atoms.

        Parameters
        ----------
        A : 'int'
            Atomic number.
        Z : 'int'
            Number of protons.
        atom_mass : 'float'
            Atom mass, amu.
        b_coh : 'float'
            Bound coherent scattering length (fm).
        b_incoh : 'float'
            Bound incoherent scattering length (fm).
        """
        self.A = A
        self.Z = Z
        self.atom_mass = atom_mass
        self.b = {"b_coh": b_coh, "b_incoh": b_incoh}

    @property
    def name(self) -> str:
        """
        Material name: element + A.

        Returns
        -------
        "str"
            Element + A

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> Al = Atom(A, Z, atomic_mass, b_coh, b_incoh)

        Test the results:
        >>> assert Al.name == "Al27"
        """
        return f"{ELEMENTS[self.Z]}{self.A}"

    @property
    def zam(self) -> int:
        """
        Zam number of a atom.

        Returns
        -------
        "int"
            Zam number

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> Al = Atom(A, Z, atomic_mass, b_coh, b_incoh)

        Test the results:
        >>> assert Al.zam == 130270
        """
        zam = self.Z * 10000 + self.A * 10
        return int(zam)

    @property
    def boundXs(self) -> float:
        """
        Bound total scattering cross section in barn.

        Returns
        -------
        "float"
            Bound total scattering xs

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> Al = Atom(A, Z, atomic_mass, b_coh, b_incoh)

        Test the results:
        >>> assert np.double(Al.boundXs).round(6) == np.double(1.503081)
        """
        return np.pi * 4 / 100 * (self.b["b_incoh"] ** 2 + self.b["b_coh"] ** 2)

    @property
    def boundIncXs(self) -> float:
        """
        Bound incoherent scattering cross section in barn.

        Returns
        -------
        "float"
            Bound incoherent scattering xs.

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> Al = Atom(A, Z, atomic_mass, b_coh, b_incoh)

        Test the results:
        >>> assert np.double(Al.boundIncXs).round(6) == np.double(0.008235)
        """
        return np.pi * 4 / 100 * self.b["b_incoh"] ** 2

    @property
    def freeXs(self) -> float:
        """
        Free scattering cross section in barn.

        Returns
        -------
        "float"
            Free scattering xs.

        Example
        -------
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> Al = Atom(A, Z, atomic_mass, b_coh, b_incoh)

        Test the results:
        >>> assert np.double(Al.freeXs).round(6) == np.double(1.396702)
        """
        A = self.atom_mass / const["neutron mass in u"][0]
        return self.boundXs * (A / (A + 1)) ** 2


class Molecule(Atom):
    """
    Class to store the properties and method for all the atom of
    the molecule.

    Attributes
    ----------
    A : 1D iterable of 'int' or 'int'
        Atomic number.
    Z : 1D iterable of 'int' or 'int'
        Number of protons.
    atom_mass : 1D iterable of 'float' or 'float'
        Atom mass, amu.
    b_coh : 1D iterable of 'float' or 'float'
        Bound coherent scattering length (fm).
    b_incoh : 1D iterable of 'float' or 'float'
        Bound incoherent scattering length (fm).
    name : 'str', optional
        Molecule name
    """

    def __init__(self, A: Union[Iterable, int], Z: Union[Iterable, int],
                 atom_mass: Union[Iterable, float], b_coh: Union[Iterable, float],
                 b_incoh: Union[Iterable, float], name: str = None):
        """
        Initialize the Molecule class to describe a molecule.

        Parameters
        ----------
        A : 1D iterable of 'int' or 'int'
            Atomic number.
        Z : 1D iterable of 'int' or 'int'
            Number of protons.
        atom_mass : 1D iterable of 'float' or 'float'
            Atom mass, amu.
        b_coh : 1D iterable of 'float' or 'float'
            Bound coherent scattering length (fm).
        b_incoh : 1D iterable of 'float' or 'float'
            Bound incoherent scattering length (fm).
        name : 'str', optional
            Name of the molecule, by default None
        """
        atoms = {}
        if isinstance(A, int):
            atom = Atom(A, Z, atom_mass, b_coh, b_incoh)
            atoms[atom.name] = atom
            name = atom.name if name is None else name
        else:
            for i in range(len(A)):
                single_atom = Atom(A[i], Z[i], atom_mass[i],
                                   b_coh[i], b_incoh[i])
                atoms[single_atom.name] = single_atom
        self.atoms = pd.Series(atoms)
        self.name = name

    @property
    def name(self) -> str:
        """
        Name of the molecule given by the user.

        Returns
        -------
        "str"
            Name of the molecule.

        Example
        -------
        Object initialization:
        Object initialization:
        >>> from solid_cinel.data.materials.Al27 import *
        >>> Al = Molecule(A, Z, atomic_mass, b_coh, b_incoh, name="Al")

        Test the results:
        >>> assert Al.name == "Al"
        >>> assert Al.atoms["Al27"].name == "Al27"
        >>> Al = Molecule(A, Z, atomic_mass, b_coh, b_incoh)
        >>> assert Al.name == "Al27"
        """
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name if name is not None else "Molecule"
