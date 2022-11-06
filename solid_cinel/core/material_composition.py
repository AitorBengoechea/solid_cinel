from solid_cinel.data import elements
import numpy as np
import pandas as pd
import collections
import pytest
collections.Callable = collections.abc.Callable

class Atom():
    def __init__(self, A, Z, atom_mass, b_coh, b_incoh):
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
        self.b = {"b_coh": b_coh,
                  "b_incoh": b_incoh}

    @property
    def name(self) -> str:
        """
        Material name: element + A

        Example
        -------
        Object initialization:
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Atom(A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

        Test the results:
        >>> assert Al.name == "Al27"

        """
        return elements.ELEMENTS[self.Z] + str(self.A)

    @property
    def zam(self) -> int:
        """
        Zam number of a atom.

        Example
        -------
        Object initialization:
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Atom(A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

        Test the results:
        >>> assert Al.zam == 130270
        """
        zam = self.Z * 10000 + self.A * 10 
        return int(zam)

class Molecule(Atom):
    def __init__(self, A, Z, atom_mass, b_coh, b_incoh, name=None):
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
        self.atoms = {}
        if isinstance(A, int):
            atom = Atom(A, Z, atom_mass, b_coh, b_incoh)
            self.atoms[atom.name] = atom
            name = atom.name if name is None else name
        else:
            for i in range(len(A)):
                single_atom = Atom(A[i], Z[i], atom_mass[i],
                                   b_coh[i], b_incoh[i])
                self.atoms[single_atom.name] = single_atom
        self.name = name

    @property
    def name(self) -> str:
        """
        Name of the molecule given by the user

        Example
        -------
        Object initialization:
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Molecule(A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27, name="Al")

        Test the results:
        >>> assert Al.name == "Al"
        >>> assert Al.atoms["Al27"].name == "Al27"
        >>> Al = Molecule(A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)
        >>> assert Al.name == "Al27"
        """
        return self._name

    @name.setter
    def name(self, name):
        if name is not None:
            self._name = name
        else:
            self._name = "Molecule"