"""
Python file for working with material composition.

@author: AB272525
"""
import numpy as np
import pandas as pd
import collections
from dataclasses import dataclass, field
from typing import Iterable, Union, Dict, List
from solid_cinel.data.elements import ELEMENTS
from scipy.constants import physical_constants as const
collections.Callable = collections.abc.Callable


@dataclass
class Atom:
    """
    Represents an atom with its basic nuclear and scattering properties.

    Attributes
    ----------
    A : int
        Mass number of the atom.
    Z : int
        Atomic number, representing the number of protons in the nucleus.
    M : float
        Atomic mass of the atom in atomic mass units (amu).
    b_coh : float
        Coherent scattering length of the atom in barns (b).
    b_incoh : float
        Incoherent scattering length of the atom in barns (b).
    b : Dict[str, float], optional
        Dictionary containing both coherent ('b_coh') and incoherent ('b_incoh') scattering lengths.

    Methods
    -------
    __post_init__(self) -> None:
        Initializes the `b` attribute with coherent and incoherent scattering lengths.

    Examples
    --------
    >>> import os
    >>> file_dir = os.path.dirname(os.path.abspath(__file__))
    >>> file_path = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
    >>> Al = Atom.from_iter(*Molecule.get_var_from_file(file_path))
    >>> assert Al.name == "Al27"
    """
    A: int
    Z: int
    M: float
    b_coh: float
    b_incoh: float
    b: Dict[str, float] = None

    def __post_init__(self) -> None:
        """
        Initialize the Atom class to describe b values.
        """
        self.b = {"b_coh": self.b_coh, "b_incoh": self.b_incoh}


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
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))
        >>> file_path = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> Al = Atom.from_iter(*Molecule.get_var_from_file(file_path))

        Test the results:
        >>> assert Al.name == "Al27"
        """
        return f"{ELEMENTS[int(self.Z)]}{int(self.A)}"

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
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))
        >>> file_path = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> Al = Atom.from_iter(*Molecule.get_var_from_file(file_path))

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
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))
        >>> file_path = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> Al = Atom.from_iter(*Molecule.get_var_from_file(file_path))

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
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))
        >>> file_path = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> Al = Atom.from_iter(*Molecule.get_var_from_file(file_path))

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
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))
        >>> file_path = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> Al = Atom.from_iter(*Molecule.get_var_from_file(file_path))

        Test the results:
        >>> assert np.double(Al.freeXs).round(6) == np.double(1.396702)
        """
        A = self.M / const["neutron mass in u"][0]
        return self.boundXs * (A / (A + 1)) ** 2

    @classmethod
    def from_dict(cls, data: Dict[str, Union[int, float]]) -> "Atom":
        """
        Create an Atom instance from a dictionary.

        Parameters
        ----------
        data : Dict[str, Union[int, float]]
            Dictionary containing the atom properties.

        Returns
        -------
        Atom
            An instance of Atom initialized with the properties from the dictionary.

        Example
        -------
        >>> data = {'A': 27, 'Z': 13, 'M': 26.98153433356103, 'b_coh': 3.449, 'b_incoh': 0.256}
        >>> atom = Atom.from_dict(data)
        >>> assert atom.A == 27
        """
        return cls(A=data["A"], Z=data["Z"], M=data["M"],
                   b_coh=data["b_coh"], b_incoh=data["b_incoh"])

    @classmethod
    def from_iter(cls, data: Iterable[Union[int, float]]) -> "Atom":
        """
        Create an Atom instance from an iterable.

        Parameters
        ----------
        data : Iterable[Union[int, float]]
            Iterable containing the atom properties in the order: A, Z, M, b_coh, b_incoh.

        Returns
        -------
        Atom
            An instance of Atom initialized with the properties from the iterable.

        Example
        -------
        >>> data = (27, 13, 26.98153433356103, 3.449, 0.256)
        >>> atom = Atom.from_iter(data)
        >>> assert atom.Z == 13
        """
        A, Z, M, b_coh, b_incoh = data
        return cls(A=A, Z=Z, M=M, b_coh=b_coh, b_incoh=b_incoh)

    @property
    def to_string(self) -> str:
        """
        Returns a string representation of the Atom instance in the specified format.

        Returns
        -------
        str
            The string representation of the Atom instance.

        Example
        -------
        >>> from pprint import pprint
        >>> data = (27, 13, 26.98153433356103, 3.449, 0.256)
        >>> atom = Atom.from_iter(data)
        >>> print(atom.to_string)
        # Al27 information:
        # A:
        27
        # Z:
        13
        # Atomic mass in amu:
        26.98153433356103
        # bound coherant cross section in barn:
        3.449
        # bound incoherent cross section in barn:
        0.256
        """
        info_str = "\n".join([
            f"# {self.name} information:",
            f"# A:\n{self.A}",
            f"# Z:\n{self.Z}",
            f"# Atomic mass in amu:\n{self.M}",
            f"# bound coherant cross section in barn:\n{self.b_coh}",
            f"# bound incoherent cross section in barn:\n{self.b_incoh}"
        ])
        return info_str


@dataclass
class Molecule:
    """
    Represents a molecule with its atomic composition and scattering properties.

    Attributes
    ----------
    A : Union[List[int], int]
        Mass number of the atoms in the molecule.
    Z : Union[List[int], int]
        Atomic number of the atoms in the molecule.
    M : Union[List[float], float]
        Atomic mass of the atoms in the molecule in atomic mass units (amu).
    b_coh : Union[List[float], float]
        Coherent scattering length of the atoms in the molecule in barns (b).
    b_incoh : Union[List[float], float]
        Incoherent scattering length of the atoms in the molecule in barns (b).
    name : str, optional
        Name of the molecule.
    atoms : pd.Series
        Series containing Atom instances for each atom in the molecule.
    """
    A: Union[List[int], int]
    Z: Union[List[int], int]
    M: Union[List[float], float]
    b_coh: Union[List[float], float]
    b_incoh: Union[List[float], float]
    name: str = None
    atoms: pd.Series = field(init=False)

    @staticmethod
    def _ensure_iterable(value):
        """Ensure the input value is iterable."""
        if isinstance(value, Iterable):
            return value
        return [value]

    def __post_init__(self) -> "Molecule":
        """
        Initialize the Molecule class to describe a molecule.
        """
        # Ensure all input values are iterable
        self.A = self._ensure_iterable(self.A)
        self.Z = self._ensure_iterable(self.Z)
        self.M = self._ensure_iterable(self.M)
        self.b_coh = self._ensure_iterable(self.b_coh)
        self.b_incoh = self._ensure_iterable(self.b_incoh)

        # Check if all input iterables have the same length
        if not all(len(lst) == len(self.A) for lst in [self.Z, self.M, self.b_coh, self.b_incoh]):
            raise ValueError("All input iterables must have the same length.")

        # Create a dictionary of Atom instances and store them in a pandas Series
        atoms_dict = {}
        self.atomNum = len(self.A)
        for i in range(self.atomNum):
            atom = Atom(self.A[i], self.Z[i], self.M[i], self.b_coh[i], self.b_incoh[i])
            atoms_dict[atom.name] = atom
        self.atoms = pd.Series(atoms_dict)

        # Set the molecule name if not provided
        if self.name is None:
            self.name = "Molecule" if self.atomNum > 1 else atom.name

    @staticmethod
    def get_var_from_file(file_path: str) -> np.ndarray:
        """
        Get the variables from a file.
        Parameters
        ----------
        file_path : str
            Path to the file containing the variables.

        Returns
        -------
        np.ndarray
            The variables from the file.

        Example
        -------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule:
        >>> file_path = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> Molecule.get_var_from_file(file_path).round(3)
        array([[27.   , 13.   , 26.982,  3.449,  0.256]])

        # 2 atoms in the molecule:
        >>> file_path = os.path.join(file_dir, '../../data/materials/UO2/UO2Composition')
        >>> Molecule.get_var_from_file(file_path).round(3)
        array([[1.60000e+01, 8.00000e+00, 1.59950e+01, 5.87800e+00, 0.00000e+00],
               [2.38000e+02, 9.20000e+01, 2.38051e+02, 8.62900e+00, 1.99000e-01]])
        """
        # Load the data from the file
        atomData = np.loadtxt(file_path)
        if len(atomData) % 5 == 0:
            atomData = atomData.reshape(-1, 5)
        else:
            raise ValueError("The file does not contain the correct number of rows.")
        return atomData
    @classmethod
    def from_file(cls, file_path: str) -> "Molecule":
        """
        Create a Molecule instance from a file.

        Parameters
        ----------
        file_path : str
            Path to the file containing the molecule properties.

        Returns
        -------
        Molecule
            An instance of Molecule initialized with the properties from the file.

        Example
        -------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule:
        >>> file_path = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> molecule = Molecule.from_file(file_path)
        >>> assert molecule.name == "Al27"
        >>> assert molecule.atoms["Al27"].A == 27
        >>> assert molecule.atoms["Al27"].Z == 13
        >>> assert molecule.atoms["Al27"].M == 26.98153433356103
        >>> assert molecule.atoms["Al27"].b_coh == 3.449
        >>> assert molecule.atoms["Al27"].b_incoh == 0.256

        # 2 atoms in the molecule:
        >>> file_path = os.path.join(file_dir, '../../data/materials/UO2/UO2Composition')
        >>> molecule = Molecule.from_file(file_path)
        >>> assert molecule.atoms["U238"].A == 238
        >>> assert molecule.atoms["U238"].Z == 92
        >>> assert molecule.atoms["U238"].M == 238.05077040419212
        >>> assert molecule.atoms["U238"].b_coh == 8.62912188811068
        >>> assert molecule.atoms["U238"].b_incoh == 0.19947114020071632
        >>> assert molecule.atoms["O16"].A == 16
        >>> assert molecule.atoms["O16"].Z == 8
        >>> assert molecule.atoms["O16"].M == 15.99491399021626
        >>> assert molecule.atoms["O16"].b_coh == 5.878374042670532
        >>> assert molecule.atoms["O16"].b_incoh == 0.0
        """
        # Load the data from the file
        atomData = cls.get_var_from_file(file_path)

        # Create a Molecule instance from the atom data
        return cls(A=atomData[:, 0], Z=atomData[:, 1], M=atomData[:, 2],
                   b_coh=atomData[:, 3], b_incoh=atomData[:, 4])

    @property
    def to_string(self) -> str:
        """
        Returns a string representation of the Molecule instance in the specified format.

        Returns
        -------
        str
            The string representation of the Molecule instance.

        Example
        -------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule:
        >>> file_path = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> molecule = Molecule.from_file(file_path)
        >>> print(molecule.to_string)
        # Al27 information:
        # A:
        27.0
        # Z:
        13.0
        # Atomic mass in amu:
        26.98153433356103
        # bound coherant cross section in barn:
        3.449
        # bound incoherent cross section in barn:
        0.256

        # 2 atoms in the molecule:
        >>> file_path = os.path.join(file_dir, '../../data/materials/UO2/UO2Composition')
        >>> molecule = Molecule.from_file(file_path)
        >>> print(molecule.to_string)
        # O16 information:
        # A:
        16.0
        # Z:
        8.0
        # Atomic mass in amu:
        15.99491399021626
        # bound coherant cross section in barn:
        5.878374042670532
        # bound incoherent cross section in barn:
        0.0
        <BLANKLINE>
        # U238 information:
        # A:
        238.0
        # Z:
        92.0
        # Atomic mass in amu:
        238.05077040419212
        # bound coherant cross section in barn:
        8.62912188811068
        # bound incoherent cross section in barn:
        0.19947114020071632
        """
        return "\n\n".join([atom.to_string for atom in self.atoms])

    def to_file(self, filename: str) -> None:
        """
        Write the Molecule instance to a file.

        Parameters
        ----------
        filename : str
            Path to the file where the Molecule instance will be written.

        Example
        -------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule:
        >>> import os
        >>> file_path = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> molecule = Molecule.from_file(file_path)
        >>> molecule.to_file("Al27Composition")
        >>> moleculeWritten = Molecule.from_file("Al27Composition")

        # Test the results:
        >>> assert molecule.to_string == moleculeWritten.to_string

        # Remove the file after the test:
        >>> os.remove("Al27Composition")
        """
        # Open the file in write mode and write the string
        with open(filename, 'w') as file:
            file.write(self.to_string)