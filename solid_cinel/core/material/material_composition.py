from solid_cinel.data.elements import ELEMENTS
from scipy.constants import physical_constants as const
from urllib.request import urlopen, Request, urlretrieve
from io import StringIO
import numpy as np
import pandas as pd
import collections
from typing import Iterable
import pytest
collections.Callable = collections.abc.Callable


class Atom:
    """Class to store properties of the atoms."""

    def __init__(self, A: int , Z: int, atom_mass: float,
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
        self.b = {"b_coh": b_coh,
                  "b_incoh": b_incoh}

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
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Atom(A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

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
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Atom(A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

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
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Atom(A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

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
        >>> A = 27
        >>> Z = 13
        >>> atomic_mass_Al27 = 26.98153433356103
        >>> b_coh_Al27  = 3.449
        >>> b_incoh_Al27 = 0.256
        >>> Al = Atom(A, Z, atomic_mass_Al27, b_coh_Al27, b_incoh_Al27)

        Test the results:
        >>> assert np.double(Al.freeXs).round(6) == np.double(1.396702)
        """
        A = self.atom_mass / const["neutron mass in u"][0]
        return self.boundXs * (A / (A + 1)) ** 2


class Molecule(Atom):
    """
    Class to store the properties and method for all the atom of
    the molecule
    """

    def __init__(self, A: Iterable[int] | int, Z: Iterable[int] | int,
                 atom_mass: Iterable[int] | float, b_coh: Iterable[int] | float,
                 b_incoh: Iterable[int] | float, name: str=None):
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
        Name of the molecule given by the user

        Returns
        -------
        "str"
            Name of the molecule.

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
    def name(self, name:str):
        self._name = name if name is not None else "Molecule"

def get_str_from_html(html) -> str:
    """
    Get a the str from a html page.

    Parameters
    ----------
    html : 'str'
        The html page.

    """
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {'User-Agent': user_agent}
    request = Request(html, None, headers)
    response = urlopen(request)
    return response.read().decode("utf-8")

def get_atom_data(A, Z, source) -> pd.DataFrame:
    """
    Get atom data from internet.

    Parameters
    ----------
    A : 'int'
        Atomic number.
    Z : 'int'
        Number of protons.
    source : _type_
        _description_

    Examples
    --------
    """
    if source.lower() == "nist":
        html = f"https://www.ncnr.nist.gov/resources/n-lengths/elements/{ELEMENTS[Z].lower()}.html"
        data_str = get_str_from_html(html)
        data_str = data_str[data_str.find("cross sections<tr>") + 19:]
        data_str = data_str[:data_str.find("</table>")]
        data_str = data_str.replace("<th>", "").replace("<td>", "").replace("---", "0.0").replace("<tr>", "")
        StringData = StringIO(data_str)
        data =pd.read_csv(StringData, sep ="\t\t", engine="python")
    return data