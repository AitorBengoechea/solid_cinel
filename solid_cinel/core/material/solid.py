"""
Python file for working with the solid structure.

@author: AB272525
"""
from solid_cinel.core.material.material_composition import Molecule
from solid_cinel.core.material.crystal_symmetry import CrystalStructure
from solid_cinel.core.material.pdos import Pdos
from solid_cinel.core.cinematic.frames import Neutron
import numpy as np
import pandas as pd
import numba as nb
import collections
from typing import Iterable, Union, Dict, List
from scipy.constants import c
from scipy.constants import physical_constants as const
from scipy.optimize import minimize
from math import pi, cos, sin, acos, exp
from io import StringIO

collections.Callable = collections.abc.Callable

# Constants:
h = const["reduced Planck constant in eV s"][0]
m_to_eV = const["atomic mass unit-electron volt relationship"][0]
mn_to_MeV = const["neutron mass energy equivalent in MeV"][0]
kb = const["Boltzmann constant in eV/K"][0]
BfacUnitChange = (4 * c ** 2 * pi**2) * h ** 2 / (m_to_eV * kb)
BraggUnitChange = 1.0e20 * h ** 2 * c ** 2 / (mn_to_MeV * 1.0e6)

# Output style:
PosCol = pd.Index(["x", "y", "z"])

class Solid(CrystalStructure, Molecule):
    """
    Class for the solid structure. It is a combination of the crystal structure
    and the molecule class with the addition of the preferred orientation and
    the unit position of the atoms in the unit cell.
    """
    def __init__(self, unit_pos: Union[dict, Iterable], length: Iterable[float],
                 angles: Iterable[float], preferred_orientation: Iterable,
                 A: Iterable[int], Z: Iterable[int], M: Iterable[float], b_coh: Iterable[float],
                 b_incoh: Iterable[float], pdos: [Pdos, dict[Pdos]] = None,
                 name: str = None):
        """
        Initialize the solid class

        Parameters
        ----------
        unit_pos : Union[dict, Iterable]
            Position of the atoms in the unit cell.
        length : Iterable[float]
            direct vector lengths
        angles : Iterable[float]
            direct vector angles in degrees
        preferred_orientation : Iterable
            Preferred orientation of the solid.
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
        # Initialize the parent classes
        CrystalStructure.__init__(self, length, angles, preferred_orientation)
        Molecule.__init__(self, A=A, Z=Z, M=M, b_coh=b_coh, b_incoh=b_incoh,
                          name=name)

        # Set the atom position in the unit cell:
        self.set_unit_pos(unit_pos)

        # Set the pdos information if it is available:
        if pdos is not None:
            self.set_pdos(pdos)

    def set_unit_pos_from_dict(self, unit_pos: Dict) -> Dict:
        """
        Set the position of the atoms in the unit cell from a dictionary.

        Parameters
        ----------
        unit_pos : Dict
            Position of the atoms in the unit cell.

        Returns
        -------
        "Dict"
            Dictionary containing the position of the atoms in the unit cell.
        """
        return {
            element: pd.DataFrame(single_unit_pos.reshape(-1, 3), columns=PosCol)
            for element, single_unit_pos in unit_pos.items()
        }

    def set_unit_pos_from_iterable(self, unit_pos: Iterable) -> Dict:
        """
        Set the position of the atoms in the unit cell from an iterable. The
        iterable must be in the order of the atoms in the Molecule class.

        Parameters
        ----------
        unit_pos : Iterable
            Position of the atoms in the unit cell.

        Returns
        -------
        "Dict"
            Dictionary containing the position of the atoms in the unit cell.
        """
        # Get the atoms name in the molecule:
        atoms = self.atoms.index

        # Get the number of atoms type in the molecule:
        Natoms = self.atomNum

        # unit pos a 3D array with the shape (Natoms, NatomInUnitCell, 3)
        if Natoms == 1:
            unit_pos = np.array(unit_pos)
            if len(unit_pos.shape) == 1:
                unit_pos = unit_pos[np.newaxis, ::]
            else:
                unit_pos = unit_pos[np.newaxis, ::, ::]

        # Transfor into a dictionary containing dataframes for the position:
        return {
            atoms[i]: pd.DataFrame(unit_pos[i].reshape(-1, 3), columns=PosCol)
            for i in range(Natoms)
        }

    def set_unit_pos(self, unit_pos: Union[Dict, Iterable]):
        """
        Set the position of the atoms in the unit cell.

        Parameters
        ----------
        unit_pos : Union[dict, Iterable]
            Position of the atoms in the unit cell.
        """
        # Transform the unit_pos into a dictionary of dataframes:
        if isinstance(unit_pos, dict):
            _unit_pos = self.set_unit_pos_from_dict(unit_pos)

        elif hasattr(unit_pos, '__len__'):
            _unit_pos = self.set_unit_pos_from_iterable(unit_pos)

        else:
            raise ValueError("The unit_pos must be a dictionary or an iterable.")

        # Set the unit position:
        self.unit_pos = pd.Series(_unit_pos)

    def set_pdos(self, multiplePdos: Union[Pdos, Iterable, Dict]):
        """
        Set the pdos information of the atoms that form the solid.

        Parameters
        ----------
        pdosDict : Iterable, Dict, Pdos
            The pdos information of the atoms that form the solid. If an Iterable
            is given, the pdos information is set for each atom in the order of
            the atoms in the solid. If a dictionary is given, the pdos information
            is set for each atom in the dictionary. If a Pdos object is given, the
            pdos information is set for all the atoms in the solid.
        """
        atoms = self.atoms.index
        if isinstance(multiplePdos, dict):
            multiplePdos = [multiplePdos.get(element) for element in atoms]
        self.pdos = pd.Series(multiplePdos, index=atoms)

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
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 2 atoms in the molecule: UO2
        >>> compositon_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Composition')
        >>> structure_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Structure')
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/UO2/UO2AtomPos')
        >>> UO2 = Solid.from_files(compositon_file, structure_file, atomPos_file)
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
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule: Al27
        >>> compositon_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> structure_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Structure')
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/Al27/Al27AtomPos')
        >>> Al = Solid.from_files(compositon_file, structure_file, atomPos_file)
        >>> assert Al.atom_number == 1

        # 2 atoms in the molecule: UO2
        >>> compositon_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Composition')
        >>> structure_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Structure')
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/UO2/UO2AtomPos')
        >>> UO2 = Solid.from_files(compositon_file, structure_file, atomPos_file)
        >>> assert UO2.atom_number == 12
        """
        return sum(self.atom_pos.apply(lambda x: x.shape[0]).values)

    @staticmethod
    def get_var_from_file(file_path: str) -> List:
        """
        Read structured data from a file separated by empty lines and return a
        list of numpy arrays.

        Parameters
        ----------
        file_path : str
            Path to the file containing the structured data.

        Returns
        -------
        list
            A list of numpy arrays, each corresponding to a section of the file
            separated by empty lines.

        Example
        -------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/UO2/UO2AtomPos')
        >>> atomPos = Solid.get_var_from_file(atomPos_file)
        >>> atomPos[0]
        array([[0.25, 0.25, 0.25],
               [0.75, 0.25, 0.25],
               [0.25, 0.75, 0.75],
               [0.75, 0.75, 0.75],
               [0.75, 0.25, 0.75],
               [0.25, 0.25, 0.75],
               [0.75, 0.75, 0.25],
               [0.25, 0.75, 0.25]])

        >>> atomPos[1]
        array([[0.5, 0. , 0. ],
               [0.5, 0.5, 0.5],
               [0. , 0. , 0.5],
               [0. , 0.5, 0. ]])
        """
        # Read the file and return as string:
        with open(file_path, 'r') as file:
            file_str = file.read()

        # Split the string by the empty line
        atomsPosInUnitCellStr = file_str.strip().split('\n\n')

        # Get the data from the string:
        atomsPosInUnitCell = []
        for SingleAtomPosInUnitCell in atomsPosInUnitCellStr:
            # Use StringIO to create file-like objects
            SingleAtomPosInUnitCell_io = StringIO(SingleAtomPosInUnitCell)

            # Read the data from the file-like object
            SingleAtomPosInUnitCell_array = np.genfromtxt(SingleAtomPosInUnitCell_io,
                                                    comments="#")

            # Append the data to the list
            atomsPosInUnitCell.append(SingleAtomPosInUnitCell_array)

        # Return the data: (Natoms, NatomInUnitCell, 3)
        return atomsPosInUnitCell

    @classmethod
    def from_files(cls, compositon_file: str, structure_file: str,
                   atomPos_file: str) -> "Solid":
        """
        Create a Solid object from the data in the files.

        Parameters
        ----------
        compositon_file : str
            Path to the file containing the composition data.
        structure_file : str
            Path to the file containing the crystal structure data.
        atomPos_file : str
            Path to the file containing the atom position data.

        Returns
        -------
        "Solid"
            A Solid object created from the data in the files.

        Example
        -------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule: Al27
        >>> compositon_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> structure_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Structure')
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/Al27/Al27AtomPos')
        >>> Al = Solid.from_files(compositon_file, structure_file, atomPos_file)
        >>> assert isinstance(Al, Solid)

        # 2 atoms in the molecule: UO2
        >>> compositon_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Composition')
        >>> structure_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Structure')
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/UO2/UO2AtomPos')
        >>> UO2 = Solid.from_files(compositon_file, structure_file, atomPos_file)
        >>> assert isinstance(UO2, Solid)
        """
        # Get the parent class data from the files:
        atomData = Molecule.get_var_from_file(compositon_file)
        A, Z, M, b_coh, b_incoh = atomData[:, 0], atomData[:, 1], atomData[:, 2], atomData[:, 3], atomData[:, 4]
        crystalData = CrystalStructure.get_var_from_file(structure_file)
        vectorLength, vectorAngles, preferredOrientation = crystalData[0], crystalData[1], crystalData[2]

        # Get the unit cell data from the file:
        unitCellData = cls.get_var_from_file(atomPos_file)
        return cls(unitCellData, vectorLength, vectorAngles, preferredOrientation,
                   A, Z, M, b_coh, b_incoh)

    def get_Bfact(self, T: float, pdosDict: dict[Pdos] = None,
                  anstrom: bool = True) -> [float, pd.Series]:
        """
        Calculate mean square displacement for a certain pdos information.
        .. math::
            B_j= \frac{4\pi^2\hbar^2}{M_jk_BT}\lambda_s

        Parameters
        ----------
        T : 'float'
            Temperature in K
        pdosDict : 'dict[Pdos]', optional
            Dictionary with the pdos for each element. The default is None.
        anstrom : 'bool', optional
            Option to obtain the B unit in A^2. The default is True.

        Returns
        -------
        "float" or "pd.Series"
            Mean square displacement.

        Examples
        --------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule: Al27
        >>> compositon_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> structure_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Structure')
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/Al27/Al27AtomPos')
        >>> Al = Solid.from_files(compositon_file, structure_file, atomPos_file)
        >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
        >>> pdosAl27 = Pdos.from_dE(rho_in_energy, interv_in_energy)

        # 2 atoms in the molecule: UO2
        >>> compositon_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Composition')
        >>> structure_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Structure')
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/UO2/UO2AtomPos')
        >>> UO2 = Solid.from_files(compositon_file, structure_file, atomPos_file)
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy, interv_in_energy
        >>> pdosUO2 = {"O16": Pdos.from_dE(rho_in_energy[0], interv_in_energy[0]), "U238": Pdos.from_dE(rho_in_energy[1], interv_in_energy[1])}

        Test the results:
        >>> T = 20
        >>> float(Al.get_Bfact(T, pdosAl27)["Al27"].round(6))
        0.274871

        >>> T = 80
        >>> float(Al.get_Bfact(T, pdosAl27)["Al27"].round(6))
        0.337081

        >>> T = 296
        >>> UO2.get_Bfact(T, pdosUO2).round(6)
        O16     0.468604
        U238    0.253845
        dtype: float64

        >>> T = 400
        >>> UO2.get_Bfact(T, pdosUO2).round(6)
        O16     0.595531
        U238    0.340297
        dtype: float64
        """
        # Check if the pdosDict is defined or initialized in the object
        if pdosDict is not None:
            self.set_pdos(pdosDict)
        elif self.pdos is None:
            raise ValueError("The Pdos information must be defined or initialized in the object.")

        # Get the Debye Waller coefficient for each atom for the give temperature:
        DebyeWallerCoeff = self.pdos.apply(lambda x: x.fix_T(T).DebyeWallerCoeff)

        # Get the mass of the atoms in the unit cell:
        Matom = self.atoms.apply(lambda x: x.M)

        # Calculate the B factor for each atom:
        Bfact = BfacUnitChange * DebyeWallerCoeff / (T * Matom)
        return Bfact * 1.0e20 if anstrom else Bfact

    def get_multiplicity(self, energyCut: float, T: float,
                         precision: list = [6, 6], d_min: float = None,
                         **kwargs) -> pd.DataFrame:
        """
        Obtain hkl data for the solid in a certain temperature and for a neutron
        certain energy filtering with the multiplicity.

        Parameters
        ----------
        T : 'float'
            Temperature in K
        energyCut : 'float'
            Energy limit for d espace limit in eV
        precision: ['int', 'int'], optional
            Precision to get the multiplicity for d_hkl and Fsq_hkl. The
            default is [6, 6].
        d_min : 'float', optional
            Minimum d espace to calculate the multiplicity. The default is None.

        Parameters for get_Bfact
        ------------------------
        pdosDict : 'dict[Pdos]', optional
            Dictionary with the pdos for each element. The default is None.
        anstrom : 'bool', optional
            Option to obtain the B unit in A^2. The default is True.

        Returns
        -------
        "pd.DataFrame", (N, 4)
            Multiplicity, d, Fsq and orientation of bragg planes.

        Examples
        --------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule: Al27
        >>> compositon_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> structure_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Structure')
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/Al27/Al27AtomPos')
        >>> Al = Solid.from_files(compositon_file, structure_file, atomPos_file)

        # Set the pdos information:
        >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
        >>> Al.set_pdos(Pdos.from_dE(rho_in_energy, interv_in_energy))

        Test the results:
        >>> T = 20
        >>> energyCut = 2.301
        >>> multiplicity = Al.get_multiplicity(energyCut, T)
        >>> multiplicity.shape[0]
        678
        >>> multiplicity.iloc[:10] #doctest: +NORMALIZE_WHITESPACE
                      d       Fsq  Orientation angle  Multiplicity
        h k l
        1 1 0  2.019999  0.115016         125.264390           6.0
            1  2.332494  0.115989          70.528779           8.0
        2 1 1  1.428355  0.111207          90.000000          12.0
          2 0  1.010000  0.103962         125.264390           6.0
            1  1.218106  0.108433         100.024988          24.0
            2  1.166247  0.107523          70.528779           8.0
        3 2 1  0.903371  0.100519         104.963217          24.0
            2  0.926839  0.101369          82.388621          24.0
          3 2  0.824661  0.097189          90.000000          24.0
            3  0.777498  0.094765          70.528779          32.0

        >>> multiplicity.round(6).iloc[667:677] #doctest: +NORMALIZE_WHITESPACE
                         d  Fsq  Orientation angle  Multiplicity
        h  k  l
        31 19 18  0.091254  0.0          87.009863         168.0
              19  0.090999  0.0          84.777020         384.0
           20 17  0.090884  0.0          90.000000         336.0
              18  0.090815  0.0          87.768637         552.0
              19  0.090609  0.0          85.544023         288.0
           21 15  0.089911  0.0          95.160350         384.0
              16  0.090157  0.0          92.954150         168.0
              17  0.090269  0.0          90.739152         216.0
              18  0.090247  0.0          88.521943         192.0
              19  0.090090  0.0          86.309150         168.0
        """
        # Get the dmin for the multiplicity calculation:
        if d_min is None:
            d_min = Neutron(energyCut).d_min

        # Get the hkl data for the solid:
        hkl_data = numba_hkl_data(d_min,
                                  self.reciproc_vec.values,
                                  self.get_Bfact(T, **kwargs),
                                  self.atom_pos,
                                  self.atoms.apply(lambda x: x.b_coh),
                                  self.preferred_orientation.values,
                                  precision)

        # Return the hkl data in the appropiate format:
        return hkl_data.sort_values(by=["h", "k", "l"]).set_index(["h", "k", "l"])

    def get_BraggEdges(self, energyCut: float, T: float,
                       xs: bool = True, difracAngles: bool = True,
                       precision: List = [6, 6], d_min: float = None,
                       pddf_kind: str = None, pddf_val: str = None,
                       threshold: float = 1.e-30) -> pd.DataFrame:
        """
        Get BraggEdges.
        .. math::
            E_n=\dfrac{h^2}{2m\lambda^2}=\dfrac{h^2}{2m(2d_{hkl})^2}

        Parameters
        ----------
        xs : 'bool', optional
            Option to get the xs related with Bragg Edges. The default is
            True
        theta : 'bool', optional
            Option to get the difraction angles vs hkl data. The default is
            True
        file_BraggEdges : 'str', optional
            Name of the file to write Bragg Edges data. The default is False.

        Parameters for get_multiplicity
        -------------------------------
        T : 'float'
            Temperature in K
        energyCut : 'float'
            Energy cut for d espace in eV
        precision: ['int', 'int'], optional
            Precision to get the multiplicity for d_hkl and Fsq_hkl. The
            default is [6, 6].
        d_min : 'float', optional
            Minimum d espace to calculate the multiplicity. The default is None.

        Parameters for get_ppdf
        -----------------------
        pddf_kind : 'str', optional
            key to calculate PDDF. The default is None. Options:
                - march_dollase
                - altomare
                - cvc
        pddf_val : 1D iterable, optional
            Value to calculate PDDF. The default is None. Options:
                - march-dollase : shape(1, 1)
                - altomare: shape(1, 2)
                - cvc: shape(1, 2)

        Parameters for get_BraggEdges_Xs
        --------------------------------
        threshold : 'float', optional
            Minimun value of xs. The default is 1.e-30.

        Returns
        -------
        "pd.DataFrame", (N, M)
            DataFrame with the selected information.

        Example
        -------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule: Al27
        >>> compositon_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> structure_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Structure')
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/Al27/Al27AtomPos')
        >>> Al = Solid.from_files(compositon_file, structure_file, atomPos_file)

        # Set the pdos information:
        >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
        >>> Al.set_pdos(Pdos.from_dE(rho_in_energy, interv_in_energy))

        Test the results:
        >>> T = 20
        >>> energyCut = 2.301
        >>> Al.get_BraggEdges(energyCut, T).round(6).iloc[:10, :4]
                      d       Fsq  Orientation angle  Multiplicity
        h k l
        1 1 1  2.332494  0.115989                0.0           8.0
            0  2.019999  0.115016                0.0           6.0
        2 1 1  1.428355  0.111207                0.0          12.0
          2 1  1.218106  0.108433                0.0          24.0
            2  1.166247  0.107523                0.0           8.0
            0  1.010000  0.103962                0.0           6.0
        3 2 2  0.926839  0.101369                0.0          24.0
            1  0.903371  0.100519                0.0          24.0
          3 2  0.824661  0.097189                0.0          24.0
            3  0.777498  0.094765                0.0          32.0

        >>> Al.get_BraggEdges(energyCut, T).round(6).iloc[:10, 4::]
                      E  PDDF        Xs      theta
        h k l
        1 1 1  0.003759   1.0  0.005370   4.632867
            0  0.005012   1.0  0.003459   5.350060
        2 1 1  0.010024   1.0  0.004729   7.568882
          2 1  0.013783   1.0  0.007865   8.877727
            2  0.015036   1.0  0.002489   9.273329
            0  0.020048   1.0  0.001563  10.711827
        3 2 2  0.023807   1.0  0.005595  11.676144
            1  0.025060   1.0  0.005407  11.980567
          3 2  0.030072   1.0  0.004773  13.128861
            3  0.033831   1.0  0.005850  13.929091
        """
        # Get multiplicity
        multiplicity = self.get_multiplicity(energyCut, T, precision=precision,
                                             d_min=d_min)

        # Get Bragg Edges energy:
        multiplicity["E"] = pi ** 2 * BraggUnitChange
        multiplicity["E"] /= 2 * multiplicity["d"] ** 2

        # Sort the data by energy
        multiplicity = multiplicity.sort_values(by=["E"])

        # Optional argument:
        # Xs:
        if xs:
            # Get PDDF:
            add_pddfToMultiplicity(multiplicity, pddf_kind, pddf_val)

            # Get Bragg Edges Xs:
            add_BraggEdgesXsToMultiplicity(multiplicity, self.unitCellVol,
                                           self.atom_number, threshold)

        # difraction angles:
        if difracAngles:
            add_difracAnglesToMultiplicity(multiplicity, energyCut)

        return multiplicity

    def get_XsCoh(self, energyCut: float, T: float, precision: List = [6, 6],
                  d_min: float = None, pddf_kind: str = None,
                  pddf_val: str = None, threshold: float = 1.e-30) -> pd.Series:
        """
        Get coherent Xs.

        Parameters
        ----------
        energyCut : 'float'
            energy cut-off, above which the peak will not be stored.
        file_Xs : 'str', optional
            Name of the file to write Xs the data. The default is False.

        Parameters for get_BraggEdges
        -----------------------------
        theta : 'bool', optional
            Option to get the difraction angles vs hkl data. The default is
            True
        file_BraggEdges : 'str', optional
            Name of the file to write Bragg Edges data. The default is False.

        Parameters for get_multiplicity
        -------------------------------
        T : 'float'
            Temperature in K
        precision: ['int', 'int'], optional
            Precision to get the multiplicity for d_hkl and Fsq_hkl. The
            default is [6, 6].
        d_min : 'float', optional
            Minimum d espace to calculate the multiplicity. The default is None.

        Parameters for get_ppdf
        -----------------------
        pddf_kind : 'str', optional
            key to calculate PDDF. The default is None. Options:
                - march_dollase
                - altomare
                - cvc
        pddf_val : 1D iterable, optional
            Value to calculate PDDF. The default is None. Options:
                - march-dollase : shape(1, 1)
                - altomare: shape(1, 2)
                - cvc: shape(1, 2)

        Parameters for get_BraggEdges_Xs
        --------------------------------
        threshold : 'float', optional
            Minimun value of xs. The default is 1.e-30.

        Returns
        -------
        "pd.DataFrame", (N, M)
            Dataframe with the coherent xs for each atom of the objetc.

        Examples
        --------
        >>> import os
        >>> file_dir = os.path.dirname(os.path.abspath(__file__))

        # 1 atom in the molecule: Al27
        >>> compositon_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
        >>> structure_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Structure')
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/Al27/Al27AtomPos')
        >>> Al = Solid.from_files(compositon_file, structure_file, atomPos_file)

        # Set the pdos information:
        >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
        >>> Al.set_pdos(Pdos.from_dE(rho_in_energy, interv_in_energy))

        # 2 atoms in the molecule: UO2
        >>> compositon_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Composition')
        >>> structure_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Structure')
        >>> atomPos_file = os.path.join(file_dir, '../../data/materials/UO2/UO2AtomPos')
        >>> UO2 = Solid.from_files(compositon_file, structure_file, atomPos_file)

        # Set the pdos information:
        >>> from solid_cinel.data.examples.UO2 import rho_in_energy, interv_in_energy
        >>> pdosUO2 = [Pdos.from_dE(rho_in_energy[0], interv_in_energy[0]),  Pdos.from_dE(rho_in_energy[1], interv_in_energy[1])]
        >>> UO2.set_pdos(pdosUO2)

        Test the results:
        >>> T = 20
        >>> energyCut = 2.301
        >>> Al.get_XsCoh(energyCut, T).round(6).iloc[:10] #doctest: +NORMALIZE_WHITESPACE
        E
        0.003759    1.428610
        0.005012    1.761554
        0.010024    1.352587
        0.013783    1.554352
        0.015036    1.590366
        0.020048    1.270746
        0.023807    1.305106
        0.025060    1.455627
        0.030072    1.371732
        0.033831    1.392236
        Name: Xs, dtype: float64

        >>> UO2.get_XsCoh(energyCut, T).round(6).iloc[:10] #doctest: +NORMALIZE_WHITESPACE
        E
        0.000664    0.000000
        0.001329    0.000000
        0.001993    3.049080
        0.002658    2.327766
        0.003322    1.862213
        0.003987    1.551844
        0.005316    6.027579
        0.005980    5.357848
        0.006645    4.822063
        0.007309    5.677412
        Name: Xs, dtype: float64
        """
        # Get the Bragg Edges
        BraggEdgesXs = self.get_BraggEdges(energyCut, T, difracAngles=False,
                                           d_min=d_min, precision=precision,
                                           pddf_kind=pddf_kind, pddf_val=pddf_val,
                                           threshold=threshold)

        # Extract the energy and cross-section information and convert to a Series
        xsCoh = BraggEdgesXs[["E", "Xs"]].set_index("E")["Xs"]

        # Filter out data where energy exceeds the cut-off
        xsCoh = xsCoh[xsCoh.index <= energyCut]

        # Sum the data if there are duplicate energy values
        if xsCoh.index.has_duplicates:
            xsCoh = xsCoh.groupby(level=0).sum()

        # Calculate the cumulative sum of the cross-sections and normalize by energy
        xsCoh = xsCoh.cumsum() / xsCoh.index.values

        return xsCoh


def add_pddfToMultiplicity(multiplicity: pd.DataFrame, kind: str = None,
                           pddf_val: str = None) -> pd.DataFrame:
    """
    Add to the hkl data dataframe the Pole Density Distribution Function.
    March-dollase:
        .. math::
            \mathcal{P}_{hkl}(\Theta)=(P_1^2\cos^2(\Theta)+P_1^{-1}\sin^2(\Theta))^{-3/2}.
    Altomare:
        .. math::
            \mathcal{P}_{hkl}(\Theta)=\exp(P_1\cos(2\Theta))
    cvc:
        .. math::
            \mathcal{P}_{hkl}(\Theta)=\exp(-P_1(1-\cos^{P_2}(\Theta)))

    Parameters
    ----------
    multiplicity: 'pd.DataFrame', (N, M)
        Frame with hkl data.
    kind : 'str', optional
        key to calculate PDDF. The default is None. Options:
            - march_dollase
            - altomare
            - cvc
    pddf_val : 1D iterable, optional
        Value to calculate PDDF. The default is None. Options:
            - march-dollase : shape(1, 1)
            - altomare: shape(1, 2)
            - cvc: shape(1, 2)

    Returns
    -------
    "pd.DataFrame", (N, M + 1)
        Pole Density Distribution Function column

    Example
    -------
    Object initialization:
    >>> import os
    >>> file_dir = os.path.dirname(os.path.abspath(__file__))
    >>> compositon_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
    >>> structure_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Structure')
    >>> atomPos_file = os.path.join(file_dir, '../../data/materials/Al27/Al27AtomPos')
    >>> Al = Solid.from_files(compositon_file, structure_file, atomPos_file)
    >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
    >>> Al.set_pdos(Pdos.from_dE(rho_in_energy, interv_in_energy))
    >>> T = 20
    >>> energyCut = 2.301
    >>> multiplicity = Al.get_multiplicity(energyCut, T)
    >>> multiplicity.iloc[:10]
                  d       Fsq  Orientation angle  Multiplicity
    h k l
    1 1 0  2.019999  0.115016         125.264390           6.0
        1  2.332494  0.115989          70.528779           8.0
    2 1 1  1.428355  0.111207          90.000000          12.0
      2 0  1.010000  0.103962         125.264390           6.0
        1  1.218106  0.108433         100.024988          24.0
        2  1.166247  0.107523          70.528779           8.0
    3 2 1  0.903371  0.100519         104.963217          24.0
        2  0.926839  0.101369          82.388621          24.0
      3 2  0.824661  0.097189          90.000000          24.0
        3  0.777498  0.094765          70.528779          32.0

    Test the results:
    >>> add_pddfToMultiplicity(multiplicity).iloc[:10]
                  d       Fsq  Orientation angle  Multiplicity  PDDF
    h k l
    1 1 0  2.019999  0.115016                0.0           6.0   1.0
        1  2.332494  0.115989                0.0           8.0   1.0
    2 1 1  1.428355  0.111207                0.0          12.0   1.0
      2 0  1.010000  0.103962                0.0           6.0   1.0
        1  1.218106  0.108433                0.0          24.0   1.0
        2  1.166247  0.107523                0.0           8.0   1.0
    3 2 1  0.903371  0.100519                0.0          24.0   1.0
        2  0.926839  0.101369                0.0          24.0   1.0
      3 2  0.824661  0.097189                0.0          24.0   1.0
        3  0.777498  0.094765                0.0          32.0   1.0

    >>> multiplicity = Al.get_multiplicity(energyCut, T)
    >>> add_pddfToMultiplicity(multiplicity, kind='march-dollase', pddf_val=2).iloc[:10]
                  d       Fsq  Orientation angle  Multiplicity      PDDF
    h k l
    1 1 0  2.019999  0.115016         125.264390           6.0  0.464758
        1  2.332494  0.115989          70.528779           8.0  1.193243
    2 1 1  1.428355  0.111207          90.000000          12.0  2.828427
      2 0  1.010000  0.103962         125.264390           6.0  0.464758
        1  1.218106  0.108433         100.024988          24.0  2.119463
        2  1.166247  0.107523          70.528779           8.0  1.193243
    3 2 1  0.903371  0.100519         104.963217          24.0  1.592384
        2  0.926839  0.101369          82.388621          24.0  2.377318
      3 2  0.824661  0.097189          90.000000          24.0  2.828427
        3  0.777498  0.094765          70.528779          32.0  1.193243

    >>> multiplicity = Al.get_multiplicity(energyCut, T)
    >>> add_pddfToMultiplicity(multiplicity, kind='altomare', pddf_val=[1, 1]).iloc[:10]
                  d       Fsq  Orientation angle  Multiplicity      PDDF
    h k l
    1 1 0  2.019999  0.115016         125.264390           6.0  1.716531
        1  2.332494  0.115989          70.528779           8.0  1.459426
    2 1 1  1.428355  0.111207          90.000000          12.0  1.367879
      2 0  1.010000  0.103962         125.264390           6.0  1.716531
        1  1.218106  0.108433         100.024988          24.0  1.390865
        2  1.166247  0.107523          70.528779           8.0  1.459426
    3 2 1  0.903371  0.100519         104.963217          24.0  1.420350
        2  0.926839  0.101369          82.388621          24.0  1.381017
      3 2  0.824661  0.097189          90.000000          24.0  1.367879
        3  0.777498  0.094765          70.528779          32.0  1.459426

    >>> multiplicity = Al.get_multiplicity(energyCut, T)
    >>> add_pddfToMultiplicity(multiplicity, kind='cvc', pddf_val=[1, 1]).iloc[:10]
                  d       Fsq  Orientation angle  Multiplicity      PDDF
    h k l
    1 1 0  2.019999  0.115016         125.264390           6.0  0.206522
        1  2.332494  0.115989          70.528779           8.0  0.513417
    2 1 1  1.428355  0.111207          90.000000          12.0  0.367879
      2 0  1.010000  0.103962         125.264390           6.0  0.206522
        1  1.218106  0.108433         100.024988          24.0  0.309104
        2  1.166247  0.107523          70.528779           8.0  0.513417
    3 2 1  0.903371  0.100519         104.963217          24.0  0.284165
        2  0.926839  0.101369          82.388621          24.0  0.419981
      3 2  0.824661  0.097189          90.000000          24.0  0.367879
        3  0.777498  0.094765          70.528779          32.0  0.513417
    """
    orientation_angle = np.deg2rad(multiplicity.loc[:, "Orientation angle"])

    if kind is None:
        multiplicity["Orientation angle"] = 0.
        multiplicity["PDDF"] = 1.

    elif kind.lower() == 'march-dollase' and isinstance(pddf_val, (int, float)):
        value = (pddf_val * np.cos(orientation_angle)) ** 2
        value += np.sin(orientation_angle) ** 2 / pddf_val
        multiplicity["PDDF"] = value ** (-1.5)

    elif kind.lower() == 'altomare' and len(pddf_val) == 2:
        value = pddf_val[0] * np.cos(2 * orientation_angle)
        multiplicity["PDDF"] = np.exp(value) + pddf_val[1]

    elif kind.lower() == 'cvc' and len(pddf_val) == 2:
        value = pddf_val[0] * (1 - np.cos(orientation_angle) ** pddf_val[1])
        multiplicity["PDDF"] = np.exp(- value)

    else:
        ValueError("Introduced kind is not available")
    return multiplicity

def add_difracAnglesToMultiplicity(multiplicity: pd.DataFrame,
                                   energyCut: float) -> pd.DataFrame:
    """
    Add to the hkl data dataframe the difraction angles(ª) vs hkl data.
    .. math::
        2\theta_{hkl}=\arccos\left(1-\dfrac{\pi^2\hbar^2}{md_{hkl}^2E}\right)

    Parameters
    ----------
    multiplicity: 'pd.DataFrame', (N, M)
        Frame with hkl data.
    energyCut : 'float'
        Energy cut for d espace in eV

    Returns
    -------
    "pd.DataFrame", (N, M + 1)
        Difraction angle column.

    Example
    -------
    Object initialization:
    >>> import os
    >>> file_dir = os.path.dirname(os.path.abspath(__file__))
    >>> compositon_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
    >>> structure_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Structure')
    >>> atomPos_file = os.path.join(file_dir, '../../data/materials/Al27/Al27AtomPos')
    >>> Al = Solid.from_files(compositon_file, structure_file, atomPos_file)
    >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
    >>> Al.set_pdos(Pdos.from_dE(rho_in_energy, interv_in_energy))
    >>> T = 20
    >>> energyCut = 2.301
    >>> multiplicity = Al.get_multiplicity(energyCut, T)
    >>> multiplicity.iloc[:10] #doctest: +NORMALIZE_WHITESPACE
                  d       Fsq  Orientation angle  Multiplicity
    h k l
    1 1 0  2.019999  0.115016         125.264390           6.0
        1  2.332494  0.115989          70.528779           8.0
    2 1 1  1.428355  0.111207          90.000000          12.0
      2 0  1.010000  0.103962         125.264390           6.0
        1  1.218106  0.108433         100.024988          24.0
        2  1.166247  0.107523          70.528779           8.0
    3 2 1  0.903371  0.100519         104.963217          24.0
        2  0.926839  0.101369          82.388621          24.0
      3 2  0.824661  0.097189          90.000000          24.0
        3  0.777498  0.094765          70.528779          32.0

    Test the results:
    >>> add_difracAnglesToMultiplicity(multiplicity, energyCut).iloc[:10]
                  d       Fsq  Orientation angle  Multiplicity     theta
    h k l
    1 1 0  2.019999  0.115016         125.264390           6.0   5.350060
        1  2.332494  0.115989          70.528779           8.0   4.632867
    2 1 1  1.428355  0.111207          90.000000          12.0   7.568882
      2 0  1.010000  0.103962         125.264390           6.0  10.711827
        1  1.218106  0.108433         100.024988          24.0   8.877727
        2  1.166247  0.107523          70.528779           8.0   9.273329
    3 2 1  0.903371  0.100519         104.963217          24.0  11.980567
        2  0.926839  0.101369          82.388621          24.0  11.676144
      3 2  0.824661  0.097189          90.000000          24.0  13.128861
        3  0.777498  0.094765          70.528779          32.0  13.929091

    """
    d = multiplicity["d"]
    angle_value = np.clip(1 - pi ** 2 * BraggUnitChange / (d ** 2 * energyCut),
                          -1, 1)
    multiplicity["theta"] = np.rad2deg(np.arccos(angle_value))
    return multiplicity


def add_BraggEdgesXsToMultiplicity(multiplicity: pd.DataFrame, unitCellVol: float,
                                   atom_number: int, threshold: float = 1.e-30) -> pd.DataFrame:
    """
    Add to the hkl data dataframe the cross section related with the
    Bragg Edges.
    .. math::
        \sigma_{\textrm{coh}}^{\textrm{el}}(E_{hkl})=\dfrac{\pi^2\hbar^2}{mN_{uc}V_{uc}E_{hkl}}M_{hkl}d_{hkl}\left|F(\vec{\tau}_{hkl})\right|^2\mathcal{P}_{hkl}(\Theta_{hkl})

    Parameters
    ----------
    multiplicity: 'pd.DataFrame', (N, M)
        Frame with hkl data.
    unitCellVol : 'float'
        Unit cell volume.
    atom_number : 'int'
        The number of atoms in the unit cell.
    threshold : 'float', optional
        Minimun value of xs. The default is 1.e-30.

    Returns
    -------
    "pd.DataFrame", (N, M + 1)
        Xs related to the Bragg Edges.

    Example
    -------
    Object initialization:
    >>> import os
    >>> file_dir = os.path.dirname(os.path.abspath(__file__))
    >>> compositon_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
    >>> structure_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Structure')
    >>> atomPos_file = os.path.join(file_dir, '../../data/materials/Al27/Al27AtomPos')
    >>> Al = Solid.from_files(compositon_file, structure_file, atomPos_file)
    >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
    >>> Al.set_pdos(Pdos.from_dE(rho_in_energy, interv_in_energy))
    >>> T = 20
    >>> energyCut = 2.301
    >>> unit_cell_vol = Al.unitCellVol
    >>> atom_number = Al.atom_number

    # Calculate Multiplicity, PDDF and Xs:
    >>> multiplicity = Al.get_multiplicity(energyCut, T)
    >>> multiplicity = add_pddfToMultiplicity(multiplicity)
    >>> multiplicity = add_BraggEdgesXsToMultiplicity(multiplicity, unit_cell_vol, atom_number)

    Test the results:
    >>> multiplicity.loc[::, "Xs"].iloc[:10]
    h  k  l
    1  1  0    0.003459
          1    0.005370
    2  1  1    0.004729
       2  0    0.001563
          1    0.007865
          2    0.002489
    3  2  1    0.005407
          2    0.005595
       3  2    0.004773
          3    0.005850
    Name: Xs, dtype: float64
    """
    # Check if the PDDF is defined in the multiplicity dataframe
    if "PDDF" not in multiplicity.columns:
        raise ValueError("The PDDF column is not defined in the multiplicity dataframe.")

    # Calculate the cross section related with the Bragg Edges:
    multiplicity["Xs"] = multiplicity["d"] * multiplicity["Fsq"]
    multiplicity["Xs"] *= multiplicity["Multiplicity"] * multiplicity["PDDF"]
    multiplicity["Xs"] *= BraggUnitChange * pi ** 2
    multiplicity["Xs"] /= unitCellVol * atom_number

    # Set the threshold value
    if threshold:
        multiplicity.loc[multiplicity["Xs"] < threshold, "Xs"] = 0.0

    return multiplicity


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
    >>> import os
    >>> file_dir = os.path.dirname(os.path.abspath(__file__))

    # 1 atom in the molecule:
    >>> file_path = os.path.join(file_dir, '../../data/materials/Al27/Al27Structure')
    >>> crys = CrystalStructure.from_file(file_path)
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


def numba_hkl_data(d_min: float, rec_vecs: np.ndarray, Bfac: pd.Series,
                   pos: pd.Series, csl: pd.Series, preferred_orientation: pd.Series,
                   precision: list) -> pd.DataFrame:
    """
    Obtain hkl data for the solid in a certain temperature and for a neutron
    certain energy.
    2 atoms test in test folder.

    Parameters
    ----------
    d_min : 'float'
        The minimum dspacing for the LEAPR module of NJOY
    rec_vecs : 'np.ndarray', (3, 3)
        Reciprocal vectors
    Bfac : 'pd.Series'
        Pandas series with the B factor for TargetMaterial object elements.
    pos : 'pd.Series'
        Pandas series with atomic position of elements in TargetMaterial
        object.
    csl : 'pd.Series'
        Coherent elastic length for each element of TargetMaterial object.
    precision: 'list', (2,):
        Array containing:
            0: Precision to reagroup in multiplicity the d_hkl
            1: Precision to reagroup in multiplicity the Fsq_hkl

    Returns
    -------
    "pd.DataFrame"
        Dataframe containing the hkl information.

    Examples
    --------
    Object initialization:
    >>> import os
    >>> file_dir = os.path.dirname(os.path.abspath(__file__))

    # 1 atom in the molecule: Al27
    >>> compositon_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Composition')
    >>> structure_file = os.path.join(file_dir, '../../data/materials/Al27/Al27Structure')
    >>> atomPos_file = os.path.join(file_dir, '../../data/materials/Al27/Al27AtomPos')
    >>> Al = Solid.from_files(compositon_file, structure_file, atomPos_file)

    # Set the pdos information:
    >>> from solid_cinel.data.examples.Al27 import rho_in_energy, interv_in_energy
    >>> Al.set_pdos(Pdos.from_dE(rho_in_energy, interv_in_energy))

    # 2 atoms in the molecule: UO2
    >>> compositon_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Composition')
    >>> structure_file = os.path.join(file_dir, '../../data/materials/UO2/UO2Structure')
    >>> atomPos_file = os.path.join(file_dir, '../../data/materials/UO2/UO2AtomPos')
    >>> UO2 = Solid.from_files(compositon_file, structure_file, atomPos_file)

    # Set the pdos information:
    >>> from solid_cinel.data.examples.UO2 import rho_in_energy, interv_in_energy
    >>> pdosUO2 = [Pdos.from_dE(rho_in_energy[0], interv_in_energy[0]),  Pdos.from_dE(rho_in_energy[1], interv_in_energy[1])]
    >>> UO2.set_pdos(pdosUO2)

    Test the results:
    >>> T = 20
    >>> E = 2.301
    >>> recs_vec = Al.reciproc_vec.values
    >>> d_min = Neutron(E).d_min
    >>> B = Al.get_Bfact(T)
    >>> pos = Al.atom_pos
    >>> csl = Al.atoms.apply(lambda x: x.b["b_coh"])
    >>> precision = np.array([6, 6])
    >>> preferred_orientation = Al.preferred_orientation.values
    >>> hkl_data = numba_hkl_data(d_min, recs_vec, B, pos, csl, preferred_orientation, precision)
    >>> hkl_data.shape[0]
    678
    >>> hkl_data.round(6).iloc[:10]
        h   k   l         d  Fsq  Orientation angle  Multiplicity
    0  31  21  20  0.089800  0.0          84.107323         336.0
    1  31  21  19  0.090090  0.0          86.309150         168.0
    2  31  21  18  0.090247  0.0          88.521943         192.0
    3  31  21  17  0.090269  0.0          90.739152         216.0
    4  31  21  16  0.090157  0.0          92.954150         168.0
    5  31  21  15  0.089911  0.0          95.160350         384.0
    6  31  20  19  0.090609  0.0          85.544023         288.0
    7  31  20  18  0.090815  0.0          87.768637         552.0
    8  31  20  17  0.090884  0.0          90.000000         336.0
    9  31  19  19  0.090999  0.0          84.777020         384.0

    >>> T = 296
    >>> E = 6.85e-1
    >>> recs_vec = UO2.reciproc_vec.values
    >>> d_min = Neutron(E).d_min
    >>> B = UO2.get_Bfact(T)
    >>> pos = UO2.atom_pos
    >>> csl = UO2.atoms.apply(lambda x: x.b["b_coh"])
    >>> precision = np.array([6, 6])
    >>> preferred_orientation = UO2.preferred_orientation.values
    >>> hkl_data = numba_hkl_data(d_min, recs_vec, B, pos, csl, preferred_orientation, precision)
    >>> hkl_data.round(6).iloc[:10]
        h  k  l         d       Fsq  Orientation angle  Multiplicity
    0  33  7  2  0.164168  0.000000          86.607081         216.0
    1  33  7  1  0.164384  0.108687          88.302052         384.0
    2  33  7  0  0.164456  0.000000          90.000000         144.0
    3  33  6  4  0.164240  0.000000          83.199199         288.0
    4  33  6  3  0.164746  0.000000          84.888910         432.0
    5  33  6  2  0.165110  0.000000          86.587580         192.0
    6  33  6  1  0.165330  0.000000          88.292276         264.0
    7  33  6  0  0.165404  0.000000          90.000000         432.0
    8  33  5  4  0.165037  0.000000          83.166021         528.0
    9  33  5  3  0.165551  0.116100          84.863872         120.0
    """
    # hkl_max calculation
    hkl_max = hkl_max_value(rec_vecs, d_min)

    # Preparation of variables to be accept in numba nopython mode:
    keys, values = nb.core.types.unicode_type, nb.core.types.float64
    Bfac_ = nb.typed.Dict.empty(key_type=keys, value_type=values)
    pos_ = nb.typed.Dict.empty(key_type=keys, value_type=values[:, :])
    csl_ = nb.typed.Dict.empty(key_type=keys, value_type=values)
    for element in Bfac.index:
        Bfac_[element], csl_[element] = Bfac[element], csl[element]
        pos_[element] = pos[element].values
    preferred_orientation_ = np.array(preferred_orientation, dtype=float)

    # Execute numba
    hkl_data_dict = hklloop(d_min, hkl_max, rec_vecs, Bfac_, pos_, csl_,
                            preferred_orientation_, np.array(precision))

    # Order the output
    columns = ["h", "k", "l", "d", "Fsq", "Orientation angle", "Multiplicity"]
    values = [[h, k, l, d_hkl, Fsq_hkl, orientation, mul] for (h, k, l), [d_hkl, Fsq_hkl, orientation, mul] in hkl_data_dict.items()]
    return pd.DataFrame(values, columns=columns)


@nb.jit(nopython=True, cache=True)
def hklloop(d_min: float, hkl_max: np.ndarray, rec_vecs: np.ndarray,
            Bfac: dict, pos: dict, csl: dict, preferred_orientation: np.ndarray,
            precision: np.ndarray) -> dict:
    """
    Get the F_hkl and d_hkl for all the posible h, k, l plane combination that
    fill the condition of d_hkl > d_min.
    .. math::
        d_{hkl} = \frac{2\pi}{\tau_{hkl}}
        F(\vec{\tau}_{hkl})=\sum_{j=1}^{N_{uc}}b_j\exp\left(-\dfrac{\hbar^2\tau_{hkl}^2}{4M_jk_BT}\Lambda_j(T)\right) e^{i\vec{\tau}_{hkl}\cdot\vec{p}_j}

    Parameters
    ----------
    d_min : 'float'
        The minimum dspacing for the LEAPR module of NJOY
    hkl_max : 'np.ndarray', (3,)
        Maximun h, k, l index for generating a d > d_min
    rec_vecs : 'np.ndarray' (3, 3)
        Reciprocal vectors
    Bfac : 'nb.typed.Dict'
        Dict with the B factor for TargetMaterial object elements.
    pos : 'nb.typed.Dict'
        Dict with atomic position of elements in TargetMaterial object.
    csl : 'nb.typed.Dict'
        Coherent elastic length for each element of TargetMaterial object.
    preferred_orientation: "np.ndarray", (3)
        Array with the preferred orientation of the solid.
    precision: "float"
        Precision of the rounding in the calculation to merge different plane
        values

    Returns
    -------
    "dict"
        Dictionary containing the hkl planes, the d_hkl, Fsq, orientation_angle.
    """
    # Get the output variables:
    hklM = {}
    hkldF = {}

    # Get the orientation norm:
    orientation_norm = np.linalg.norm(preferred_orientation)

    # Loop over the hkl planes:
    h_range, k_range, l_range = [np.arange(-x, x + 1) for x in hkl_max]

    # Loop over the hkl planes:
    for h in h_range[::-1]:  # to get positive hkl order
        for k in k_range[::-1]:
            for l in l_range[::-1]:
                if h ** 2 + k ** 2 + l ** 2 == 0:  # (0, 0, 0) is excluded
                    continue

                # Get the d_hkl and Fsq_hkl
                vec_tau_hkl = h * rec_vecs[0] + k * rec_vecs[1] + l * rec_vecs[2]
                d_hkl = 2 * np.pi / np.linalg.norm(vec_tau_hkl)

                # Check if d_hkl is greater than d_min
                if d_hkl < d_min:
                    continue

                # Get the Fsq_hkl squared
                Fsq = Fsq_hkl(vec_tau_hkl, Bfac, csl, pos)

                # same dspacing and Fsquared with precision will be regrouped
                d_rnd, Fsq_rnd = round(d_hkl, precision[0]), round(Fsq, precision[1])

                # Write the output
                if (d_rnd, Fsq_rnd) in hkldF:
                    hklM[hkldF[(d_rnd, Fsq_rnd)]][-1] += 1
                else:
                    hkldF[(d_rnd, Fsq_rnd)] = (h, k, l)
                    OA_num = np.sum(vec_tau_hkl * preferred_orientation)
                    OA_den = np.linalg.norm(vec_tau_hkl) * orientation_norm
                    hklM[(h, k, l)] = np.array([d_hkl, Fsq,
                                                acos(OA_num / OA_den) * 180 / pi,
                                                1])
    return hklM


@nb.jit(nopython=True, cache=True)
def Fsq_hkl(vec_tau_hkl: np.ndarray, Bfac: dict, csl: dict, pos: dict) -> float:
    """
    Get F_hkl:
    .. math::
        F(\vec{\tau}_{hkl})=\sum_{j=1}^{N_{uc}}b_j\exp\left(-\dfrac{\hbar^2\tau_{hkl}^2}{4M_jk_BT}\Lambda_j(T)\right) e^{i\vec{\tau}_{hkl}\cdot\vec{p}_j}

    Parameters
    ----------
    vec_tau_hkl : 'np.ndarray', (3, 3)
        Reciprocal vectors
    Bfac : 'nb.typed.Dict'
        Dict with the B factor for TargetMaterial object elements.
    pos : 'nb.typed.Dict'
        Dict with atomic position of elements in TargetMaterial object.
    csl : 'nb.typed.Dict'
        Coherent elastic length for each element of TargetMaterial object.

    Returns
    -------
    "float"
        Fsq_hkl value for that (h, k, l) plane
    """
    real = 0.
    imag = 0.
    constant = - 0.5 * np.linalg.norm(vec_tau_hkl) ** 2 / (8 * pi ** 2)
    for element in Bfac:
        for iep in range(len(pos[element])):
            real += cos(np.sum(vec_tau_hkl * pos[element][iep]))
            imag += sin(np.sum(vec_tau_hkl * pos[element][iep]))
        expon_hkl = exp(constant * Bfac[element])
        real *= csl[element] * 0.1 * expon_hkl
        imag *= csl[element] * 0.1 * expon_hkl
    return real ** 2 + imag ** 2
