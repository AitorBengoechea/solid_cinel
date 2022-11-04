# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:56:43 2022

@author: AB272525
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class File_manager():
    T: float = None
    A : int = None
    Z : int = None
    preferred_orientation : list[float] = field(default_factory=list)
    dir_vec_length : list[float] = field(default_factory=list)
    dir_vec_angles : list[float] = field(default_factory=list)
    unit_pos: list[float] = field(default_factory=list)
    atom_mass: float = None
    b : dict[float] = field(default_factory=dict)
    interval_energy : float = None
    rho : list[float] = field(default_factory=list)