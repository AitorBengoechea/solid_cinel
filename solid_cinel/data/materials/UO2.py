import numpy as np

# Structural information for the material:
preferred_orientation = [0, 0, 1]
unit_pos_U_str = '''
0.500000  0.000000  0.000000
0.500000  0.500000  0.500000
0.000000  0.000000  0.500000
0.000000  0.500000  0.000000'''
unit_pos_U = np.fromstring(unit_pos_U_str, dtype=np.float64, sep=' ')\
               .reshape(-1, 3)
unit_pos_O_str = '''
0.250000  0.250000  0.250000
0.750000  0.250000  0.250000
0.250000  0.750000  0.750000
0.750000  0.750000  0.750000
0.750000  0.250000  0.750000
0.250000  0.250000  0.750000
0.750000  0.750000  0.250000
0.250000  0.750000  0.250000'''
unit_pos_O = np.fromstring(unit_pos_O_str, dtype=np.float64, sep=' ')\
               .reshape(-1, 3)
unit_pos = {"O16": unit_pos_O, "U238": unit_pos_U}
a = 5.54781
dir_vec_length = [a, a, a]
dir_vec_angles = [90, 90, 90]
energy_sup = 5.  # eV
energy_cut = 6.85e-1
A = [16, 238]
Z = [8, 92]
atom_mass = [15.99491399021626, 238.05077040419212]
b_coh = [5.878374042670532, 8.62912188811068]
b_incoh = [0.0, 0.19947114020071632]
