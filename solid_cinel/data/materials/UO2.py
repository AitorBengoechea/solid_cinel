import numpy as np

# Material information for U238:
A_U238 = 238
Z_U238 = 92
atom_mass_U238 = 238.05077040419212
b_coh_U238 = 8.62912188811068
b_incoh_U238 = 0.19947114020071632

# Material information for O16:
A_O16 = 16
Z_O16 = 8
atom_mass_O16 = 15.99491399021626
b_coh_O16 = 5.878374042670532
b_incoh_O16 = 0.0

# Material information:
A = [A_O16, A_U238]
Z = [Z_O16, Z_U238]
atom_mass = [atom_mass_O16, atom_mass_U238]
b_coh = [b_coh_O16, b_coh_U238]
b_incoh = [b_incoh_O16, b_incoh_U238]

# Unit cell information:
a = 5.54781
dir_vec_length = [a, a, a]
dir_vec_angles = [90, 90, 90]
preferred_orientation = [0, 0, 1]

# Unit cell information for U238 in UO2:
unit_pos_U_str = '''
0.500000  0.000000  0.000000
0.500000  0.500000  0.500000
0.000000  0.000000  0.500000
0.000000  0.500000  0.000000'''
unit_pos_U = np.fromstring(unit_pos_U_str, dtype=np.float64, sep=' ')\
               .reshape(-1, 3)

# Unit cell information for O16 in UO2:
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

# Energy information:
energy_sup = 5.  # eV
energy_cut = 6.85e-1