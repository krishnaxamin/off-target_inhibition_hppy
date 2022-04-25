from pandas import read_csv, concat
from math import sqrt

"""
Script to calculate the RMSD between the original ligand in the PDB crystal structure and the re-docked ligand. 
The pdbqt file for the re-docked ligand has been altered to include only rows with information about the atoms in Model 
1, not including rows detailing branches.
"""

docked = read_csv(r'C:\Users\krish\Documents\Cambridge\2021-2022\SysBio\Project\docking\rigid_outputs\6G2_out_py.pdbqt', sep='\s+', header=None)
original = read_csv(r'C:\Users\krish\Documents\Cambridge\2021-2022\SysBio\Project\structures\ligands\pdb\6G2.pdb', sep='\s+', header=None)
original = original.iloc[:26, :]

docked_info = concat([docked.iloc[:, 2], docked.iloc[:, 6:9]], axis=1)
docked_info.columns = ['atom', 'docked_x', 'docked_y', 'docked_z']

original_info = concat([original.iloc[:, 2], original.iloc[:, 6:9]], axis=1)
original_info.columns = ['atom', 'original_x', 'original_y', 'original_z']

coords = docked_info.merge(original_info, on='atom')
square_distances = []
for i in range(len(coords)):
    x1 = coords.iloc[i, 1]
    y1 = coords.iloc[i, 2]
    z1 = coords.iloc[i, 3]
    x2 = coords.iloc[i, 4]
    y2 = coords.iloc[i, 5]
    z2 = coords.iloc[i, 6]
    square_distance = (x2-x1) ** 2 + (y2-y1) ** 2 + (z2-z1) ** 2
    square_distances.append(square_distance)

coords['square_distances'] = square_distances

rmsd = sqrt(sum(square_distances)/len(square_distances))
