from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops

from itertools import product, combinations, permutations
from collections import Counter
from collections import defaultdict
import copy
import numpy as np
import torch
import dill
import re
import signal

def read_smiles(file_path, binary = None):
    if binary is None:
        if file_path.endswith('.pkl'):
            binary = True
        else:
            binary = False

    if binary:
        with open(file_path, 'rb') as f:
            smiles = dill.load(f)
    else:
        if file_path.endswith('.sdf'):
            suppl = Chem.SDMolSupplier(file_path)
            smiles = [Chem.MolToSmiles(mol) for mol in suppl]

        with open(file_path, 'r') as f:
            smiles = [line.strip("\r\n ").split()[0] for line in f]
            
    return smiles

def save_smiles(smiles, file_path, binary = None):
    if binary is None:
        if file_path.endswith('.pkl'):
            binary = True
        else:
            binary = False

    if binary:
        with open(file_path, 'wb') as f:
            dill.dump(smiles, f)
    else:
        with open(file_path, 'w') as f:
            for smi in smiles:
                f.write(smi + '\n')

def save_mol(mols, file_path):
    writer = Chem.SDWriter(file_path)
    for mol in mols:
        writer.write(mol)
    writer.close()

def load_mol(file_path):
    suppl = Chem.SDMolSupplier(file_path)
    mols = [mol for mol in suppl]
    return mols

def set_atommap(mol, num = 0):
    for i,atom in enumerate(mol.GetAtoms(), start = num):
        atom.SetAtomMapNum(i)
    return mol

#smiles->Mol
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol

#Mol->smiles
def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles = True)

def chem_bond_to_token(bond_type):
    if bond_type == Chem.BondType.SINGLE:
        bonding_type = "-"
    elif bond_type == Chem.BondType.DOUBLE:
        bonding_type = "="
    elif bond_type == Chem.BondType.TRIPLE:
        bonding_type = "#"
    elif bond_type == Chem.BondType.AROMATIC:
        bonding_type = ':'
    else:
        raise ValueError("Invalid bond type.")
    return bonding_type

#Mol->Mol (Error->None)
def sanitize(mol, kekulize = True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

#Valence adjustment by hydrogen addition after decomposition
def add_Hs(rwmol, a1, a2, bond):
    if str(bond.GetBondType()) == 'SINGLE':
        num = 1
    elif str(bond.GetBondType()) == 'DOUBLE':
        num = 2
    elif str(bond.GetBondType()) == 'TRIPLE':
        num = 3
    elif str(bond.GetBondType()) == 'AROMATIC':
        # print("error in add_Hs 1")
        return rwmol
    else:
        print("error in add_Hs 2")
        
    for i in range(num):
        new_idx = rwmol.AddAtom(Chem.Atom(1))
        rwmol.GetAtomWithIdx(new_idx).SetAtomMapNum(0)
        rwmol.AddBond(new_idx, a1.GetIdx(), Chem.BondType.SINGLE)
        new_idx = rwmol.AddAtom(Chem.Atom(1))
        rwmol.GetAtomWithIdx(new_idx).SetAtomMapNum(0)
        rwmol.AddBond(new_idx, a2.GetIdx(), Chem.BondType.SINGLE)
    return rwmol

def get_atom_tokens(mol):
    """
    Extract tokens for individual atoms in a molecule, representing their bonding and structure.
    One bond type is prefixed for each atom token.
    
    Parameters:
        mol (Chem.Mol): The molecule to analyze (RDKit Mol object).
    
    Returns:
        list: A list of unique atom tokens in the format "<first_bonding_type><atom_symbol>".
    """
    sort_order = {'-': 0, '=': 1, '#': 2, ':': 3}

    atom_tokens = set()
    
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()  # Get the symbol of the atom (e.g., 'C', 'O')
        if atom_symbol == "H":  # Skip hydrogen
            continue
        bonding_type = "-"
        
        # Find the first bond type for this atom
        bonds = atom.GetBonds()
        bond_tokens = []
        if bonds:
            for bond in bonds:  # Process all bonds for the atom
                bond_type = bond.GetBondType()
                bonding_type = chem_bond_to_token(bond_type)
                bond_tokens.append(bonding_type)

            sort_order = {'-': 0, '=': 1, '#': 2, ':': 3}
            bond_tokens = sorted(bond_tokens, key=lambda x: sort_order[x])
            
            atom_token = [atom_symbol]
            for bond_token in bond_tokens:
                atom_token.extend([0, bond_token])
            atom_token = tuple(atom_token)
            atom_tokens.add(atom_token)
    
    # Return unique atom tokens as a sorted list
    return sorted(atom_tokens)


def atom_tokens_sort(strings):
    """
    Sort atom tokens based on specific rules:
    - Split strings into three parts using regular expressions.
    - First part: Sorted using a custom sort order.
    - Second part: Sorted based on atomic number (from element symbol).
    - Third part: Sorted using custom order or alphabetically if not in order.
    
    Parameters:
        strings (list): List of strings to sort.
    
    Returns:
        list: Sorted list of strings.
    """
    # Custom sort order for specific characters
    sort_order = {'-': 0, '=': 1, '#': 2, ':': 3}
    
    # Atomic number mapping for element symbols
    atomic_number = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        # Add more elements as needed
    }
    
    def sort_key(s):
        # Use regex to split the string into three parts
        symbol = atomic_number.get(s[0], float('inf'))
        bonds = s[2::2]
        bonds = [sort_order.get(b, float('inf')) for b in bonds]
        
        if len(bonds) < 4:
            bonds += [-1] * (4 - len(bonds))
        
        return tuple([symbol]+bonds)
    
    # Sort the strings using the custom key
    return sorted(strings, key=sort_key)


class TimeoutException(Exception):
    """Custom exception for handling timeouts."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Function execution timed out.")

# def normalize_bond_info(bond_info, frag_atom_indices):
#     """
#     Normalize the bond information by mapping atom indices based on SMILES output order.
#     This ensures equivalent representation of atom indices and bonds for comparison.
    
#     Parameters:
#         bond_info (tuple): A tuple containing SMILES string, atom indices, bond types, and bond orders.
    
#     Returns:
#         list: Normalized bond information where equivalent bonds are assigned consistent indices.
#     """
#     signal.signal(signal.SIGALRM, timeout_handler)
#     # signal.alarm(10)

#     try:
#         # Parse the molecular structure from the SMILES string
#         mol = Chem.MolFromSmiles(bond_info[0])
#         degree = (len(bond_info) - 1) // 2  # Determine the number of bonds described
#         total_atoms = mol.GetNumAtoms()

#         # Step 1: Identify target atom indices from the bond information
#         tgt_atom_indices = []
#         route_between_targets = {}
#         for i in range(degree):
#             tgt_atom_indices.append(bond_info[2 * i + 1])

#         # Step 2: Initialize dictionary to track equivalent atom indices
#         equivalent_atom_indices = {}
#         for i in range(degree):
#             atom_idx = bond_info[2 * i + 1]
#             bond_type = bond_info[2 * i + 2]
            
#             # Traverse through neighboring atoms to map equivalent routes
#             atom = mol.GetAtomWithIdx(atom_idx)
#             neighbor_level = 1
#             neighbor_routes = {0: ['']}
#             atom_position = {atom_idx: [0, 0]}  # Initialize atom positions
#             next_atoms_to_explore = [atom_idx]
#             route_between_targets[(atom_idx, atom_idx)] = ''

#             # Continue traversal until all atoms are mapped
#             while len(atom_position) < total_atoms:
#                 temp_atom_positions = {}
#                 temp_neighbor_routes = []
#                 for current_idx in next_atoms_to_explore:
#                     current_atom = mol.GetAtomWithIdx(current_idx)
#                     for neighbor in current_atom.GetNeighbors():
#                         neighbor_idx = neighbor.GetIdx()  # Get the neighbor atom's index
#                         if neighbor_idx in atom_position:
#                             continue
#                         neighbor_symbol = neighbor.GetSymbol()  # Get the neighbor atom's symbol
#                         bond = mol.GetBondBetweenAtoms(current_idx, neighbor_idx)  # Get the bond object
#                         bond_type = bond.GetBondType()  # Get the bond type

#                         # Map bond type to a numerical representation
#                         if bond_type == Chem.BondType.SINGLE:
#                             bond_type_num = 0
#                         elif bond_type == Chem.BondType.DOUBLE:
#                             bond_type_num = 1
#                         elif bond_type == Chem.BondType.TRIPLE:
#                             bond_type_num = 2
#                         elif bond_type == Chem.BondType.AROMATIC:
#                             bond_type_num = 3
#                         else:
#                             raise ValueError("Invalid bond type.")
                        
#                         # Build a route string based on bond type and atom symbol
#                         prev_level, prev_list_idx = atom_position[current_idx]
#                         route_str = neighbor_routes[prev_level][prev_list_idx] + f'_{bond_type_num}_{neighbor_symbol}'

#                         # Update positions and routes for the neighbor atom
#                         if neighbor_idx not in temp_atom_positions:
#                             temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
#                             temp_neighbor_routes.append(route_str)
#                         else:
#                             if route_str < temp_neighbor_routes[temp_atom_positions[neighbor_idx]]:
#                                 temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
#                             temp_neighbor_routes.append(route_str)
#                 neighbor_routes[neighbor_level] = temp_neighbor_routes.copy()
#                 for neighbor_idx, list_idx in temp_atom_positions.items():
#                     atom_position[neighbor_idx] = [neighbor_level, list_idx]
#                     if neighbor_idx in tgt_atom_indices:
#                         route_between_targets[(atom_idx, neighbor_idx)] = temp_neighbor_routes[list_idx]
#                 next_atoms_to_explore = list(temp_atom_positions.keys())
#                 neighbor_level += 1

#             # Step 3: Compare routes to determine equivalence
#             reference_routes = []
#             for level in range(max(neighbor_routes.keys())+1):
#                 reference_routes.append('+'.join(sorted(neighbor_routes[level])))
            
#             candidate_atom_indices = [i for i in range(total_atoms) if i != atom_idx]

#             # Check if candidate atoms have equivalent routes to the reference
#             for candi_atom_idx in reversed(candidate_atom_indices):
#                 atom = mol.GetAtomWithIdx(candi_atom_idx)
#                 neighbor_level = 1
#                 neighbor_routes = {0: ['']}
#                 atom_position = {candi_atom_idx: [0, 0]}
#                 next_atoms_to_explore = [candi_atom_idx]

#                 while len(atom_position) < total_atoms and candi_atom_idx in candidate_atom_indices:
#                     temp_atom_positions = {}
#                     temp_neighbor_routes = []
#                     for current_idx in next_atoms_to_explore:
#                         current_atom = mol.GetAtomWithIdx(current_idx)
#                         for neighbor in current_atom.GetNeighbors():
#                             neighbor_idx = neighbor.GetIdx()
#                             if neighbor_idx in atom_position:
#                                 continue
#                             neighbor_symbol = neighbor.GetSymbol()
#                             bond = mol.GetBondBetweenAtoms(current_idx, neighbor_idx)
#                             bond_type = bond.GetBondType()

#                             # Map bond type to numerical representation
#                             if bond_type == Chem.BondType.SINGLE:
#                                 bond_type_num = 0
#                             elif bond_type == Chem.BondType.DOUBLE:
#                                 bond_type_num = 1
#                             elif bond_type == Chem.BondType.TRIPLE:
#                                 bond_type_num = 2
#                             elif bond_type == Chem.BondType.AROMATIC:
#                                 bond_type_num = 3
#                             else:
#                                 raise ValueError("Invalid bond type.")
                            
#                             # Build a route string
#                             prev_level, prev_list_idx = atom_position[current_idx]
#                             route_str = neighbor_routes[prev_level][prev_list_idx] + f'_{bond_type_num}_{neighbor_symbol}'

#                             # Update positions and routes
#                             if neighbor_idx not in temp_atom_positions:
#                                 temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
#                                 temp_neighbor_routes.append(route_str)
#                             else:
#                                 if route_str < temp_neighbor_routes[temp_atom_positions[neighbor_idx]]:
#                                     temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
#                                 temp_neighbor_routes.append(route_str)
#                     neighbor_routes[neighbor_level] = temp_neighbor_routes.copy()
#                     candi_route_str = '+'.join(sorted(neighbor_routes[neighbor_level]))
#                     if candi_route_str != reference_routes[neighbor_level]:
#                         candidate_atom_indices.remove(candi_atom_idx)
#                         break
#                     for neighbor_idx, list_idx in temp_atom_positions.items():
#                         atom_position[neighbor_idx] = [neighbor_level, list_idx]
#                     next_atoms_to_explore = list(temp_atom_positions.keys())
#                     neighbor_level += 1
            
#             equivalent_atom_indices[atom_idx] = candidate_atom_indices
        
#         # Step 4: Generate normalized bond information
#         pos_indices = {}
#         routes = route_between_targets.copy()
#         for atom_idx, indices in equivalent_atom_indices.items():
#             pos_indices[atom_idx] = [atom_idx] + indices

#         total_indices = []
#         for values in pos_indices.values():
#             total_indices += values
#         total_indices = list(set(total_indices))

#         combs_cnt = 1
#         for values in pos_indices.values():
#             combs_cnt *= len(values)
#             if combs_cnt > 10000:
#                 raise ValueError("Too many combinations.")


#         combs = list(product(*[arr for arr in pos_indices.values()]))
#         combs = sorted(combs, key=lambda x: sorted(x))
        
#         success_pair = tgt_atom_indices.copy()
#         for comb in combs:
#             all_pairs = list(combinations(comb, 2))
#             all_idx_pairs = list(combinations(range(len(comb)), 2))
#             success = False
#             for i, pair in enumerate(all_pairs):
#                 # Register routes if not already present
#                 if pair not in routes:
#                     atom_idx = pair[0]
#                     tgt_atom_idx = pair[1]
#                     atom = mol.GetAtomWithIdx(atom_idx)
#                     neighbor_level = 1
#                     neighbor_routes = {0: ['']}
#                     atom_position = {atom_idx: [0, 0]}
#                     next_atoms_to_explore = [atom_idx]
#                     routes[(atom_idx, atom_idx)] = ''
#                     routes[(tgt_atom_idx, tgt_atom_idx)] = ''

#                     while len(atom_position) < total_atoms:
#                         temp_atom_positions = {}
#                         temp_neighbor_routes = []
#                         for current_idx in next_atoms_to_explore:
#                             current_atom = mol.GetAtomWithIdx(current_idx)
#                             for neighbor in current_atom.GetNeighbors():
#                                 neighbor_idx = neighbor.GetIdx()
#                                 if neighbor_idx in atom_position:
#                                     continue
#                                 neighbor_symbol = neighbor.GetSymbol()
#                                 bond = mol.GetBondBetweenAtoms(current_idx, neighbor_idx)
#                                 bond_type = bond.GetBondType()

#                                 # Map bond type to numerical representation
#                                 if bond_type == Chem.BondType.SINGLE:
#                                     bond_type_num = 0
#                                 elif bond_type == Chem.BondType.DOUBLE:
#                                     bond_type_num = 1
#                                 elif bond_type == Chem.BondType.TRIPLE:
#                                     bond_type_num = 2
#                                 elif bond_type == Chem.BondType.AROMATIC:
#                                     bond_type_num = 3
#                                 else:
#                                     raise ValueError("Invalid bond type.")
                                
#                                 # Build a route string
#                                 prev_level, prev_list_idx = atom_position[current_idx]
#                                 route_str = neighbor_routes[prev_level][prev_list_idx] + f'_{bond_type_num}_{neighbor_symbol}'

#                                 # Update positions and routes
#                                 if neighbor_idx not in temp_atom_positions:
#                                     temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
#                                     temp_neighbor_routes.append(route_str)
#                                 else:
#                                     if route_str < temp_neighbor_routes[temp_atom_positions[neighbor_idx]]:
#                                         temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
#                                     temp_neighbor_routes.append(route_str)
#                         neighbor_routes[neighbor_level] = temp_neighbor_routes.copy()
#                         for neighbor_idx, list_idx in temp_atom_positions.items():
#                             atom_position[neighbor_idx] = [neighbor_level, list_idx]
#                             if neighbor_idx in total_indices:
#                                 routes[(atom_idx, neighbor_idx)] = temp_neighbor_routes[list_idx]
#                                 routes[(neighbor_idx, atom_idx)] = temp_neighbor_routes[list_idx]
#                         next_atoms_to_explore = list(temp_atom_positions.keys())
#                         neighbor_level += 1

#                 tgt_route_str1 = routes[tgt_atom_indices[all_idx_pairs[i][0]], tgt_atom_indices[all_idx_pairs[i][1]]]
#                 tgt_route_str2 = routes[tgt_atom_indices[all_idx_pairs[i][1]], tgt_atom_indices[all_idx_pairs[i][0]]]
#                 route_str = routes[pair]
#                 if route_str == tgt_route_str1 or route_str == tgt_route_str2:
#                     continue
#                 else:
#                     break
#             else:
#                 success = True
#                 success_pair = comb
#                 break

#         # Prepare final bond information
#         final_bond_info = [bond_info[0]]
#         change_map = {}
#         temp = []
#         for i, atom_idx in enumerate(success_pair):
#             temp.append([atom_idx, bond_info[2 * i + 2]])
#             change_map[bond_info[2 * i + 1]] = atom_idx
#         temp = sorted(temp, key=lambda x: x[0])
#         for i, t in enumerate(temp):
#             final_bond_info.append(t[0])
#             final_bond_info.append(t[1])
#             # final_bond_info.append(i + 1)
#     except:
#         final_bond_info = bond_info.copy()
#         final_atom_indices = frag_atom_indices.copy()
#         return final_bond_info, final_atom_indices
#     finally:
#         signal.alarm(0)


#     # Prepare fragment atom indices
#     # Compute the adjacency matrix of the molecule
#     adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
    
#     # Use Floyd-Warshall algorithm to calculate shortest distances
#     num_atoms = mol.GetNumAtoms()
#     distance_matrix = np.full((num_atoms, num_atoms), float('inf'))
#     np.fill_diagonal(distance_matrix, 0)  # Distance from atom to itself is 0
    
#     # Set initial distances based on bonds (adjacency matrix)
#     for i in range(num_atoms):
#         for j in range(num_atoms):
#             if adjacency_matrix[i, j]:
#                 distance_matrix[i, j] = 1  # Direct bond distance is 1
    
#     # Floyd-Warshall algorithm to compute shortest paths
#     for k in range(num_atoms):
#         for i in range(num_atoms):
#             for j in range(num_atoms):
#                 distance_matrix[i, j] = min(
#                     distance_matrix[i, j], distance_matrix[i, k] + distance_matrix[k, j]
#                 )

#     return final_bond_info, frag_atom_indices


def normalize_bond_info(bond_info, frag_atom_indices):
    """
    Normalize the bond information by mapping atom indices based on SMILES output order.
    This ensures equivalent representation of atom indices and bonds for comparison.
    
    Parameters:
        bond_info (tuple): A tuple containing SMILES string, atom indices, bond types, and bond orders.
    
    Returns:
        list: Normalized bond information where equivalent bonds are assigned consistent indices.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)

    try:
        mol = Chem.MolFromSmiles(bond_info[0])
        total_atoms = mol.GetNumAtoms()

        route_matrix = [[None for _ in range(total_atoms)] for _ in range(total_atoms)]
        
        # Parse the molecular structure from the SMILES string
        degree = (len(bond_info) - 1) // 2  # Determine the number of bonds described

        # Identify target atom indices from the bond information
        tgt_atom_indices = []
        for i in range(degree):
            tgt_atom_idx = bond_info[2 * i + 1]
            tgt_atom_indices.append(tgt_atom_idx)
        
        bond_infoes = []
        for i in range(degree):
            bond_infoes.append([bond_info[2 * i + 1], bond_info[2 * i + 2]])


        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            bond_type = chem_bond_to_token(bond_type)
            s_atom_idx = bond.GetBeginAtom().GetIdx()
            e_atom_idx = bond.GetEndAtom().GetIdx()

            if route_matrix[s_atom_idx][s_atom_idx] is None:
                route_matrix[s_atom_idx][s_atom_idx] = [[str(s_atom_idx)]]

            
            if route_matrix[e_atom_idx][e_atom_idx] is None:
                route_matrix[e_atom_idx][e_atom_idx] = [[str(e_atom_idx)]]

            another_route = False
            add_routes = defaultdict(dict)
            # s_atom_idx -> e_atom_idx
            if route_matrix[s_atom_idx][e_atom_idx] is None:
                route_matrix[s_atom_idx][e_atom_idx] = [[str(s_atom_idx), bond_type, str(e_atom_idx)]]
            else:
                add_routes[s_atom_idx][e_atom_idx] = route_matrix[s_atom_idx][e_atom_idx].copy()
                add_routes[s_atom_idx][e_atom_idx].append([str(s_atom_idx), bond_type, str(e_atom_idx)])
                another_route = True
            
            # e_atom_idx -> s_atom_idx
            if route_matrix[e_atom_idx][s_atom_idx] is None:
                route_matrix[e_atom_idx][s_atom_idx] = [[str(e_atom_idx), bond_type, str(s_atom_idx)]]
            else:
                add_routes[e_atom_idx][s_atom_idx] = route_matrix[e_atom_idx][s_atom_idx].copy()
                add_routes[e_atom_idx][s_atom_idx].append([str(e_atom_idx), bond_type, str(s_atom_idx)])
                another_route = True
            
            if not another_route:
                # s_atom_idx -> e_atom_idx -> other_atom_idx
                for i in range(total_atoms):
                    if route_matrix[e_atom_idx][i] is None:
                        continue
                    
                    tmp_routes = []
                    for j, route in enumerate(route_matrix[e_atom_idx][i]):
                        # Do not extend to the root of adjacent bonds
                        if (i == s_atom_idx and len(route) == 3) or (i == e_atom_idx and len(route) == 1): 
                            continue
                        route = route.copy()
                        route.insert(0, bond_type)
                        route.insert(0, str(s_atom_idx))
                        tmp_routes.append(route)
                    if len(tmp_routes) > 0:
                        add_routes[s_atom_idx][i] = tmp_routes
                    
                # other_atom_idx -> s_atom_idx -> e_atom_idx
                for i in range(total_atoms):
                    if route_matrix[i][s_atom_idx] is None:
                        continue
                    
                    tmp_routes = []
                    for j, route in enumerate(route_matrix[i][s_atom_idx]):
                        # Do not extend to the root of adjacent bonds
                        if (i == e_atom_idx and len(route) == 3) or (i == s_atom_idx and len(route) == 1): 
                            continue
                        route = route.copy()
                        route.append(bond_type)
                        route.append(str(e_atom_idx))
                        tmp_routes.append(route)
                    if len(tmp_routes) > 0:
                        add_routes[i][e_atom_idx] = tmp_routes

                # e_atom_idx -> s_atom_idx -> other_atom_idx
                for i in range(total_atoms):
                    if route_matrix[s_atom_idx][i] is None:
                        continue
                    
                    tmp_routes = []
                    for j, route in enumerate(route_matrix[s_atom_idx][i]):
                        # Do not extend to the root of adjacent bonds
                        if (i == e_atom_idx and len(route) == 3) or (i == s_atom_idx and len(route) == 1): 
                            continue
                        route = route.copy()
                        route.insert(0, bond_type)
                        route.insert(0, str(e_atom_idx))
                        tmp_routes.append(route)
                    if len(tmp_routes) > 0:
                        add_routes[e_atom_idx][i] = tmp_routes

                # other_atom_idx -> e_atom_idx -> s_atom_idx
                for i in range(total_atoms):
                    if route_matrix[i][e_atom_idx] is None:
                        continue
                    
                    tmp_routes = []
                    for j, route in enumerate(route_matrix[i][e_atom_idx]):
                        # Do not extend to the root of adjacent bonds
                        if (i == s_atom_idx and len(route) == 3) or (i == e_atom_idx and len(route) == 1): 
                            continue
                        route = route.copy()
                        route.append(bond_type)
                        route.append(str(s_atom_idx))
                        tmp_routes.append(route)
                    if len(tmp_routes) > 0:
                        add_routes[i][s_atom_idx] = tmp_routes
            
            # Update another route
            else:
                candidate_atom_indices = set()
                for route in route_matrix[s_atom_idx][e_atom_idx]:
                    candidate_atom_indices.update(route[::2])
                candidate_atom_indices = list(candidate_atom_indices)

                for i in range(total_atoms):
                    for j in range(total_atoms):
                        if i == j:
                            continue
                        another_routes = route_matrix[i][j].copy()
                        for route in route_matrix[i][j]:
                            route_a_indices = route[::2]
                            s_idx = -1
                            s_part = []
                            for k, r in enumerate(route_a_indices):
                                if r in candidate_atom_indices:
                                    s_idx = r
                                    if k != 0:
                                        s_part = route[:2*(k-1)]
                                    break
                            if s_idx == -1:
                                continue
                            e_idx = -1
                            e_part = []
                            for k, r in enumerate(reversed(route_a_indices)):
                                if r in candidate_atom_indices:
                                    e_idx = r
                                    if k != 0:
                                        e_part = route[len(route)-2*k:]
                                    break
                            if s_atom_idx == e_idx:
                                continue

                            list1 = route_matrix[int(s_idx)][s_atom_idx]
                            list2 = route_matrix[e_atom_idx][int(e_idx)]
                            route_idx_combinations = product(list(range(len(list1))), list(range(len(list2))))
                            for c in route_idx_combinations:
                                set1 = set(s_part[::2])
                                set2 = set(list1[c[0]][::2])
                                set3 = set(list2[c[1]][::2])
                                set4 = set(e_part[::2])
                                intersection = list((set1 & set2) | (set1 & set3) | (set1 & set4) | (set2 & set3) | (set2 & set4) | (set3 & set4))
                                if len(intersection) > 0:
                                    continue
                                route = s_part + list1[c[0]] + [bond_type] + list2[c[1]] + e_part
                                another_routes.append(route)
                            
                            list1 = route_matrix[int(s_idx)][e_atom_idx]
                            list2 = route_matrix[s_atom_idx][int(e_idx)]
                            route_idx_combinations = product(list(range(len(list1))), list(range(len(list2))))
                            for c in route_idx_combinations:
                                set1 = set(s_part[::2])
                                set2 = set(list1[c[0]][::2])
                                set3 = set(list2[c[1]][::2])
                                set4 = set(e_part[::2])
                                intersection = list((set1 & set2) | (set1 & set3) | (set1 & set4) | (set2 & set3) | (set2 & set4) | (set3 & set4))
                                if len(intersection) > 0:
                                    continue
                                route = s_part + list1[c[0]] + [bond_type] + list2[c[1]] + e_part
                                another_routes.append(route)

                        if len(another_routes) > len(route_matrix[i][j]):
                            add_routes[i][j] = another_routes
            
            for i, j_routes in add_routes.items():
                for j, routes in j_routes.items():
                    routes = set([tuple(route) for route in routes])
                    route_matrix[i][j] = [list(route) for route in routes]

        
        # atom idx -> symbol
        atom_idx_symbol = {}
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()
            atom_idx_symbol[str(atom_idx)] = atom_symbol
        
        # route -> position
        atom_route_pos_symbol = {}
        atom_route_pos_symbol_with_joint = {}
        atom_route_symbol = {k1: {k2: [] for k2 in range(total_atoms)} for k1 in range(total_atoms)}
        for i in range(total_atoms):
            order1 = []
            order1_with_joint = []
            for j in range(total_atoms):
                order2 = []
                order2_with_joint = []
                for route in route_matrix[i][j]:
                    symbol_route = [atom_idx_symbol[r] if i % 2 == 0 else r for i, r in enumerate(route)]
                    symbol_route_with_joint = [
                        r if i % 2 != 0 else
                        atom_idx_symbol[r] + str(bond_infoes[tgt_atom_indices.index(int(r))][1]) if (i % 2 == 0 and int(r)) in tgt_atom_indices else
                        atom_idx_symbol[r]
                        for i, r in enumerate(route)
                    ]
                    order2.append('_'.join(symbol_route))
                    order2_with_joint.append('_'.join(symbol_route_with_joint))
                order1.append(','.join(sorted(order2)))
                order1_with_joint.append(','.join(sorted(order2_with_joint)))
                atom_route_symbol[i][j] = ','.join(sorted(order2))
            atom_route_pos_symbol[i] = '+'.join(sorted(order1))
            atom_route_pos_symbol_with_joint[i] = '+'.join(sorted(order1_with_joint))
            

        # Identify target atom indices from the bond information
        atom_route_pos_group = defaultdict(list)
        tgt_atom_route_pos_group_cnt = defaultdict(int)
        for tgt_atom_idx in tgt_atom_indices:
            tgt_atom_route_pos_group_cnt[atom_route_pos_symbol[tgt_atom_idx]] += 1
        
        for i in range(total_atoms):
            atom_route_pos_group[atom_route_pos_symbol[i]].append(i)

        tgt_group_route_str = '+'.join(sorted([atom_route_symbol[comb[0]][comb[1]] for comb in combinations(sorted(tgt_atom_indices), 2)]))


        # combination of atom indices
        lists = []
        selection_counts = []
        for key, value in tgt_atom_route_pos_group_cnt.items():
            selection_counts.append(value)
            lists.append(sorted(atom_route_pos_group[key]))
        
        # Generate combinations dynamically for each list
        select_indices_comb = []
        for i in range(len(lists)):
            if selection_counts[i] > 0:
                select_indices_comb.append(list(combinations(lists[i], selection_counts[i])))
        current_select_indices = [0] * len(select_indices_comb)

        result_list = []
        for i in range(len(current_select_indices)):
            result_list.extend(select_indices_comb[i][0])

        while True:
            ans_group_route_str = '+'.join(sorted([atom_route_symbol[comb[0]][comb[1]] for comb in list(combinations(sorted(result_list), 2))]))
            if ans_group_route_str == tgt_group_route_str:
                break
            
            min_i = -1
            result_list = None
            for i, select_indices in enumerate(current_select_indices):
                tmp = []
                if select_indices  >= len(select_indices_comb[i]) - 1:
                    continue
                for j in range(len(current_select_indices)):
                    if i == j:
                        tmp.extend(select_indices_comb[j][current_select_indices[j] + 1])
                    else:
                        tmp.extend(select_indices_comb[j][current_select_indices[j]])
                tmp = tuple(sorted(tmp))
                if min_i == -1 or tmp < result_list:
                    min_i = i
                    result_list = tmp
            if min_i == -1:
                break
            current_select_indices[min_i] += 1
        
        adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
        post_to_pre_idx = [0] * total_atoms
        pre_visited = []
        post_visited = []
        post_tgt_atom_indices = []
        tgt_pre_to_post_idx = {}

        tgt_same_pos_group = []
        for atom_group in atom_route_pos_group.values():
            tmp1 = []
            tmp2 = []
            for i in tgt_atom_indices:
                if i in atom_group:
                    tmp1.append(i)
            for i in result_list:
                if i in atom_group:
                    tmp2.append(i)
            if len(tmp1) != len(tmp2):
                raise ValueError("Invalid atom group.")
            if len(tmp1) == 1 and len(tmp2) == 1:
                tgt_pre_to_post_idx[tmp1[0]] = tmp2[0]
            else:
                tgt_same_pos_group.append((tmp1, tmp2))

        def generate_pairs_iter(candidate_pair, selected=None):
            if selected is None:
                selected = []

            # Get the length of the lists (both lists are assumed to have the same length)
            n = len(candidate_pair[0])

            # Base case: if only one pair remains, yield the final result
            if n == 1:
                selected.append((candidate_pair[0][0], candidate_pair[1][0]))
                yield selected
                return

            # Iterate through each candidate in the second list
            for i in range(len(candidate_pair[1])):
                # Select the current pair
                pair = (candidate_pair[0][0], candidate_pair[1][i])

                # Remove the selected elements from both lists
                remaining_0 = candidate_pair[0][1:]  # Remove the first element
                remaining_1 = candidate_pair[1][:i] + candidate_pair[1][i + 1:]  # Remove the selected element

                # Recursively generate pairs from the remaining elements
                yield from generate_pairs_iter((remaining_0, remaining_1), selected + [pair])

        for candidate_pair in tgt_same_pos_group:
            for pairs in generate_pairs_iter(candidate_pair):
                tmp_tgt_pre_to_post_idx = {}
                for pair in pairs:
                    tmp_tgt_pre_to_post_idx[pair[0]] = pair[1]
                    flag = True
                    for key, value in tgt_pre_to_post_idx.items():
                        if atom_route_symbol[pair[0]][key] != atom_route_symbol[pair[1]][value]:
                            flag = False
                            break
                    if not flag:
                        break
                else:
                    for key, value in tmp_tgt_pre_to_post_idx.items():
                        tgt_pre_to_post_idx[key] = value
                    break
            pass

        post_ori_idx = result_list[0]
        for key, value in tgt_pre_to_post_idx.items():
            post_to_pre_idx[value] = key
            if value == post_ori_idx:
                next_idx = [(key, post_ori_idx)]
                pre_visited.append(key)
                post_visited.append(post_ori_idx)
        post_tgt_atom_indices = [tgt_pre_to_post_idx[i] for i in tgt_atom_indices]

        # route -> post position
        post_atom_route_pos_symbol_with_joint = {}
        for i in range(total_atoms):
            order1_with_joint = []
            for j in range(total_atoms):
                order2_with_joint = []
                for route in route_matrix[i][j]:
                    symbol_route_with_joint = [
                        r if i % 2 != 0 else
                        atom_idx_symbol[r] + str(bond_infoes[post_tgt_atom_indices.index(int(r))][1]) if (i % 2 == 0 and int(r)) in result_list else
                        atom_idx_symbol[r]
                        for i, r in enumerate(route)
                    ]
                    order2_with_joint.append('_'.join(symbol_route_with_joint))
                order1_with_joint.append(','.join(sorted(order2_with_joint)))
            post_atom_route_pos_symbol_with_joint[i] = '+'.join(sorted(order1_with_joint))
        
        while len(pre_visited) < total_atoms:
            new_next_idx = []
            for pre_idx, post_idx in next_idx:
                pre_neighbor_indices = np.where(adjacency_matrix[pre_idx, :] == 1)[0]
                pre_neighbor_indices = [i for i in pre_neighbor_indices if i not in pre_visited]
                post_neighbor_indices = np.where(adjacency_matrix[post_idx, :] == 1)[0]
                post_neighbor_indices = [i for i in post_neighbor_indices if i not in post_visited]

                for pre_n_idx in pre_neighbor_indices:
                    for post_n_idx in post_neighbor_indices:
                        if atom_route_pos_symbol_with_joint[pre_n_idx] == post_atom_route_pos_symbol_with_joint[post_n_idx]:
                        # if atom_route_pos_symbol[pre_n_idx] == atom_route_pos_symbol[post_n_idx]:
                            post_to_pre_idx[post_n_idx] = pre_n_idx
                            pre_visited.append(pre_n_idx)
                            post_visited.append(post_n_idx)
                            new_next_idx.append((pre_n_idx, post_n_idx))
                            post_neighbor_indices.remove(post_n_idx)
                            break
            next_idx = new_next_idx
        
        new_atom_indices = [frag_atom_indices[post_to_pre_idx[i]] for i in range(total_atoms)]

        # Prepare final bond information
        final_bond_info = [bond_info[0]]
        temp = [(post_to_pre_idx.index(bond_idx), bond_type) for bond_idx, bond_type in zip(bond_info[1::2], bond_info[2::2])]
        temp = sorted(temp, key=lambda x: x[0])
        for t in temp:
            final_bond_info.append(t[0])
            final_bond_info.append(t[1])

        final_bond_info = tuple(final_bond_info)
        print(f'{frag_atom_indices} -> {new_atom_indices}')
        print(f'{bond_info} -> {final_bond_info}')
        print()
        
    except Exception as e:
        final_bond_info = bond_info.copy()
        final_atom_indices = frag_atom_indices.copy()
        return final_bond_info, final_atom_indices
    finally:
        signal.alarm(0)

    return final_bond_info, new_atom_indices


def calculate_advanced_scores(smiles_or_mol: str, sort=False):
    """
    Calculate advanced scores for each atom in a molecule based on:
    - Symbol score (atomic type)
    - Join type score (bond type)
    - Distance from molecule center
    """
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    elif isinstance(smiles_or_mol, Chem.Mol):
        mol = smiles_or_mol
    else:
        raise ValueError("Invalid input type. Expected SMILES string or RDKit Mol object.")
    
    if mol is None:
        raise ValueError(f"Invalid mol")

    total_atoms = mol.GetNumAtoms()
    distance_matrix = rdmolops.GetDistanceMatrix(mol)

    # Updated symbol_scores with Carbon having the highest score
    symbol_scores = {
        "C": 2.0, "N": 1.5, "O": 1.4, "S": 1.3, "P": 1.2, 
        "F": 1.1, "Cl": 1.0, "Br": 0.9, "I": 0.8, 
        "Si": 0.7, "B": 0.6, "Li": 0.5, "Na": 0.4, 
        "K": 0.3, "Mg": 0.2, "Ca": 0.1
    }

    bond_scores = {Chem.BondType.SINGLE: 1.0, Chem.BondType.DOUBLE: 1.5, Chem.BondType.TRIPLE: 2.0, Chem.BondType.AROMATIC: 2.5}

    scores = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        if symbol == "H":  # Skip hydrogen
            continue
        max_distance = distance_matrix[atom_idx].max()  # Average distance
        symbol_score = symbol_scores.get(symbol, 0.1)  # Default to minimum score for unknown atoms

        # Calculate join type score
        join_type_score = sum(
            bond_scores.get(bond.GetBondType(), 0) for bond in atom.GetBonds()
        )

        # Calculate the final score
        atom_score = sum(
            ((max_distance - dist) / max_distance / join_type_score * symbol_score) / total_atoms
            for dist in distance_matrix[atom_idx]
        )
        scores.append((atom_idx, symbol, atom_score))
    
    if sort:
        scores = sorted(scores, key=lambda x: x[2], reverse=True)
    return scores