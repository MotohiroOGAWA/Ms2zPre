from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops

from itertools import product, combinations
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
                if bond_type == Chem.BondType.SINGLE:
                    bonding_type = "-"
                elif bond_type == Chem.BondType.DOUBLE:
                    bonding_type = "="
                elif bond_type == Chem.BondType.TRIPLE:
                    bonding_type = "#"
                elif bond_type == Chem.BondType.AROMATIC:
                    bonding_type = ':'
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

def normalize_bond_info(bond_info):
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
        # Parse the molecular structure from the SMILES string
        mol = Chem.MolFromSmiles(bond_info[0])
        degree = (len(bond_info) - 1) // 2  # Determine the number of bonds described
        total_atoms = mol.GetNumAtoms()

        # Step 1: Identify target atom indices from the bond information
        tgt_atom_indices = []
        route_between_targets = {}
        for i in range(degree):
            tgt_atom_indices.append(bond_info[2 * i + 1])

        # Step 2: Initialize dictionary to track equivalent atom indices
        equivalent_atom_indices = {}
        for i in range(degree):
            atom_idx = bond_info[2 * i + 1]
            bond_type = bond_info[2 * i + 2]
            
            # Traverse through neighboring atoms to map equivalent routes
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbor_level = 1
            neighbor_routes = {0: ['']}
            atom_position = {atom_idx: [0, 0]}  # Initialize atom positions
            next_atoms_to_explore = [atom_idx]
            route_between_targets[(atom_idx, atom_idx)] = ''

            # Continue traversal until all atoms are mapped
            while len(atom_position) < total_atoms:
                temp_atom_positions = {}
                temp_neighbor_routes = []
                for current_idx in next_atoms_to_explore:
                    current_atom = mol.GetAtomWithIdx(current_idx)
                    for neighbor in current_atom.GetNeighbors():
                        neighbor_idx = neighbor.GetIdx()  # Get the neighbor atom's index
                        if neighbor_idx in atom_position:
                            continue
                        neighbor_symbol = neighbor.GetSymbol()  # Get the neighbor atom's symbol
                        bond = mol.GetBondBetweenAtoms(current_idx, neighbor_idx)  # Get the bond object
                        bond_type = bond.GetBondType()  # Get the bond type

                        # Map bond type to a numerical representation
                        if bond_type == Chem.BondType.SINGLE:
                            bond_type_num = 0
                        elif bond_type == Chem.BondType.DOUBLE:
                            bond_type_num = 1
                        elif bond_type == Chem.BondType.TRIPLE:
                            bond_type_num = 2
                        elif bond_type == Chem.BondType.AROMATIC:
                            bond_type_num = 3
                        else:
                            raise ValueError("Invalid bond type.")
                        
                        # Build a route string based on bond type and atom symbol
                        prev_level, prev_list_idx = atom_position[current_idx]
                        route_str = neighbor_routes[prev_level][prev_list_idx] + f'_{bond_type_num}_{neighbor_symbol}'

                        # Update positions and routes for the neighbor atom
                        if neighbor_idx not in temp_atom_positions:
                            temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
                            temp_neighbor_routes.append(route_str)
                        else:
                            if route_str < temp_neighbor_routes[temp_atom_positions[neighbor_idx]]:
                                temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
                            temp_neighbor_routes.append(route_str)
                neighbor_routes[neighbor_level] = temp_neighbor_routes.copy()
                for neighbor_idx, list_idx in temp_atom_positions.items():
                    atom_position[neighbor_idx] = [neighbor_level, list_idx]
                    if neighbor_idx in tgt_atom_indices:
                        route_between_targets[(atom_idx, neighbor_idx)] = temp_neighbor_routes[list_idx]
                next_atoms_to_explore = list(temp_atom_positions.keys())
                neighbor_level += 1

            # Step 3: Compare routes to determine equivalence
            reference_routes = []
            for level in range(max(neighbor_routes.keys())+1):
                reference_routes.append('+'.join(sorted(neighbor_routes[level])))
            
            candidate_atom_indices = [i for i in range(total_atoms) if i != atom_idx]

            # Check if candidate atoms have equivalent routes to the reference
            for candi_atom_idx in reversed(candidate_atom_indices):
                atom = mol.GetAtomWithIdx(candi_atom_idx)
                neighbor_level = 1
                neighbor_routes = {0: ['']}
                atom_position = {candi_atom_idx: [0, 0]}
                next_atoms_to_explore = [candi_atom_idx]

                while len(atom_position) < total_atoms and candi_atom_idx in candidate_atom_indices:
                    temp_atom_positions = {}
                    temp_neighbor_routes = []
                    for current_idx in next_atoms_to_explore:
                        current_atom = mol.GetAtomWithIdx(current_idx)
                        for neighbor in current_atom.GetNeighbors():
                            neighbor_idx = neighbor.GetIdx()
                            if neighbor_idx in atom_position:
                                continue
                            neighbor_symbol = neighbor.GetSymbol()
                            bond = mol.GetBondBetweenAtoms(current_idx, neighbor_idx)
                            bond_type = bond.GetBondType()

                            # Map bond type to numerical representation
                            if bond_type == Chem.BondType.SINGLE:
                                bond_type_num = 0
                            elif bond_type == Chem.BondType.DOUBLE:
                                bond_type_num = 1
                            elif bond_type == Chem.BondType.TRIPLE:
                                bond_type_num = 2
                            elif bond_type == Chem.BondType.AROMATIC:
                                bond_type_num = 3
                            else:
                                raise ValueError("Invalid bond type.")
                            
                            # Build a route string
                            prev_level, prev_list_idx = atom_position[current_idx]
                            route_str = neighbor_routes[prev_level][prev_list_idx] + f'_{bond_type_num}_{neighbor_symbol}'

                            # Update positions and routes
                            if neighbor_idx not in temp_atom_positions:
                                temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
                                temp_neighbor_routes.append(route_str)
                            else:
                                if route_str < temp_neighbor_routes[temp_atom_positions[neighbor_idx]]:
                                    temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
                                temp_neighbor_routes.append(route_str)
                    neighbor_routes[neighbor_level] = temp_neighbor_routes.copy()
                    candi_route_str = '+'.join(sorted(neighbor_routes[neighbor_level]))
                    if candi_route_str != reference_routes[neighbor_level]:
                        candidate_atom_indices.remove(candi_atom_idx)
                        break
                    for neighbor_idx, list_idx in temp_atom_positions.items():
                        atom_position[neighbor_idx] = [neighbor_level, list_idx]
                    next_atoms_to_explore = list(temp_atom_positions.keys())
                    neighbor_level += 1
            
            equivalent_atom_indices[atom_idx] = candidate_atom_indices
        
        # Step 4: Generate normalized bond information
        pos_indices = {}
        routes = route_between_targets.copy()
        for atom_idx, indices in equivalent_atom_indices.items():
            pos_indices[atom_idx] = [atom_idx] + indices

        total_indices = []
        for values in pos_indices.values():
            total_indices += values
        total_indices = list(set(total_indices))

        combs_cnt = 1
        for values in pos_indices.values():
            combs_cnt *= len(values)
            if combs_cnt > 10000:
                raise ValueError("Too many combinations.")


        combs = list(product(*[arr for arr in pos_indices.values()]))
        combs = sorted(combs, key=lambda x: sorted(x))
        
        success_pair = tgt_atom_indices.copy()
        for comb in combs:
            all_pairs = list(combinations(comb, 2))
            all_idx_pairs = list(combinations(range(len(comb)), 2))
            success = False
            for i, pair in enumerate(all_pairs):
                # Register routes if not already present
                if pair not in routes:
                    atom_idx = pair[0]
                    tgt_atom_idx = pair[1]
                    atom = mol.GetAtomWithIdx(atom_idx)
                    neighbor_level = 1
                    neighbor_routes = {0: ['']}
                    atom_position = {atom_idx: [0, 0]}
                    next_atoms_to_explore = [atom_idx]
                    routes[(atom_idx, atom_idx)] = ''
                    routes[(tgt_atom_idx, tgt_atom_idx)] = ''

                    while len(atom_position) < total_atoms:
                        temp_atom_positions = {}
                        temp_neighbor_routes = []
                        for current_idx in next_atoms_to_explore:
                            current_atom = mol.GetAtomWithIdx(current_idx)
                            for neighbor in current_atom.GetNeighbors():
                                neighbor_idx = neighbor.GetIdx()
                                if neighbor_idx in atom_position:
                                    continue
                                neighbor_symbol = neighbor.GetSymbol()
                                bond = mol.GetBondBetweenAtoms(current_idx, neighbor_idx)
                                bond_type = bond.GetBondType()

                                # Map bond type to numerical representation
                                if bond_type == Chem.BondType.SINGLE:
                                    bond_type_num = 0
                                elif bond_type == Chem.BondType.DOUBLE:
                                    bond_type_num = 1
                                elif bond_type == Chem.BondType.TRIPLE:
                                    bond_type_num = 2
                                elif bond_type == Chem.BondType.AROMATIC:
                                    bond_type_num = 3
                                else:
                                    raise ValueError("Invalid bond type.")
                                
                                # Build a route string
                                prev_level, prev_list_idx = atom_position[current_idx]
                                route_str = neighbor_routes[prev_level][prev_list_idx] + f'_{bond_type_num}_{neighbor_symbol}'

                                # Update positions and routes
                                if neighbor_idx not in temp_atom_positions:
                                    temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
                                    temp_neighbor_routes.append(route_str)
                                else:
                                    if route_str < temp_neighbor_routes[temp_atom_positions[neighbor_idx]]:
                                        temp_atom_positions[neighbor_idx] = len(temp_neighbor_routes)
                                    temp_neighbor_routes.append(route_str)
                        neighbor_routes[neighbor_level] = temp_neighbor_routes.copy()
                        for neighbor_idx, list_idx in temp_atom_positions.items():
                            atom_position[neighbor_idx] = [neighbor_level, list_idx]
                            if neighbor_idx in total_indices:
                                routes[(atom_idx, neighbor_idx)] = temp_neighbor_routes[list_idx]
                                routes[(neighbor_idx, atom_idx)] = temp_neighbor_routes[list_idx]
                        next_atoms_to_explore = list(temp_atom_positions.keys())
                        neighbor_level += 1

                tgt_route_str1 = routes[tgt_atom_indices[all_idx_pairs[i][0]], tgt_atom_indices[all_idx_pairs[i][1]]]
                tgt_route_str2 = routes[tgt_atom_indices[all_idx_pairs[i][1]], tgt_atom_indices[all_idx_pairs[i][0]]]
                route_str = routes[pair]
                if route_str == tgt_route_str1 or route_str == tgt_route_str2:
                    continue
                else:
                    break
            else:
                success = True
                success_pair = comb
                break

        # Prepare final bond information
        final_bond_info = [bond_info[0]]
        temp = []
        for i, atom_idx in enumerate(success_pair):
            temp.append([atom_idx, bond_info[2 * i + 2]])
        temp = sorted(temp, key=lambda x: x[0])
        for i, t in enumerate(temp):
            final_bond_info.append(t[0])
            final_bond_info.append(t[1])
            # final_bond_info.append(i + 1)
    except:
        final_bond_info = bond_info.copy()
    finally:
        signal.alarm(0)

    return final_bond_info



def calculate_advanced_scores(smiles_or_mol: str):
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

    return scores