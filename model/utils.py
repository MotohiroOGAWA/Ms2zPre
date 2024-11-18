from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import copy
import numpy as np
import torch
import dill
from itertools import product, combinations
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
        print("error in add_Hs 1")
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

#Calculate frequency after decomposition
def count_fragments(mol):
    mol = Chem.rdmolops.RemoveHs(mol)
    new_mol = Chem.RWMol(mol)
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    sep_sets = [] #Set of atom maps of joints
    
    for bond in mol.GetBonds():
        if bond.IsInRing(): continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        #If both are inside the ring, split there.
        if a1.IsInRing() and a2.IsInRing():
            sep_sets.append((a1.GetIdx(), a2.GetIdx()))
        #If one atom is in a ring and the other has a bond order greater than 2, split there.
        elif (a1.IsInRing() and a2.GetDegree() > 1) or (a2.IsInRing() and a1.GetDegree() > 1):
            sep_sets.append((a1.GetIdx(), a2.GetIdx()))   
    sep_idx = 1
    atommap_dict = defaultdict(list) #key->AtomIdx, value->sep_idx (In the whole compound before decomposition)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if ((a1.GetIdx(),a2.GetIdx()) in sep_sets) or ((a2.GetIdx(),a1.GetIdx()) in sep_sets):
            a1map = new_mol.GetAtomWithIdx(a1.GetIdx()).GetAtomMapNum()
            a2map = new_mol.GetAtomWithIdx(a2.GetIdx()).GetAtomMapNum()
            atommap_dict[a1map].append(sep_idx)
            atommap_dict[a2map].append(sep_idx)
            new_mol = add_Hs(new_mol, a1, a2, bond)
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
            sep_idx += 1
    for i in list(atommap_dict.keys()):
        atommap_dict[i] = sorted(atommap_dict[i])  
    for i in list(atommap_dict.keys()):
        if atommap_dict[i] == []:
            atommap_dict.pop(i)
    new_mol = new_mol.GetMol()
    new_mol = sanitize(new_mol, kekulize = False)
    new_smiles = Chem.MolToSmiles(new_mol)
    fragments = [Chem.MolFromSmiles(fragment) for fragment in new_smiles.split('.')]
    fragments = [sanitize(fragment, kekulize = False) for fragment in fragments]
    
    count_labels = []
    for i, fragment in enumerate(fragments):
        order_list = [] #Stores join orders in the substructures
        count_label = []
        frag_mol = copy.deepcopy(fragment)
        for atom in frag_mol.GetAtoms():
            frag_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0)
        frag_smi = Chem.MolToSmiles(sanitize(frag_mol, kekulize = False))
        #Fix AtomIdx as order changes when AtomMap is deleted.
        atom_order = list(map(int, frag_mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
        for atom in fragment.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap in list(atommap_dict.keys()):
                order_list.append(atommap_dict[amap])
        order_list = sorted(order_list)
        count_label.append(frag_smi)
        for atom in fragment.GetAtoms():
            amap = atom.GetAtomMapNum()
            if amap in list(atommap_dict.keys()):
                for seq_idx in atommap_dict[amap]:
                    for amap2 in list(atommap_dict.keys()):
                        if (seq_idx in atommap_dict[amap2]) and (amap != amap2):
                            bond = mol.GetBondBetweenAtoms(amap, amap2)
                            bond_type = bond.GetBondType()
                            bond_type_str = ""
                            if bond_type == Chem.BondType.SINGLE:
                                bond_type_str = "SINGLE"
                            elif bond_type == Chem.BondType.DOUBLE:
                                bond_type_str = "DOUBLE"
                            elif bond_type == Chem.BondType.TRIPLE:
                                bond_type_str = "TRIPLE"
                            elif bond_type == Chem.BondType.AROMATIC:
                                bond_type_str = "AROMATIC"

                            count_label.append(atom_order.index(atom.GetIdx()))
                            count_label.append(bond_type_str)
                            count_label.append(order_list.index(atommap_dict[amap]) + 1)
    
        count_label = normalize_bond_info(count_label)
        count_labels.append(tuple(count_label))
    return count_labels, fragments

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
        degree = (len(bond_info) - 1) // 3  # Determine the number of bonds described
        total_atoms = mol.GetNumAtoms()

        # Step 1: Identify target atom indices from the bond information
        tgt_atom_indices = []
        route_between_targets = {}
        for i in range(degree):
            tgt_atom_indices.append(bond_info[3 * i + 1])

        # Step 2: Initialize dictionary to track equivalent atom indices
        equivalent_atom_indices = {}
        for i in range(degree):
            atom_idx = bond_info[3 * i + 1]
            bond_type = bond_info[3 * i + 2]
            bond_order = bond_info[3 * i + 3]
            
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
            temp.append([atom_idx, bond_info[3 * i + 2]])
        temp = sorted(temp, key=lambda x: x[0])
        for i, t in enumerate(temp):
            final_bond_info.append(t[0])
            final_bond_info.append(t[1])
            final_bond_info.append(i + 1)
    except:
        final_bond_info = bond_info.copy()
    finally:
        signal.alarm(0)

    return final_bond_info
