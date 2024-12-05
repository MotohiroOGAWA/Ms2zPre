from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolops

from itertools import product, combinations, permutations, chain
from collections import Counter
from collections import defaultdict
from bidict import bidict
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

def token_to_chem_bond(token):
    if token == "-":
        bond_type = Chem.BondType.SINGLE
    elif token == "=":
        bond_type = Chem.BondType.DOUBLE
    elif token == "#":
        bond_type = Chem.BondType.TRIPLE
    elif token == ":":
        bond_type = Chem.BondType.AROMATIC
    else:
        raise ValueError("Invalid bond token.")
    return bond_type

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

def remove_Hs(rwmol, a1, a2, bond_type):
    try:
        if str(bond_type) == 'SINGLE':
            num = 1
        elif str(bond_type) == 'DOUBLE':
            num = 2
        elif str(bond_type) == 'TRIPLE':
            num = 3
        elif str(bond_type) == 'AROMATIC':
            print("error in remove_Hs 1")
        else:
            print("error in remove_Hs 2")
    except:
        if bond_type == 0:
            num = 1
        elif bond_type == 1:
            num = 2
        elif bond_type == 2:
            num = 3
        else:
            raise
    rwmol = Chem.AddHs(rwmol)
    rwmol = Chem.RWMol(rwmol)
    #Set hydrogen maps for connected atoms
    h_map1 = 2000000
    h_map2 = 3000000
    f_h_map1 = copy.copy(h_map1)
    f_h_map2 = copy.copy(h_map2)
    for b in rwmol.GetBonds():
        s_atom = b.GetBeginAtom()
        e_atom = b.GetEndAtom()
        if (e_atom.GetIdx() == a1.GetIdx()) and (s_atom.GetSymbol() == 'H'):
            s_atom.SetAtomMapNum(h_map1)
            h_map1 += 1
        elif (s_atom.GetIdx() == a1.GetIdx()) and (e_atom.GetSymbol() == 'H'):
            e_atom.SetAtomMapNum(h_map1)
            h_map1 += 1
        elif (e_atom.GetIdx() == a2.GetIdx()) and (s_atom.GetSymbol() == 'H'):
            s_atom.SetAtomMapNum(h_map2)
            h_map2 += 1
        elif (s_atom.GetIdx() == a2.GetIdx()) and (e_atom.GetSymbol() == 'H'):
            e_atom.SetAtomMapNum(h_map2)
            h_map2 += 1
    for i in range(num):
        try:
            for atom in rwmol.GetAtoms():
                if atom.GetAtomMapNum() == f_h_map1 + i:
                    rwmol.RemoveAtom(atom.GetIdx())
                    break
            for atom in rwmol.GetAtoms():
                if atom.GetAtomMapNum() == f_h_map2 + i:
                    rwmol.RemoveAtom(atom.GetIdx())
                    break
        except:
            print("Remove Hs times Error!!")
            raise
    rwmol = rwmol.GetMol()
    rwmol = sanitize(rwmol, kekulize = False)
    rwmol = Chem.RemoveHs(rwmol)
    rwmol = Chem.RWMol(rwmol)
    return rwmol

def split_fragment_info(frag_info, cut_bond_indices, ori_atom_indices):
    """
    Break bonds at specified atom indices and generate new frag_info for resulting fragments.
    
    Parameters:
        frag_info (tuple): The original fragment information, e.g., ('CC(C)CCCCCCOC(=O)', 0, '-', 10, '-').
        bond_indices (list of tuples): Pairs of atom indices to break bonds (e.g., [(1, 3)]).
    
    Returns:
        list of tuples: New frag_info for each resulting fragment.
    """
    # Extract the SMILES string from frag_info
    smiles = frag_info[0]

    # Convert SMILES to Mol object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string in frag_info.")
    mol = Chem.rdmolops.RemoveHs(mol)
    
    # Create editable version of the molecule
    new_mol = Chem.RWMol(mol)

    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    
    # Break specified bonds
    bond_types = {}
    for idx1, idx2 in cut_bond_indices:
        if new_mol.GetBondBetweenAtoms(idx1, idx2) is not None:
            atom1 = new_mol.GetAtomWithIdx(idx1)
            atom2 = new_mol.GetAtomWithIdx(idx2)
            bond = new_mol.GetBondBetweenAtoms(idx1, idx2)
            new_mol = add_Hs(new_mol, atom1, atom2, bond)  # Add hydrogens to adjust valence
            new_mol.RemoveBond(atom1.GetIdx(), atom2.GetIdx())
            if idx1 < idx2:
                bond_types[(idx1, idx2)] = chem_bond_to_token(bond.GetBondType())
            else:
                bond_types[(idx2, idx1)] = chem_bond_to_token(bond.GetBondType())
        else:
            raise ValueError(f"No bond found between atom indices {idx1} and {idx2}.")
    
    # Generate new fragment information for each resulting fragment
    new_mol = new_mol.GetMol()
    new_mol = sanitize(new_mol, kekulize = False)
    new_smiles = Chem.MolToSmiles(new_mol)
    fragment_mols = [Chem.MolFromSmiles(fragment) for fragment in new_smiles.split('.')]
    fragment_mols = [sanitize(fragment, kekulize = False) for fragment in fragment_mols]

    frag_infoes = []
    atom_dicts = {}
    bond_poses_dict = defaultdict(lambda: defaultdict(list))
    for frag_idx, frag_mol in enumerate(fragment_mols):
        
        atom_dict = {}
        for i, atom in enumerate(frag_mol.GetAtoms()):
            amap = atom.GetAtomMapNum()
            atom_dict[amap] = i
            
        for atom in frag_mol.GetAtoms():
            frag_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0)

        frag_smi = Chem.MolToSmiles(frag_mol)
        atom_order = list(map(int, frag_mol.GetProp('_smilesAtomOutputOrder')[1:-2].split(",")))

        for key in atom_dict:
            atom_dict[key] = atom_order.index(atom_dict[key])
        
        bond_infoes = []
        for i in range(1, len(frag_info), 2):
            if frag_info[i] in atom_dict:
                bond_infoes.append([atom_dict[frag_info[i]], frag_info[i+1]])
        for (idx1, idx2) in cut_bond_indices:
            if idx1 in atom_dict:
                idx = idx1
            elif idx2 in atom_dict:
                idx = idx2
            else:
                continue
            
            bond_type = bond_types.get((idx1, idx2), bond_types.get((idx2, idx1)))
            bond_infoes.append([atom_dict[idx], bond_type])
        bond_infoes = sorted(bond_infoes, key = lambda x: x[0])

        new_frag_info = [frag_smi]
        for i in range(len(bond_infoes)):
            new_frag_info.append(bond_infoes[i][0])
            new_frag_info.append(bond_infoes[i][1])
            bond_poses_dict[frag_idx][(bond_infoes[i][0], bond_infoes[i][1])].append(i)
        frag_infoes.append([tuple(new_frag_info), []])
        atom_dicts[frag_idx] = atom_dict
    
    for (bond_idx1, bond_idx2) in cut_bond_indices:
        for frag_idx, atom_dict in atom_dicts.items():
            if bond_idx1 in atom_dict:
                bond_i1 = atom_dict[bond_idx1]
                frag_idx1 = frag_idx
            if bond_idx2 in atom_dict:
                bond_i2 = atom_dict[bond_idx2]
                frag_idx2 = frag_idx
        bond = mol.GetBondBetweenAtoms(bond_idx1, bond_idx2)
        bond_type = chem_bond_to_token(bond.GetBondType())
        frag_infoes[frag_idx1][1].append((bond_poses_dict[frag_idx1][(bond_i1, bond_type)][0], (frag_idx2, bond_poses_dict[frag_idx2][(bond_i2, bond_type)][0])))
        frag_infoes[frag_idx2][1].append((bond_poses_dict[frag_idx2][(bond_i2, bond_type)].pop(0), (frag_idx1, bond_poses_dict[frag_idx1][(bond_i1, bond_type)].pop(0))))
    
    for i in range(len(frag_infoes)):
        frag_infoes[i][1] = sorted(frag_infoes[i][1], key = lambda x: x[0])

    normalize_bond_dict = defaultdict(dict)
    normalize_frag_infoes = []
    final_atom_maps = {}
    for frag_idx, frag_mol in enumerate(fragment_mols):
        frag_info = frag_infoes[frag_idx][0]
        new_frag_info, atom_maps = normalize_bond_info(frag_info, list(range(frag_mol.GetNumAtoms())))
        normalize_frag_infoes.append([new_frag_info, []])
        for global_atom_i, pre_frag_atom_i in atom_dicts[frag_idx].items():
            final_atom_maps[ori_atom_indices[global_atom_i]] = (frag_idx, atom_maps.index(pre_frag_atom_i))
        new_bond_dict = defaultdict(list)
        for i, (new_atom_idx, bond_type) in enumerate(zip(new_frag_info[1::2], new_frag_info[2::2])):
            new_bond_dict[(new_atom_idx, bond_type)].append(i)
        for i, pre_atom_idx in enumerate(frag_info[1::2]):
            normalize_bond_dict[frag_idx][i] = new_bond_dict[(atom_maps.index(pre_atom_idx), frag_info[2*i+2])].pop(0)

    for frag_idx, frag_info in enumerate(frag_infoes):
        for bond_pos, (to_frag_idx, to_bond_pos) in frag_info[1]:
            normalize_frag_infoes[frag_idx][1].append((normalize_bond_dict[frag_idx][bond_pos], (to_frag_idx, normalize_bond_dict[to_frag_idx][to_bond_pos])))
        
    return normalize_frag_infoes, final_atom_maps

def merge_fragment_info(frag_infoes, merge_bond_poses, atom_id_list = None):
    """
    Merge multiple molecular fragments into a single molecule by combining and bonding them.
    
    Args:
        frag_infoes (list of tuples): Each tuple contains fragment information. 
            The first element is the SMILES string of the fragment. Subsequent elements are:
            (smiles, atom_idx1, bond_type1, atom_idx2, bond_type2, ...)
                - The atom indices involved in bonds
                - The bond types (e.g., '-', '=', etc.)
        merge_bond_poses (list of tuples): Specifies the bonds to be created between fragments.
            Each tuple contains:
            ((frag_idx1, bond_pos1), bond_type, (frag_idx2, bond_pos2))
                - Position of the first fragment and its bond index
                - Bond type (e.g., '-', '=', etc.)
                - Position of the second fragment and its bond index
        atom_id_list (list of lists, optional): Maps fragment atom indices to global atom IDs.
            If not provided, default indices [0, 1, ..., N] are used for each fragment.

    Returns:
        tuple: A tuple containing:
            - final_frag_info (tuple): The SMILES string of the combined molecule and bond information.
            - final_atom_map (bidict): Maps global atom indices in the combined molecule to fragment indices and atom IDs.
    """

    # Convert SMILES to RDKit molecules for each fragment
    mols = [Chem.MolFromSmiles(frag_info[0]) for frag_info in frag_infoes]

    # If no atom_id_list is provided, use default atom indices for each fragment
    if atom_id_list is None:
        atom_id_list = [list(range(mol.GetNumAtoms())) for mol in mols]

    # Initialize atom mapping and remaining bond positions
    atom_map = bidict()
    remaining_bond_poses = []
    offset = 0

    # Combine molecules and assign atom map numbers
    for i, mol in enumerate(mols):
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom.SetAtomMapNum(atom_idx + offset)  # Assign unique atom map numbers
            atom_map[atom.GetIdx() + offset] = (i, atom_idx)
            
        if i == 0:
            combined_mol = copy.deepcopy(mol)  # Start with the first fragment
        else:
            combined_mol = Chem.CombineMols(combined_mol, mol)  # Add subsequent fragments

        # Track remaining bonds in the fragment
        for j in range((len(frag_infoes[i]) - 1) // 2):
            remaining_bond_poses.append((i, j))
        offset += mol.GetNumAtoms()  # Update offset for the next fragment

    # Convert the combined molecule to an editable RWMol
    combined_rwmol = Chem.RWMol(combined_mol)

    # Add specified bonds between fragments
    for i, (joint_pos1, bond_type, joint_pos2) in enumerate(merge_bond_poses):
        frag_idx1, bond_pos1 = joint_pos1
        map_number1 = atom_map.inv[(frag_idx1, frag_infoes[frag_idx1][2 * bond_pos1 + 1])]

        frag_idx2, bond_pos2 = joint_pos2
        map_number2 = atom_map.inv[(frag_idx2, frag_infoes[frag_idx2][2 * bond_pos2 + 1])]

        # Find atom indices by map number
        atom_idx1 = next(atom.GetIdx() for atom in combined_rwmol.GetAtoms() if atom.GetAtomMapNum() == map_number1)
        atom_idx2 = next(atom.GetIdx() for atom in combined_rwmol.GetAtoms() if atom.GetAtomMapNum() == map_number2)

        atom1 = combined_rwmol.GetAtomWithIdx(atom_idx1)
        atom2 = combined_rwmol.GetAtomWithIdx(atom_idx2)
        bond_type = token_to_chem_bond(bond_type)  # Convert bond type to RDKit format
        combined_rwmol.AddBond(atom_idx1, atom_idx2, bond_type)  # Add bond
        combined_rwmol = remove_Hs(combined_rwmol, atom1, atom2, bond_type)  # Remove hydrogens
        remaining_bond_poses.remove((frag_idx1, bond_pos1))
        remaining_bond_poses.remove((frag_idx2, bond_pos2))

    # Generate the final combined molecule and SMILES
    combined_mol = combined_rwmol.GetMol()
    atom_map2 = bidict()
    for i, atom in enumerate(combined_mol.GetAtoms()):
        atom_map2[i] = atom_map[atom.GetAtomMapNum()]
    for atom in combined_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(combined_mol, isomericSmiles=True)

    # Extract atom order from SMILES
    atom_order = list(map(int, combined_mol.GetProp('_smilesAtomOutputOrder')[1:-2].split(",")))

    new_atom_maps = bidict({i: atom_map2[order] for i, order in enumerate(atom_order)})

    # Map new atom indices to original fragments and atom IDs
    # new_atom_maps = bidict()
    # for i, pre_atom_idx in enumerate(atom_order):
    #     frag_idx, atom_idx = atom_map[pre_atom_idx]
    #     new_atom_maps[i] = (frag_idx, atom_id_list[frag_idx][atom_idx])

    # Collect remaining bond information
    bond_infoes = []
    for frag_idx, bond_pos in remaining_bond_poses:
        atom_idx = frag_infoes[frag_idx][2 * bond_pos + 1]
        bond_type = frag_infoes[frag_idx][2 * bond_pos + 2]
        bond_infoes.append((atom_map.inv[(frag_idx, atom_id_list[frag_idx][atom_idx])], bond_type))

    # Create the new fragment information
    new_frag_info = [smiles]
    for bond_info in bond_infoes:
        new_frag_info.extend(bond_info)
    new_frag_info = tuple(new_frag_info)

    # Normalize bond information
    final_frag_info, new_atom_indices = normalize_bond_info(new_frag_info, list(range(len(new_atom_maps))))

    # Create the final atom map
    final_atom_map = bidict()
    for i, atom_idx in enumerate(new_atom_indices):
        final_atom_map[i] = new_atom_maps[atom_idx]

    return final_frag_info, final_atom_map
    
        
def match_fragment(tgt_frag_info, tgt_start_pos, qry_frag_info, qry_start_pos):
    tgt_mol = Chem.MolFromSmiles(tgt_frag_info[0])
    qry_mol = Chem.MolFromSmiles(qry_frag_info[0])
    tgt_bond_infoes = [(bond_idx, bond_type) for bond_idx, bond_type in zip(tgt_frag_info[1::2], tgt_frag_info[2::2])]

    qry_bond_infoes = [(bond_idx, bond_type) for bond_idx, bond_type in zip(qry_frag_info[1::2], qry_frag_info[2::2])]


    if tgt_bond_infoes[tgt_start_pos][1] != qry_bond_infoes[qry_start_pos][1]:
        return None

    qry_route_trees = build_route_tree(qry_frag_info, qry_start_pos, mol=qry_mol)
    max_route = max([len(x['idx']) for x in qry_route_trees])
    tgt_route_trees = build_route_tree(tgt_frag_info, tgt_start_pos, mol=tgt_mol, options={'max_route': max_route})

    qry_route_trees = sorted(qry_route_trees, key = lambda x: "_".join(x['route']), reverse = True)
    qry_route_strs = ["_".join(x['route']) for x in qry_route_trees]
    tgt_route_trees = sorted(tgt_route_trees, key = lambda x: "_".join(x['route']), reverse = True)
    tgt_route_strs = ["_".join(x['route']) for x in tgt_route_trees]
    
    route_pair = defaultdict(list)
    nj = 0
    for i, qry_route_tree in enumerate(qry_route_trees):
        j = nj
        while j < len(tgt_route_trees):
            if qry_route_strs[i] > tgt_route_strs[j]:
                nj = j
                break
            if qry_route_strs[i] == tgt_route_strs[j]:
                route_pair[i].append(j)
            j += 1
    
    if len(qry_route_strs) != len(route_pair):
        return None
    
    atom_idx_dicts = []
    for qry_route_idx, tgt_route_indices in route_pair.items():
        qry_route_tree = qry_route_trees[qry_route_idx]
        if len(atom_idx_dicts) == 0:
            for tgt_route_idx in tgt_route_indices:
                atom_idx_dict = {qry_atom_idx: tgt_atom_idx for qry_atom_idx, tgt_atom_idx in zip(qry_route_tree['idx'], tgt_route_trees[tgt_route_idx]['idx'])}
                atom_idx_dicts.append(atom_idx_dict)
        else:
            new_atom_idx_dicts = []
            for atom_idx_dict in atom_idx_dicts:
                for tgt_route_idx in tgt_route_indices:
                    tmp_atom_idx_dict = copy.deepcopy(atom_idx_dict)
                    for qry_atom_idx, tgt_atom_idx in zip(qry_route_tree['idx'], tgt_route_trees[tgt_route_idx]['idx']):
                        if (qry_atom_idx in tmp_atom_idx_dict) and tmp_atom_idx_dict[qry_atom_idx] != tgt_atom_idx:
                            break
                        if tgt_atom_idx in tmp_atom_idx_dict.values():
                            continue
                        tmp_atom_idx_dict[qry_atom_idx] = tgt_atom_idx
                    else:
                        new_atom_idx_dicts.append(tmp_atom_idx_dict)
            atom_idx_dicts = new_atom_idx_dicts

        if len(atom_idx_dicts) == 0:
            break
    
    atom_idx_dicts = [atom_idx_dict for atom_idx_dict in atom_idx_dicts if len(atom_idx_dict) == len(qry_mol.GetAtoms())]
    if len(atom_idx_dicts) == 0:
        return None

    return atom_idx_dicts[0]


def build_route_tree(frag_info, start_bond_pos, mol=None, options=None):
    if mol is None:
        mol = Chem.MolFromSmiles(frag_info[0])
    
    if options is not None and 'max_route' in options:
        max_route = options['max_route']
    else:
        max_route = float('inf')
        
    bond_infoes = [(bond_idx, bond_type) for bond_idx, bond_type in zip(frag_info[1::2], frag_info[2::2])]
    frag_info_dict = defaultdict(list)
    for i in range(len(bond_infoes)):
        frag_info_dict[bond_infoes[i][0]].append(i)
    
    visited = set()
    completed_routes = []
    start_atom_idx = bond_infoes[start_bond_pos][0]
    start_atom = mol.GetAtomWithIdx(start_atom_idx)
    current_routes = [{'idx': [start_atom_idx], 'route': []}]
    current_routes[0]['route'].append(bond_infoes[start_bond_pos][1])
    current_routes[0]['route'].append(get_atom_symbol(start_atom))
    visited.add(bond_infoes[start_bond_pos][0])
    for i, bond_info in enumerate(bond_infoes):
        if i == start_bond_pos:
            continue
        if bond_info[0] == start_atom_idx:
            route = copy.deepcopy(current_routes[0])
            route['route'].append(bond_info[1])
            completed_routes.append(route)

    if len(visited) == mol.GetNumAtoms():
        if len(completed_routes) == 0: # -O などの1つの原子で続きの結合がない場合 
            completed_routes.append(current_routes[0])
        next_routes = []
        current_routes = []
    
    route_cnt = 1
    while len(current_routes) > 0:
        next_routes = []
        for i, current_route in enumerate(reversed(current_routes)):
            current_atom = mol.GetAtomWithIdx(current_route['idx'][-1])
            neighbors = [neighbor for neighbor in current_atom.GetNeighbors() if neighbor.GetIdx() not in visited]

            if len(neighbors) == 0:
                if len(current_route['route']) % 2 == 0: # -C-C などの続きの結合がない場合
                    completed_routes.append(current_route)
                continue

            for neighbor in neighbors:
                neighbor_idx = neighbor.GetIdx()
                visited.add(neighbor_idx)
                new_route = copy.deepcopy(current_route)
                bond = mol.GetBondBetweenAtoms(current_route['idx'][-1], neighbor_idx)
                bond_type = chem_bond_to_token(bond.GetBondType())
                new_route['route'].append(bond_type)
                if route_cnt < max_route:
                    new_route['idx'].append(neighbor_idx)
                    new_route['route'].append(get_atom_symbol(neighbor))
                
                    for i, bond_info in enumerate(bond_infoes):
                        if neighbor_idx != bond_info[0]:
                            continue
                        route = new_route.copy()
                        route['route'].append(bond_info[1])
                        completed_routes.append(route)
                
                next_routes.append(new_route)

        current_routes = next_routes
        route_cnt += 1
        if route_cnt > max_route:
            break

    
    for current_route in current_routes:
        completed_routes.append(current_route)
    
    return completed_routes

def get_atom_symbol(atom):
    symbol = atom.GetSymbol()
    if atom.GetNumExplicitHs() > 0:
        symbol += "H" * atom.GetNumExplicitHs()
        symbol = f"[{symbol}]"
    return symbol

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

def normalize_bond_info(bond_info, frag_atom_indices, timeout_seconds=10):
    """
    Normalize the bond information with a time limit.
    If the processing time exceeds the limit, return the bond_info with an error message.
    
    Parameters:
        bond_info (tuple): The bond information to normalize.
        frag_atom_indices (list): Atom indices for the fragment.
        timeout_seconds (int): The maximum allowed time in seconds.
    
    Returns:
        tuple: Normalized bond_info and atom indices or the original bond_info with an error message if timed out.
    """
    sort_order = {'-': 0, '=': 1, '#': 2, ':': 3}
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        if len(bond_info) == 1:
            return bond_info, frag_atom_indices

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

        if total_atoms == 1:
            # bond_infoes = sorted(bond_infoes, key = lambda x: sorted(x[1]))
            final_bond_info = [bond_info[0]]
            bond_infoes = sorted(bond_infoes, key = lambda x: sort_order.get(x[1], float('inf')))
            for bond_info in bond_infoes:
                final_bond_info.extend(bond_info)
            new_atom_indices = frag_atom_indices.copy()
            return tuple(final_bond_info), new_atom_indices

        route_matrix[0][0] = [[str(0)]]
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
            if atom.GetNumExplicitHs() > 0:
                atom_symbol += 'H' * atom.GetNumExplicitHs()
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
        for tgt_atom_idx in list(set(tgt_atom_indices)):
            tgt_atom_route_pos_group_cnt[atom_route_pos_symbol[tgt_atom_idx]] += 1
        
        for i in range(total_atoms):
            atom_route_pos_group[atom_route_pos_symbol[i]].append(i)

        tgt_group_route_str = '+'.join(sorted([atom_route_symbol[comb[0]][comb[1]] for comb in combinations(sorted(list(set(tgt_atom_indices))), 2)]))


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
        sorted_tgt_candidates = sorted([tuple(sorted(sum(comb, ()))) for comb in product(*select_indices_comb)])

        for tgt_candidate in sorted_tgt_candidates:
            ans_group_route_str = '+'.join(sorted([atom_route_symbol[comb[0]][comb[1]] for comb in list(combinations(sorted(tgt_candidate), 2))]))
            if ans_group_route_str == tgt_group_route_str:
                result_list = tgt_candidate
                break
        
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
            for i in list(set(tgt_atom_indices)):
                if i in atom_group:
                    tmp1.append(i)
            for i in result_list:
                if i in atom_group:
                    tmp2.append(i)
            if len(tmp1) != len(tmp2):
                raise ValueError("Invalid atom group.")
            if len(tmp1) == 1 and len(tmp2) == 1:
                tgt_pre_to_post_idx[tmp1[0]] = tmp2[0]
            elif len(tmp1) > 1 and len(tmp2) > 1:
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

        bond_pos_counter = Counter([(bond_info[2 * i + 1],bond_info[2 * i + 2]) for i in range(degree)])
        bond_type_dict = {bond_info[2 * i + 1]: bond_info[2 * i + 2] for i in range(degree)}
        def calc_pair_score(pair):
            # score = sum([(4*bond_pos_counter[p[0]]+(3-sort_order[])) * 4**(p[1]) for p in pair])
            pass
            # return score

        if len(tgt_same_pos_group) > 0:
            pairs_iters = [list(generate_pairs_iter(candidate_pair)) for candidate_pair in tgt_same_pos_group]
            candidate_pairs = list(product(*pairs_iters))
            candidate_pairs = sorted(candidate_pairs, key=lambda x: calc_pair_score(sum(x, [])))
            for candidate_pair in candidate_pairs:
                candidate_pair = sum(candidate_pair, [])
                candidate_pair.extend([(pre_idx, post_idx) for pre_idx, post_idx in tgt_pre_to_post_idx.items()])
                for pair in combinations(candidate_pair, 2):
                    if atom_route_symbol[pair[0][0]][pair[1][0]] != atom_route_symbol[pair[0][1]][pair[1][1]]:
                        break
                else:
                    for pre_idx, post_idx in candidate_pair:
                        tgt_pre_to_post_idx[pre_idx] = post_idx
                    break

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
        temp = sorted(temp, key=lambda x: (x[0], sort_order.get(x[1], float('inf'))))
        for t in temp:
            final_bond_info.append(t[0])
            final_bond_info.append(t[1])

        final_bond_info = tuple(final_bond_info)
        
    except TimeoutException as e:
        raise TimeoutException(f'Timeout occurred in normalize_bond_info after {timeout_seconds} seconds: {bond_info}')
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