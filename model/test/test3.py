from rdkit import Chem
from itertools import product, combinations

def normalize_bond_structure(bond_structure_info):
    """
    Normalizes the bond structure information based on atom indices and connectivity in the SMILES representation.
    
    Parameters:
    bond_structure_info (tuple): A tuple containing SMILES string and bond information.

    Returns:
    list: Normalized representation of bond structure information.
    """
    mol = Chem.MolFromSmiles(bond_structure_info[0])
    bond_count = (len(bond_structure_info) - 1) // 3
    total_atoms = mol.GetNumAtoms()

    target_atom_indices = []
    route_str_between_targets = {}
    for i in range(bond_count):
        target_atom_indices.append(bond_structure_info[3 * i + 1])

    equivalent_atom_indices = {}
    for i in range(bond_count):
        atom_idx = bond_structure_info[3 * i + 1]
        bond_type = bond_structure_info[3 * i + 2]
        bond_idx = bond_structure_info[3 * i + 3]
        
        atom = mol.GetAtomWithIdx(atom_idx)
        current_level = 1
        level_neighbor_info = {}
        level_neighbor_info[0] = ['']
        atom_remote_position = {}
        atom_remote_position[atom_idx] = [0, 0] # level, list_index
        next_atom_indices = [atom_idx]

        while len(atom_remote_position) < total_atoms:
            temp_atom_remote_position = {}
            temp_neighbor_info = []
            for current_atom_idx in next_atom_indices:
                current_atom = mol.GetAtomWithIdx(current_atom_idx)
                for neighbor in current_atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in atom_remote_position:
                        continue
                    neighbor_symbol = neighbor.GetSymbol()
                    bond = mol.GetBondBetweenAtoms(current_atom_idx, neighbor_idx)
                    bond_type = bond.GetBondType()

                    bond_type_num = get_bond_type_number(bond_type)
                    current_atom_level, current_list_index = atom_remote_position[current_atom_idx]
                    route_str = level_neighbor_info[current_atom_level][current_list_index] + f'_{bond_type_num}_{neighbor_symbol}'

                    if neighbor_idx not in temp_atom_remote_position:
                        temp_atom_remote_position[neighbor_idx] = len(temp_neighbor_info)
                        temp_neighbor_info.append(route_str)
                    else:
                        if route_str < temp_neighbor_info[temp_atom_remote_position[neighbor_idx]]:
                            temp_atom_remote_position[neighbor_idx] = len(temp_neighbor_info)
                        temp_neighbor_info.append(route_str)
            level_neighbor_info[current_level] = temp_neighbor_info.copy()
            for neighbor_idx, list_index in temp_atom_remote_position.items():
                atom_remote_position[neighbor_idx] = [current_level, list_index]
                if neighbor_idx in target_atom_indices:
                    route_str_between_targets[(atom_idx, neighbor_idx)] = temp_neighbor_info[list_index]
            next_atom_indices = list(temp_atom_remote_position.keys())
            current_level += 1

        equivalent_atom_indices[atom_idx] = find_equivalent_atoms(mol, total_atoms, atom_idx, level_neighbor_info, route_str_between_targets)
    
    # Generate final representation
    final_structure = generate_final_structure(bond_structure_info, equivalent_atom_indices, target_atom_indices)
    return final_structure

def get_bond_type_number(bond_type):
    """
    Get a numerical representation of the bond type.
    
    Parameters:
    bond_type (rdkit.Chem.rdchem.BondType): Bond type.

    Returns:
    int: Corresponding numerical representation.
    """
    if bond_type == Chem.BondType.SINGLE:
        return 0
    elif bond_type == Chem.BondType.DOUBLE:
        return 1
    elif bond_type == Chem.BondType.TRIPLE:
        return 2
    elif bond_type == Chem.BondType.AROMATIC:
        return 3
    else:
        raise ValueError("Invalid bond type.")

def find_equivalent_atoms(mol, total_atoms, atom_idx, level_neighbor_info, route_str_between_targets):
    """
    Find equivalent atoms based on bond connectivity and route string matching.
    
    Parameters:
    mol (rdkit.Chem.Mol): Molecule object.
    total_atoms (int): Total number of atoms in the molecule.
    atom_idx (int): Target atom index.
    level_neighbor_info (dict): Level-wise neighbor information.
    route_str_between_targets (dict): Route string information between atoms.

    Returns:
    list: Equivalent atom indices.
    """
    candidate_atom_indices = [i for i in range(total_atoms) if i != atom_idx]
    ref_remote_route_str = ['+'.join(sorted(level_neighbor_info[level])) for level in range(max(level_neighbor_info.keys()) + 1)]

    for candidate_atom_idx in reversed(candidate_atom_indices):
        if not check_route_equivalence(mol, candidate_atom_idx, total_atoms, ref_remote_route_str, level_neighbor_info):
            candidate_atom_indices.remove(candidate_atom_idx)
    return candidate_atom_indices

def check_route_equivalence(mol, candidate_atom_idx, total_atoms, ref_remote_route_str, level_neighbor_info):
    """
    Check if the route strings match the reference remote route string.
    
    Parameters:
    mol (rdkit.Chem.Mol): Molecule object.
    candidate_atom_idx (int): Candidate atom index.
    total_atoms (int): Total number of atoms in the molecule.
    ref_remote_route_str (list): Reference remote route strings.
    level_neighbor_info (dict): Level-wise neighbor information.

    Returns:
    bool: Whether the candidate atom matches the reference.
    """
    current_level = 1
    next_atom_indices = [candidate_atom_idx]
    atom_remote_position = {candidate_atom_idx: [0, 0]}
    while len(atom_remote_position) < total_atoms:
        temp_atom_remote_position, temp_neighbor_info = get_neighbor_info(mol, next_atom_indices, atom_remote_position)
        candidate_route_str = '+'.join(sorted(temp_neighbor_info))
        if candidate_route_str != ref_remote_route_str[current_level]:
            return False
        next_atom_indices = list(temp_atom_remote_position.keys())
        current_level += 1
    return True

def get_neighbor_info(mol, next_atom_indices, atom_remote_position):
    """
    Get the neighbor information for the next set of atoms.
    
    Parameters:
    mol (rdkit.Chem.Mol): Molecule object.
    next_atom_indices (list): List of atom indices to process.
    atom_remote_position (dict): Current remote positions of atoms.

    Returns:
    tuple: Updated atom remote positions and neighbor information.
    """
    temp_atom_remote_position = {}
    temp_neighbor_info = []
    for current_atom_idx in next_atom_indices:
        current_atom = mol.GetAtomWithIdx(current_atom_idx)
        for neighbor in current_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in atom_remote_position:
                continue
            neighbor_symbol = neighbor.GetSymbol()
            bond = mol.GetBondBetweenAtoms(current_atom_idx, neighbor_idx)
            bond_type = bond.GetBondType()

            bond_type_num = get_bond_type_number(bond_type)
            current_atom_level, current_list_index = atom_remote_position[current_atom_idx]
            route_str = f'{current_atom_level}_{bond_type_num}_{neighbor_symbol}'

            if neighbor_idx not in temp_atom_remote_position:
                temp_atom_remote_position[neighbor_idx] = len(temp_neighbor_info)
                temp_neighbor_info.append(route_str)
            else:
                if route_str < temp_neighbor_info[temp_atom_remote_position[neighbor_idx]]:
                    temp_atom_remote_position[neighbor_idx] = len(temp_neighbor_info)
                temp_neighbor_info.append(route_str)
    return temp_atom_remote_position, temp_neighbor_info

def generate_final_structure(bond_structure_info, equivalent_atom_indices, target_atom_indices):
    """
    Generate the final normalized structure representation.
    
    Parameters:
    bond_structure_info (tuple): Original bond structure information.
    equivalent_atom_indices (dict): Equivalent atom indices information.
    target_atom_indices (list): Target atom indices.

    Returns:
    list: Normalized structure representation.
    """
    normalized_structure = [bond_structure_info[0]]
    temp = [[atom_idx, bond_structure_info[3 * i + 2]] for i, atom_idx in enumerate(target_atom_indices)]
    temp_sorted = sorted(temp, key=lambda x: x[0])
    for i, t in enumerate(temp_sorted):
        normalized_structure.extend([t[0], t[1], i + 1])
    return normalized_structure

# Example bond information
bond_structure_3 = ('c1ccc2cc3ccccc3cc2c1', 2, 'SINGLE', 1, 4, 'SINGLE', 2)
bond_structure_4 = ('c1ccc2cc3ccccc3cc2c1', 4, 'SINGLE', 1, 6, 'SINGLE', 2)

# Normalize bond structures
normalized_bond_structure_3 = normalize_bond_structure(bond_structure_3)
normalized_bond_structure_4 = normalize_bond_structure(bond_structure_4)

# Check results
print("Normalized Bond Structure 3:", normalized_bond_structure_3)
print("Normalized Bond Structure 4:", normalized_bond_structure_4)
print("Are bond structures equivalent?", normalized_bond_structure_3 == normalized_bond_structure_4)
