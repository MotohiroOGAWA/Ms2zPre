from rdkit import Chem
from itertools import product, combinations

def normalize_bond_info(bond_info):
    """
    Normalize the bond information by mapping atom indices based on SMILES output order.
    This ensures equivalent representation of atom indices and bonds for comparison.
    
    Parameters:
        bond_info (tuple): A tuple containing SMILES string, atom indices, bond types, and bond orders.
    
    Returns:
        list: Normalized bond information where equivalent bonds are assigned consistent indices.
    """
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

    return final_bond_info


# Example bond information
bond_n = [
    ('c1ccc2cc3ccccc3cc2c1', 2, 'SINGLE', 1, 4, 'SINGLE', 2),
    ('c1ccc2cc3ccccc3cc2c1', 4, 'SINGLE', 1, 6, 'SINGLE', 2),
    ('c1ccc2cc3ccccc3cc2c1', 13, 'SINGLE', 1, 4, 'SINGLE', 1, 6, 'SINGLE', 2)
]
# Normalize bonds
for i, bond_info in enumerate(bond_n):
    normalized_bond_info = normalize_bond_info(bond_info)
    print(f"Normalized Bond {i+1}:", normalized_bond_info)
