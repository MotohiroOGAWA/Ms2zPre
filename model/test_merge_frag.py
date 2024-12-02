from rdkit import Chem
from rdkit.Chem import rdmolops
from bidict import bidict
import copy

from utils import *

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
        if i == 0:
            combined_mol = copy.deepcopy(mol)  # Start with the first fragment
        else:
            combined_mol = Chem.CombineMols(combined_mol, mol)  # Add subsequent fragments
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom.SetAtomMapNum(atom_idx + offset)  # Assign unique atom map numbers
            atom_map[atom.GetIdx() + offset] = (i, atom_idx)
        # Track remaining bonds in the fragment
        for j in range((len(frag_infoes[i]) - 1) // 2):
            remaining_bond_poses.append((i, j))
        offset += mol.GetNumAtoms()  # Update offset for the next fragment

    # Convert the combined molecule to an editable RWMol
    combined_rwmol = Chem.RWMol(combined_mol)

    # Add specified bonds between fragments
    for i, (joint_pos1, bond_type, joint_pos2) in enumerate(merge_bond_poses):
        frag_idx1, bond_pos1 = joint_pos1
        atom_idx1 = atom_map.inv[(frag_idx1, frag_infoes[frag_idx1][2 * bond_pos1 + 1])]
        frag_idx2, bond_pos2 = joint_pos2
        atom_idx2 = atom_map.inv[(frag_idx2, frag_infoes[frag_idx2][2 * bond_pos2 + 1])]
        bond_type = token_to_chem_bond(bond_type)  # Convert bond type to RDKit format
        combined_rwmol.AddBond(atom_idx1, atom_idx2, bond_type)  # Add bond
        remaining_bond_poses.remove((frag_idx1, bond_pos1))
        remaining_bond_poses.remove((frag_idx2, bond_pos2))

    # Generate the final combined molecule and SMILES
    combined_mol = combined_rwmol.GetMol()
    smiles = Chem.MolToSmiles(combined_mol, isomericSmiles=True)

    # Extract atom order from SMILES
    atom_order = list(map(int, combined_mol.GetProp('_smilesAtomOutputOrder')[1:-2].split(",")))

    # Map new atom indices to original fragments and atom IDs
    new_atom_maps = bidict()
    for i, pre_atom_idx in enumerate(atom_order):
        frag_idx, atom_idx = atom_map[pre_atom_idx]
        new_atom_maps[i] = (frag_idx, atom_id_list[frag_idx][atom_idx])

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
    



if __name__ == "__main__":
    frag_infoes = [
        ("CCC(C)C", 1, '-', 4, '='),
        ("CC(C)(O)CC", 0, '-'),
        ("CC(C)C", 1, '-', 3, '='),
    ]
    merge_bond_posed = [
        ((0, 0), "-", (1, 0)),
        ((0, 1), "=", (2, 1)),
    ]
    atom_maps_list = [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3],
    ]
    frag_info, atom_map = merge_fragment_info(frag_infoes, merge_bond_posed, atom_maps_list)
    print(frag_info)