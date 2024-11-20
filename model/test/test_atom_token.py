from rdkit import Chem
from collections import Counter

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
                    bonding_type = ":"
                bond_tokens.append(bonding_type)
            
            counter = Counter(bond_tokens)
            for bonding_type in counter:
                remaining_bonding_type = []
                for key,count in counter.items():
                    if bonding_type == key:
                        remaining_bonding_type.extend([key]*(count-1))
                    else:
                        remaining_bonding_type.extend([key]*count)
                remaining_bonding_type = sorted(remaining_bonding_type, key=lambda x: sort_order.get(x, float('inf')))
                remaining_bonding_type_str = ''.join(remaining_bonding_type)
                atom_token = f"{bonding_type}{atom_symbol}{remaining_bonding_type_str}"
                atom_tokens.add(atom_token)
    
    # Return unique atom tokens as a sorted list
    return sorted(atom_tokens)

# Example usage
smiles = "CC(=O)OC"
mol = Chem.MolFromSmiles(smiles)

atom_tokens = get_atom_tokens(mol)
print("Atom tokens:", atom_tokens)
