from rdkit import Chem
from rdkit.Chem import rdmolops

def calculate_advanced_scores(smiles: str):
    """
    Calculate advanced scores for each atom in a molecule based on:
    - Symbol score (atomic type)
    - Join type score (bond type)
    - Distance from molecule center
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

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

# SMILES list for examples with diverse elements
smiles_list = [
    "CC(O)C",       # Isopropanol
    "CC(=O)O",      # Acetic acid
    "C1=CC=CC=C1",  # Benzene
    "CCCl",         # Chloroethane
    "CC(F)(F)F",    # Trifluoromethane
    "CCBr",         # Bromoethane
    "C1CC1",        # Cyclopropane
    "P(CC)CC",      # Triethylphosphine
]

# Iterate over SMILES and calculate scores
for smiles in smiles_list:
    print(f"SMILES: {smiles}")
    atom_scores = calculate_advanced_scores(smiles)
    for atom_idx, symbol, score in atom_scores:
        print(f"  Atom Index: {atom_idx}, Symbol: {symbol}, Advanced Score: {score:.2f}")
    max_score_idx = max(atom_scores, key=lambda x: x[2])[0]
    print(f"  Atom with Maximum Score: {max_score_idx}, Score: {atom_scores[max_score_idx][2]:.2f}")
    print()
