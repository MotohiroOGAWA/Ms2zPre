from rdkit import Chem

smiles = 'c1nc[nH]n1'
mol = Chem.MolFromSmiles(smiles)

for atom in mol.GetAtoms():
    if atom.GetSymbol() == 'N' and atom.GetIsAromatic():
        print(f"Atom Index: {atom.GetIdx()}")
        print(f"Explicit Hs: {atom.GetNumExplicitHs()}")
        print(f"Implicit Hs: {atom.GetNumImplicitHs()}")


from rdkit import Chem

def check_radicals(smiles):
    """
    Check for radicals in a molecule and display the radical information.

    Args:
        smiles (str): The SMILES string of the molecule.
    """
    # 分子を生成
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    
    # 各原子のラジカル情報を確認
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        radical_electrons = atom.GetNumRadicalElectrons()
        if radical_electrons > 0:
            print(f"Atom {atom_idx} ({atom_symbol}) is a radical with {radical_electrons} unpaired electrons.")
        else:
            print(f"Atom {atom_idx} ({atom_symbol}) is not a radical.")

# サンプルSMILES
smiles = "CS(=O)(=O)c1cc[c]cc1"  # ラジカルを含む
check_radicals(smiles)

