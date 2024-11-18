from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_arrays(smiles):
    # SMILESから分子オブジェクトを生成
    mol = Chem.MolFromSmiles(smiles)
    atom_data = []
    bond_data = []

    # 原子情報を抽出
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atom_data.append(atomic_num)  # 原子の原子番号をリストに追加

    # 結合情報を抽出
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        # 結合タイプを数値にマッピング
        if bond_type == Chem.BondType.SINGLE:
            bond_type_num = 0
        elif bond_type == Chem.BondType.DOUBLE:
            bond_type_num = 1
        elif bond_type == Chem.BondType.TRIPLE:
            bond_type_num = 2
        elif bond_type == Chem.BondType.AROMATIC:
            bond_type_num = 3
        else:
            bond_type_num = -1  # その他の結合は未サポート

        # 立体化学情報を取得
        if bond_type == Chem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            if stereo == Chem.BondStereo.STEREOCIS:
                stereo_num = -1  # シス
            elif stereo == Chem.BondStereo.STEREOTRANS:
                stereo_num = 1  # トランス
            else:
                stereo_num = 0  # 立体化学なし
        else:
            stereo_num = 0  # 非二重結合の場合、立体化学なし

        bond_data.append([begin_idx, end_idx, bond_type_num, stereo_num])

    return atom_data, bond_data

def arrays_to_smiles(atom_data, bond_data):
    # 新しい分子を構築
    mol = Chem.RWMol()
    atom_indices = []

    # 原子を追加
    for atomic_num in atom_data:
        atom = Chem.Atom(atomic_num)
        idx = mol.AddAtom(atom)
        atom_indices.append(idx)

    # 結合を追加
    for bond_info in bond_data:
        atom1, atom2, bond_type_num, stereo_num = bond_info

        # 結合タイプをRDKitのBondTypeに変換
        if bond_type_num == 0:
            bond_type = Chem.BondType.SINGLE
        elif bond_type_num == 1:
            bond_type = Chem.BondType.DOUBLE
        elif bond_type_num == 2:
            bond_type = Chem.BondType.TRIPLE
        elif bond_type_num == 3:
            bond_type = Chem.BondType.AROMATIC
        else:
            continue  # サポートされていない結合タイプはスキップ

        mol.AddBond(atom_indices[atom1], atom_indices[atom2], bond_type)

        # 二重結合の場合、立体化学の設定
        if bond_type_num == 1:
            bond = mol.GetBondBetweenAtoms(atom_indices[atom1], atom_indices[atom2])

            # 二重結合に接続する非水素原子を取得
            neighbors1 = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(atom1).GetNeighbors() if nbr.GetAtomicNum() != 1 and nbr.GetIdx() != atom2]
            neighbors2 = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(atom2).GetNeighbors() if nbr.GetAtomicNum() != 1 and nbr.GetIdx() != atom1]

            # ステレオアトムを設定して立体化学情報を適用
            if len(neighbors1) == 1 and len(neighbors2) == 1:
                bond.SetStereoAtoms(neighbors1[0], neighbors2[0])
                if stereo_num == -1:
                    bond.SetStereo(Chem.BondStereo.STEREOCIS)
                elif stereo_num == 1:
                    bond.SetStereo(Chem.BondStereo.STEREOTRANS)

    # サニタイズし、ケクレ化をスキップして芳香族SMILESを保持
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    # ケクレ化せずに芳香族を含むSMILESに変換
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=False)
    return smiles

if __name__ == "__main__":
    smiles_list = [
        "C/C=C/C=O",
        "C/C=C\\C=O",
        "c1ccc(Nc2cc(Nc3ccccc3)c3cc(-c4ccccc4)[nH]c3c2)cc1",
        "C1=CC=C2C(=C1)C=CC3=CC4=C(C=CC5=CC=CC(=C45)C=C3)C=C2",
        "CC(C)NCC(O)C1=CC=C(C=C1)S(=O)(=O)N(C)C2=CC=CC=C2",
    ]
    for smiles in smiles_list:
        atom_data, bond_data = smiles_to_arrays(smiles)
        smiles_trans_output = arrays_to_smiles(atom_data, bond_data)
        print("In:", smiles)
        print("Out:", smiles_trans_output)
        print()
