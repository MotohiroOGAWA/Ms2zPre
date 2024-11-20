from rdkit import Chem
from rdkit.Chem import rdmolops
import networkx as nx
from collections import defaultdict

def extract_and_label_substructures(mol):
    """
    Mol オブジェクトを分割し、部分構造の再番号付け、開裂した結合情報を記録します。

    Args:
        mol (Chem.Mol): RDKit の Mol オブジェクト。

    Returns:
        list: 各部分構造の情報を格納した辞書のリスト。
              - 'group': 部分構造の番号。
              - 'atoms': 部分構造内で再割り当てされた原子インデックス。
              - 'smiles': 部分構造の SMILES。
              - 'bonds': 開裂した結合情報（ローカルインデックスと結合の種類）。
    """
    mol = Chem.rdmolops.RemoveHs(mol)  # 水素を削除
    new_mol = Chem.RWMol(mol)

    # 原子マップ番号を設定して元の原子インデックスを追跡
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    # 分子全体のグラフを構築
    graph = nx.Graph()
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        graph.add_edge(a1, a2, bond_type=bond.GetBondType())

    sep_sets = []  # 分割対象となる結合
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if bond.IsInRing():
            continue
        if a1.IsInRing() and a2.IsInRing():
            sep_sets.append((a1.GetIdx(), a2.GetIdx()))
        elif (a1.IsInRing() and a2.GetDegree() > 1) or (a2.IsInRing() and a1.GetDegree() > 1):
            sep_sets.append((a1.GetIdx(), a2.GetIdx()))

    # 結合を削除して分子を分割
    for a1_idx, a2_idx in sep_sets:
        new_mol.RemoveBond(a1_idx, a2_idx)

    new_mol = new_mol.GetMol()
    Chem.SanitizeMol(new_mol)
    fragments = Chem.MolToSmiles(new_mol).split('.')
    fragment_mols = [Chem.MolFromSmiles(frag) for frag in fragments]

    results = []
    for group_id, fragment in enumerate(fragment_mols):
        atom_map = {}  # ローカルインデックスと元のインデックスの対応
        for atom in fragment.GetAtoms():
            atom_map[atom.GetIdx()] = atom.GetAtomMapNum() - 1  # 元のインデックス（1始まりを0始まりに変換）

        broken_bonds = []
        for atom_idx, original_idx in atom_map.items():
            neighbors = graph.neighbors(original_idx)
            for nbr_idx in neighbors:
                if nbr_idx not in atom_map.values():  # 外部の原子との結合を探す
                    bond = graph.get_edge_data(original_idx, nbr_idx)
                    if bond:
                        bond_type = {
                            Chem.BondType.SINGLE: "SINGLE",
                            Chem.BondType.DOUBLE: "DOUBLE",
                            Chem.BondType.TRIPLE: "TRIPLE",
                            Chem.BondType.AROMATIC: "AROMATIC",
                        }.get(bond['bond_type'], "UNKNOWN")
                        broken_bonds.append({
                            "atom": atom_idx,  # ローカルインデックス
                            "bond_type": bond_type
                        })

        # SMILESを生成し、原子マップ番号を削除
        for atom in fragment.GetAtoms():
            atom.SetAtomMapNum(0)
        smiles = Chem.MolToSmiles(fragment)

        results.append({
            "group": group_id,
            "atoms": list(atom_map.keys()),  # ローカルインデックス
            "smiles": smiles,
            "bonds": broken_bonds
        })

    return results

# 入力 SMILES
smiles = "CC(C)CCCCCCOC(=O)c1cc(C(=O)OCCCCCCC(C)C)c(C(=O)O)cc1C(=O)O"
mol = Chem.MolFromSmiles(smiles)

substructures = extract_and_label_substructures(mol)

print(f"Input SMILES: {smiles}")
for sub in substructures:
    print(f"Group {sub['group']}: Atoms: {sub['atoms']}, SMILES: {sub['smiles']}, Bonds: {sub['bonds']}")
