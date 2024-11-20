from rdkit import Chem
from utils import *

class SubstructureMatcher:
    func_group=[
        '[!#1]-[OH]', # ヒドロキシ基
        '[!#1;X1,X2,X3](=O)-,=[!#1]', 
        '[!#1]-O-[!#1]', # エーテル結合

        '[!#1]-[NH2]', # アミノ基
        '[!#1]=[NH]', # アミノ基
        '[!#1]-[N+](=O)[O-]', # ニトロ基
        # '[!#1]-N=O', # ニトロソ基
        '[!#1]-N=N-[!#1]', # アゾ基
        '[!#1]-[NH]-[NH]-[!#1]', # ヒドラゾ基
        '[!#1]-C#N', # シアノ基

        '[!#1]-[SH1]', # チオール基
        '[!#1]=S', # チオール基
        '[S](=O)(=O)-[!#1]',
        '[!#1]-[S](=O)(=O)-[!#1]', # スルホン
        '[!#1]-S-[!#1]', # チオエーテル結合

        '[!#1]-[P](=O)(-O)(-O)',

        '[C;X3](=O)(-N-[!#1])', # アミド結合
        '[!#1]-[F,Cl,Br,I]', # ハロゲン化物

    ],
    substructure =[
        # '[OH]', # ヒドロキシ基
        '[C;X3](=O)(-[N;X2,X3])'
        '[P](=O)(-O)(-O)',
    ]


    def __init__(self):
        func_group = {}
        for functional_g in self.func_group[0]:
            func_group[functional_g] = Chem.MolFromSmarts(functional_g)
        self.func_group = func_group

    def split_molecule_by_functional_groups(self, mol, min_non_ring_neighbors=0):
        """
        Splits a molecule into fragments based on functional groups and connectivity,
        while tracking broken bond information and ensuring proper handling of ring and non-ring regions.

        Args:
            mol (Chem.Mol): Input RDKit molecule object.
            min_non_ring_neighbors (int, optional): Minimum number of non-ring neighbors
                an atom must have for the bond to be split. Defaults to 0.

        Returns:
            tuple:
                - count_labels (list): List of tuples containing fragment SMILES,
                bond type, and positional information.
                - fragments (list): List of RDKit molecule objects for each fragment.
                - atom_tokens (list): List of atom tokens from the original molecule.
        """
        mol = Chem.rdmolops.RemoveHs(mol)  # Remove explicit hydrogens
        atom_tokens = [atom.GetSymbol() for atom in mol.GetAtoms()]  # Atom tokens for later reference

        # Create a new editable molecule
        new_mol = Chem.RWMol(mol)

        # Assign AtomMapNum to track original atom indices
        for atom in new_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())

        sep_sets = []  # List to store bonds selected for splitting
        sep_sets = self.separate_functional(mol)

        # Identify bonds to split based on ring membership and connectivity
        for bond in mol.GetBonds():
            if bond.IsInRing():  # Skip bonds inside rings
                continue
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()

            # If both atoms are inside a ring, split the bond
            if a1.IsInRing() and a2.IsInRing():
                if not (((a1.GetIdx(), a2.GetIdx()) in sep_sets) or
                    ((a2.GetIdx(), a1.GetIdx()) in sep_sets)):
                    sep_sets.append((a1.GetIdx(), a2.GetIdx()))
            # Split if one atom is in a ring and the other is highly connected
            elif ((a1.IsInRing() and a2.GetDegree() > min_non_ring_neighbors) 
                  or (a2.IsInRing() and a1.GetDegree() > min_non_ring_neighbors)):
                if not (((a1.GetIdx(), a2.GetIdx()) in sep_sets) or
                    ((a2.GetIdx(), a1.GetIdx()) in sep_sets)):
                    sep_sets.append((a1.GetIdx(), a2.GetIdx()))

        # Dictionary to map original atoms to split indices
        atommap_dict = defaultdict(list)
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
        return count_labels, fragments, atom_tokens

    def separate_functional(self, molecule):
        """
        Find all broken bonds in a molecule by identifying connections between groups of atoms.

        Parameters:
            molecule (Chem.Mol): The molecule to analyze.

        Returns:
            list: A list of pairs of atom indices representing broken bonds (inter-group bonds).
        """
        # Step 1: Identify matches for substructures
        match_indices = []
        for key, value in self.func_group.items():
            matches = molecule.GetSubstructMatches(value)
            for match in matches:
                tmp = []
                for m in match:
                    atom = molecule.GetAtomWithIdx(m)
                    if not atom.IsInRing():
                        tmp.append(m)
                match = tuple(tmp)
                match_indices.append(match)

        # Step 2: Group matched atoms
        groups = []
        mapping = {}
        for tup in match_indices:
            current_groups = set(mapping.get(num, -1) for num in tup)
            current_groups = {group for group in current_groups if group != -1}
            
            if current_groups:
                # Merge groups if necessary
                new_group = set(tup)
                for group_index in current_groups:
                    new_group.update(groups[group_index])
                    groups[group_index] = set()  # Mark group as merged
                groups.append(new_group)
                new_group_index = len(groups) - 1
                for num in new_group:
                    mapping[num] = new_group_index
            else:
                # Create a new group
                new_group = set(tup)
                groups.append(new_group)
                new_group_index = len(groups) - 1
                for num in new_group:
                    mapping[num] = new_group_index

        # Step 3: Identify unmatched atoms
        matched_atoms = set(atom for group in groups for atom in group)
        all_atoms = set(range(molecule.GetNumAtoms()))
        unmatched_atoms = all_atoms - matched_atoms

        # Step 4: Group unmatched atoms based on connectivity
        visited = set()
        unmatched_groups = []

        def dfs(atom_idx, current_group):
            """Depth-first search to group connected unmatched atoms."""
            visited.add(atom_idx)
            current_group.add(atom_idx)
            for neighbor in molecule.GetAtomWithIdx(atom_idx).GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in unmatched_atoms and neighbor_idx not in visited:
                    dfs(neighbor_idx, current_group)

        for atom in unmatched_atoms:
            if atom not in visited:
                current_group = set()
                dfs(atom, current_group)
                unmatched_groups.append(current_group)

        # Step 5: Combine matched and unmatched groups
        all_groups = [group for group in groups if group] + unmatched_groups

        # Step 6: Identify inter-group bonds (broken bonds)
        broken_bonds = []
        atom_to_group = {}
        for group_id, group in enumerate(all_groups):
            for atom in group:
                atom_to_group[atom] = group_id

        for bond in molecule.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            group1 = atom_to_group.get(a1, -1)
            group2 = atom_to_group.get(a2, -1)

            # If atoms are in different groups, the bond is broken
            if group1 != group2:
                broken_bonds.append((a1, a2))

        return broken_bonds
    


if __name__ == "__main__":
    matcher =  SubstructureMatcher()

    smiles_list = [
        'OC1(O)CCCCC1',
        "CC(C)CCCCCCOC(=O)c1cc(C(=O)OCCCCCCC(C)C)c(C(=O)O)cc1C(=O)O",
        'CC#CC=CC(=O)CCC',
        'CN(C)C(N(C)C)(N(C)c1ccccc1)N(C)c1ccccc1',
        'Cc1ccc(-c2cc(O)cc3c2Cc2ccccc2C3)cc1',
        'C=C(C)C(C)(C)C(O)CCCCCCCCC(C)=O',
    ]

    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        count_labels, fragments, atom_tokens = matcher.split_molecule_by_functional_groups(molecule)
        print(f"SMILES: {smiles}")
        for count_label, fragment in zip(count_labels, fragments):
            print(f"Fragment: {count_label[0]}, Bond: {count_label[1:]}")
        print()
            

if False:
    # 分子を定義
    # target_smiles = "CC(O)[N+](=O)[O-]"  # 硝酸エチル
    target_smiles = "CCCO"  # プロパン-2-オン
    target_molecule = Chem.MolFromSmiles(target_smiles)

    # 部分構造を定義 (二重結合1つと単結合2つを持つ炭素)
    # substructure = Chem.MolFromSmarts('[C;X3](=O)-[!#1]')
    # substructure = Chem.MolFromSmarts('[!#1;X1,X2,X3](=O)-,=[!#1]')
    substructure = Chem.MolFromSmarts('[OH]')

    # 部分構造が含まれるか確認
    if target_molecule.HasSubstructMatch(substructure):
        print("指定された炭素構造が見つかりました！")

        # マッチする炭素原子のインデックスを取得
        matches = target_molecule.GetSubstructMatches(substructure)
        print("部分構造に一致する炭素のインデックス:", matches)

        # 詳細を表示
        for match in matches:
            print(f"一致した炭素の原子インデックス: {match[0]}")
            atom = target_molecule.GetAtomWithIdx(match[0])
            for bond in atom.GetBonds():
                neighbor_idx = bond.GetOtherAtomIdx(match[0])
                bond_type = bond.GetBondType()
                print(f"  結合相手: {neighbor_idx}, 結合種類: {bond_type}")
    else:
        print("指定された炭素構造は見つかりませんでした。")
