from rdkit import Chem
import dill
from tqdm import tqdm

try:
    from .utils import *
except ImportError:
    from utils import *
    

class Fragmentizer:
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
        atom_tokens = get_atom_tokens(mol)  # Atom tokens for later reference

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
        fragment_mols = [Chem.MolFromSmiles(fragment) for fragment in new_smiles.split('.')]
        fragment_mols = [sanitize(fragment, kekulize = False) for fragment in fragment_mols]
        
        fragment_labels = []
        fragment_atom_mapping = []  # To track atom indices corresponding to each fragment

        for i, fragment_mol in enumerate(fragment_mols):
            order_list = [] #Stores join orders in the substructures
            fragment_label = []
            frag_atom_indices = []  # To store original atom indices for this fragment
            frag_mol = copy.deepcopy(fragment_mol)
            for atom in frag_mol.GetAtoms():
                frag_mol.GetAtomWithIdx(atom.GetIdx()).SetAtomMapNum(0)
            frag_smi = Chem.MolToSmiles(sanitize(frag_mol, kekulize = False))
            #Fix AtomIdx as order changes when AtomMap is deleted.
            atom_order = list(map(int, frag_mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
            a_dict = {}
            for i, atom in enumerate(fragment_mol.GetAtoms()):
                amap = atom.GetAtomMapNum()
                a_dict[i] = amap
                if amap in list(atommap_dict.keys()):
                    order_list.append(atommap_dict[amap])

            for order in atom_order:
                frag_atom_indices.append(a_dict[order])

            order_list = sorted(order_list)
            fragment_label.append(frag_smi)
            for atom in fragment_mol.GetAtoms():
                amap = atom.GetAtomMapNum()
                if amap in list(atommap_dict.keys()):
                    for seq_idx in atommap_dict[amap]:
                        for amap2 in list(atommap_dict.keys()):
                            if (seq_idx in atommap_dict[amap2]) and (amap != amap2):
                                bond = mol.GetBondBetweenAtoms(amap, amap2)
                                bond_type = bond.GetBondType()
                                bond_type_str = ""
                                if bond_type == Chem.BondType.SINGLE:
                                    bond_type_str = "-"
                                elif bond_type == Chem.BondType.DOUBLE:
                                    bond_type_str = "="
                                elif bond_type == Chem.BondType.TRIPLE:
                                    bond_type_str = "#"
                                elif bond_type == Chem.BondType.AROMATIC:
                                    bond_type_str = ":"

                                fragment_label.append(atom_order.index(atom.GetIdx()))
                                fragment_label.append(bond_type_str)
                                # count_label.append(order_list.index(atommap_dict[amap]) + 1)

            fragment_label, frag_atom_indices = normalize_bond_info(fragment_label, frag_atom_indices)
            fragment_atom_mapping.append(frag_atom_indices)
            fragment_labels.append(tuple(fragment_label))
        return fragment_labels, fragment_mols, atom_tokens, fragment_atom_mapping

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
    
    def assign_vocabulary(self, count_labels):
        """
        Assigns a vocabulary to the count labels for encoding.

        Args:
            count_labels (list): List of tuples containing fragment SMILES,
                bond type, and positional information.

        Returns:
            dict: A dictionary mapping unique vocabulary tokens to integers.
        """
        vocab = set()
        for count_label in count_labels:
            vocab.update(count_label)
        vocab = sorted(list(vocab))
        vocab_dict = {token: idx for idx, token in enumerate(vocab)}
        return vocab_dict

    @staticmethod
    def load_token_count(path, no_count=False):
        if path.endswith('.pkl'):
            count_labels = dill.load(open(path, 'rb'))
            return count_labels
        else:
            with open(path, 'r') as f:
                if not no_count:
                    count_labels = [(eval(line.strip().split('\t')[0]), int(line.strip().split('\t')[1]))
                        for line in f]
                else:
                    count_labels = [(eval(line.strip().split('\t')[0]), 1)
                        for line in f]
            return Counter(dict(count_labels))

class Vocab:
    def __init__(self, atom_tokens_path, fragment_label_path, threshold):
        atom_tokens = Fragmentizer.load_token_count(atom_tokens_path, no_count=True)
        fragment_labels = Fragmentizer.load_token_count(fragment_label_path)
        self.vocab = set(atom_tokens.keys())
        self.vocab.update([k for k in fragment_labels.keys() if fragment_labels[k] >= threshold])
        self.vocab = {v:i for i, v in enumerate(self.vocab)}

        self.tree = FragmentTree()
        self.tree.add_fragment_list(self.vocab.keys())
        
        self.fragmentizer = Fragmentizer()

    def assign_vocab(self, mol):
        fragment_labels, fragment_mols, atom_tokens, fragment_atom_mapping = self.fragmentizer.split_molecule_by_functional_groups(mol)
          
        for i, frag_label in enumerate(fragment_labels):
            print(f'{frag_label}: {self.vocab.get(frag_label, -1)}, G: {fragment_atom_mapping[i]}')
        print()

        frag_to_idx = {}
        for i, frag_label in enumerate(fragment_labels):
            if frag_label in frag_to_idx:
                continue
            idx = self.vocab.get(frag_label, -1)
            if idx == -1:
                ring_info = fragment_mols[i].GetRingInfo()
                # Nonvalid fragment if it contains a ring
                if ring_info.NumRings() > 0:
                    return None

            frag_to_idx[frag_label] = idx
        
        atom_scores = calculate_advanced_scores(mol)
        max_atom_score_tuple = max(atom_scores, key=lambda x: x[2])
        max_atom_score_idx = max_atom_score_tuple[0]
        # max_atom_score = max_atom_score_tuple[2]

        for i, atom_mapping in enumerate(fragment_atom_mapping):
            if max_atom_score_idx in atom_mapping:
                ori_frag_idx = i
                break

            
        vocab_list = []
        visited = set()
        current_frag_idx = -1

        def dfs(parent_info, start_bond_info):
            nonlocal current_frag_idx
            frag_idx, s_bond_idx = start_bond_info
            fragment_label = fragment_labels[frag_idx]
            atom_mapping = fragment_atom_mapping[frag_idx]

            vocab_idx = frag_to_idx[fragment_label]
            if vocab_idx == -1:
                pass
            else:
                vocab_list.append({'smiles': fragment_label, 'idx': vocab_idx, 'next': []})
                current_frag_idx += 1
                order_frag_idx = current_frag_idx
                visited.update(atom_mapping)

            joint_dict_group = {}
            joint_dict = []
            for i, fragment_l in enumerate(zip(fragment_label[1::2], fragment_label[2::2])):
                joint_dict.append(fragment_l)
                if i == s_bond_idx:
                    continue
                if fragment_l not in joint_dict_group:
                    joint_dict_group[fragment_l] = []
                joint_dict_group[fragment_l].append(i)

            for fragment_l, joint_idx_list in joint_dict_group.items():
                joint_atom_idx = atom_mapping[fragment_l[0]]
                atom = mol.GetAtomWithIdx(joint_atom_idx)

                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    next_atom_infoes = []
                    if neighbor_idx not in visited:
                        bond = mol.GetBondBetweenAtoms(joint_atom_idx, neighbor_idx)
                        bond_type = bond.GetBondType()
                        bonding_type = chem_bond_to_token(bond_type)
                        if bonding_type == fragment_l[1]:
                            next_atom_infoes.append((joint_idx_list.pop(0), bonding_type, neighbor_idx))
                
                if len(joint_idx_list) != 0:
                        raise ValueError('Invalid Fragment')

                for next_atom_info in next_atom_infoes:
                    for i, atom_mapping2 in enumerate(fragment_atom_mapping):
                        if next_atom_info[2] in atom_mapping2:
                            next_frag_idx = i
                            next_bond_pos = atom_mapping2.index(next_atom_info[2])
                            break
                    else:
                        raise ValueError('Not Found Next Fragment')
                    next_frag_idx = dfs(parent_info=(order_frag_idx, next_atom_info[0]), start_bond_info=(next_frag_idx, next_bond_pos))
                    vocab_list[order_frag_idx]['next'].append((next_atom_info[0], next_atom_info[1], (next_frag_idx, next_bond_pos)))
            return order_frag_idx
            

        dfs(parent_info=(-1,-1), start_bond_info=(ori_frag_idx, -1))
        pass
        
        

        
        

            


class FragmentNode:
    """
    Represents a node in the fragment tree.
    Each node has a depth, a dictionary of child nodes, and a list of leaves.
    """
    def __init__(self, depth):
        self.children = {}  # Dictionary of child nodes
        self.depth = depth  # Depth of the node in the tree
        self.leaves = []    # List of leaf values (associated fragments)
    
    def add_child(self, keys:list, value):
        """
        Add a child to the current node recursively.
        :param keys: A list of keys representing the path to the value.
        :param value: The value to store at the leaf node.
        """
        key = keys[self.depth]
        if key not in self.children:
            self.children[key] = FragmentNode(self.depth + 1)

        if self.depth < len(keys) - 1:
            self.children[key].add_child(keys, value)
        else:
            self.children[key].leaves.append(value)

    def display(self, level=0, prefix=""):
        """
        Recursively print the tree structure for visualization with tree-like characters.
        :param level: Current indentation level for the node.
        :param prefix: Prefix string used for tree visualization.
        """
        # Display the current node's leaves
        if self.leaves:
            print(f"{prefix}+- : {self.leaves}")

        # Display the children with tree structure
        for i, (key, child) in enumerate(self.children.items()):
            is_last = i == len(self.children) - 1
            connector = "`-" if is_last else "|-"
            child_prefix = prefix + ("   " if is_last else "|  ")
            print(f"{prefix}{connector} {key}")
            child.display(level + 1, child_prefix)

    def _write_to_file(self, file, prefix=""):
        """
        Recursively write the tree structure to an open file object.
        :param file: An open file object to write the tree structure.
        :param prefix: Prefix string used for tree visualization.
        """
        if self.leaves:
            file.write(f"{prefix}+- : {self.leaves}\n")

        for i, (key, child) in enumerate(self.children.items()):
            is_last = i == len(self.children) - 1
            connector = "`-" if is_last else "|-"
            child_prefix = prefix + ("   " if is_last else "|  ")
            file.write(f"{prefix}{connector} {key}\n")
            child._write_to_file(file, child_prefix)
    
class FragmentTree:
    """
    Represents the fragment tree structure, with methods for adding fragments,
    performing DFS traversal, and saving/loading the tree.
    """
    bond_sord_order = {'-': 0, '=': 1, '#': 2, ':': 3}

    def __init__(self):
        self.root = FragmentNode(depth=0)
    
    def add_fragment(self, fragment, fragment_mol):
        """
        Add a fragment to the tree based on the molecular structure and binding sites.
        :param fragment: A tuple representing the fragment structure.
        :param fragment_mol: The RDKit molecule object for the fragment.
        """
        binding_sites_len = (len(fragment) - 1) // 2
        binding_sites = list(zip(fragment[1::2], fragment[2::2]))

        for i, binding_site in enumerate(binding_sites):
            traversal_order = [[binding_site[1]]]
            visited = set()
            self.dfs(fragment_mol, visited, traversal_order, prev_atom_indices=[binding_site[0]])
            traversal_order = [sorted(v, key=lambda x: self.bond_sord_order.get(x, x)) for v in traversal_order]
            tree_keys = [','.join(map(str, v)) for v in traversal_order]
            self.root.add_child(tree_keys, (fragment, i))

    def dfs(self, mol, visited, traversal_order, prev_atom_indices):
        """
        Perform a depth-first search to explore the molecule graph and record traversal order.
        :param mol: The RDKit molecule object.
        :param visited: A set of visited atom indices.
        :param traversal_order: The order of atom and bond traversal.
        :param prev_atom_indices: List of atom indices to traverse from.
        """
        next_atom_indices = []
        current_symbols = []
        current_bonds = []

        for atom_index in prev_atom_indices:
            if atom_index in visited:
                continue
            atom = mol.GetAtomWithIdx(atom_index)
            visited.add(atom_index)
            current_symbols.append(atom.GetAtomicNum())

            for neighbor in atom.GetNeighbors():
                neighbor_index = neighbor.GetIdx()
                if neighbor_index not in visited:
                    bond = mol.GetBondBetweenAtoms(atom_index, neighbor_index)
                    bond_type = bond.GetBondType()
                    bonding_type = chem_bond_to_token(bond_type)
                    current_bonds.append(bonding_type)
                    next_atom_indices.append(neighbor_index)

        next_atom_indices = list(set(next_atom_indices))
        traversal_order.append(current_symbols)

        if len(next_atom_indices) == 0:
            return traversal_order
        else:
            traversal_order.append(current_bonds)
            self.dfs(mol, visited, traversal_order, next_atom_indices)

    # Example function to build the tree from a list of fragments
    def add_fragment_list(self, fragments, fragment_mols=None):
        # Root node for the tree
        # root = FragmentNode('', 'Root')
        for i, fragment in tqdm(enumerate(fragments), total=len(fragments), mininterval=0.5, desc='Building Fragment Tree'):
            if fragment_mols is None:
                fragment_mol = Chem.MolFromSmiles(fragment[0])
            else:
                fragment_mol = fragment_mols[i]
            self.add_fragment(fragment, fragment_mol)

    def save_tree(self, file_path):
        """
        Save the fragment tree to a file in binary format using dill.
        :param file_path: Path to the file where the tree will be saved.
        """
        with open(file_path, "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load_tree(file_path):
        """
        Load a fragment tree from a binary file.
        :param file_path: Path to the file from which the tree will be loaded.
        :return: The loaded FragmentTree object.
        """
        with open(file_path, "rb") as file:
            return dill.load(file)

    def display_tree(self):
        """
        Display the entire tree structure for visualization.
        """
        self.root.display()

    def save_to_file_incrementally(self, file_path):
        """
        Save the tree structure to a file incrementally while keeping the file open.
        :param file_path: Path to the file where the tree will be saved.
        """
        with open(file_path, "w") as file:
            self.root._write_to_file(file)

if __name__ == "__main__":
    if False:
        matcher =  Fragmentizer()
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
    elif True:
        atom_tokens_file = '/workspaces/hgraph/mnt/Ms2z/data/graph/pubchem/atom_tokens.txt'
        counter_labels_file = '/workspaces/hgraph/mnt/Ms2z/data/graph/pubchem/count_labels.pkl'
        vocab = Vocab(atom_tokens_file, counter_labels_file, threshold=0)

        smiles = 'CC(C)CCCCCCOC(=O)c1cc(C(=O)OCCCCCCC(C)C)c(C(=O)O)cc1C(=O)O'
        # smiles = 'Oc1c(Br)cc2cc1C=CC2'
        # smiles = 'C=CNc1ccccc1C'
        # smiles = 'c1cncc(-c2nc(-c3cccc4ccccc34)nc(-c3ccc4ccc5c(-c6nc(-c7cccnc7)nc(-c7cccc8ccccc78)n6)ccc6ccc3c4c65)n2)c1'
        smiles = 'C#Cc1c2ccccc2cc2cccc(-c3ccc(-c4cccc5cc6ccccc6c(C#C)c45)cc3)c12'
        # smiles = 'c1ccccc1'
        # smiles = 'COc1ccccc1-c1cccc2c1OCO2'
        # smiles = 'c1ccc2cc3ccccc3cc2c1'
        mol = Chem.MolFromSmiles(smiles)
        print(f"Input SMILES: {smiles}")
        suc = vocab.assign_vocab(mol)
        print()
    
    elif False:
        # Example usage
        fragments = [
            ('C=CN(COC)-C(C)=O', 1, '-', 2, '-'),  # Example fragment
            ('N=C', 1, '='),   # Another fragment
            ('C=C', 0, '='),   # Another fragment
        ]

        fragment_mols = [Chem.MolFromSmiles(frag[0]) for frag in fragments]

        # Build the tree
        fragment_tree = FragmentTree()
        fragment_tree.add_fragment_list(fragments, fragment_mols)

        # Print the tree structure
        file_path = '/workspaces/hgraph/mnt/Ms2z/data/fragment_tree.pkl'
        fragment_tree.save_tree(file_path)
        fragment_tree = FragmentTree.load_tree(file_path)
        fragment_tree.display_tree()
        fragment_tree.save_to_file_incrementally(file_path.replace('.pkl', '.txt'))


    


            

