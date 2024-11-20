from model.utils import *
from model.stats_utils import *
from model.vocab import *

from collections import Counter
from tqdm import tqdm
import os

def main(args):
    os.makedirs(args.save_dir, exist_ok = True)

    suppl = None
    mols = []
    if args.smiles_path.endswith('.sdf'):
        print("Reading Chem.Mol from SDF file")
        suppl = Chem.SDMolSupplier(args.smiles_path)
    else:
        all_smiles = read_smiles(args.smiles_path)
        print("Number of SMILES entered: ", len(all_smiles))
        
        cou = 0
        if args.save_mol:
            writer = Chem.SDWriter(args.save_dir + "/molecules.sdf")
        with tqdm(range(len(all_smiles)), total=len(all_smiles), desc = 'Process 1/2') as pbar:
            for i in pbar:
                try:
                    mol = Chem.MolFromSmiles(all_smiles[i])
                    mol = sanitize(mol, kekulize = False)
                    mol = Chem.RemoveHs(mol)
                    mols.append(mol)

                    if args.save_mol:
                        writer.write(mol)
                except:
                    cou += 1
                if i % 1000 == 0:
                    pbar.set_postfix({'Error': cou})
            if args.save_mol:
                writer.close()
        if cou > 0:
            raise ValueError("There might be some errors. Check your SMILES data.")

    # print("Process 2/9 is running", end = '...')
    substruct_matcher = SubstructureMatcher()
    count_labels = Counter() #(substructureSMILES,(AtomIdx in substructure, join order)xN)->frequency of use of label
    # fragments = []
    atom_tokens = set()

    if suppl is not None:
        iterator = tqdm(enumerate(suppl), total = len(suppl), desc='Process 2/2')
    else:
        iterator = tqdm(enumerate(mols), total = len(mols), desc='Process 2/2')

    for i, m in iterator:
        try:
            cl, frag, a_tokens = substruct_matcher.split_molecule_by_functional_groups(m)
            count_labels.update(cl)
            # fragments.append(frag)
            atom_tokens.update(a_tokens)
        except:
            pass
    count_labels = dict(count_labels.most_common())
    atom_tokens = atom_tokens_sort(list(atom_tokens))
    if 'b' in  args.save_cnt_label:
        dill.dump(count_labels, open(args.save_dir + "/count_labels.pkl", "wb"))
    if 't' in  args.save_cnt_label:
        with open(args.save_dir + "/count_labels.txt", "w") as f:
            f.write("\n".join([str(k) + '\t' + str(v) for k, v in count_labels.items()]))
    with open(args.save_dir + "/atom_tokens.txt", "w") as f:
        f.write("\n".join([str(k) for k in atom_tokens]))
    distribution_df = plot_counter_distribution(count_labels, save_file=os.path.join(args.save_dir, "plot", 'vocab_count_labels_0.png'), bin_width=1, y_scale='log')
    distribution_df.to_csv(os.path.join(args.save_dir, "plot", 'vocab_count_labels.tsv'), index=True, sep='\t')
    distribution_df = plot_counter_distribution(count_labels, save_file=os.path.join(args.save_dir, "plot", 'vocab_count_labels_5.png'), bin_width=1, display_thresh=5, y_scale='log')
    print('done')
    

# python vocab.py -f /workspaces/hgraph/mnt/Ms2z/data/SMILES/pubchem/pubchem_smiles_1M.pkl
# python vocab.py -f /workspaces/hgraph/mnt/Ms2z/data/SMILES/pubchem/pubchem_smiles_10k.pkl
if __name__ == "__main__":
    import warnings
    import argparse
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--smiles_path", type = str, required=True, help = "Path of SMILES data for input compounds (delete SMILES containing '.')")
    parser.add_argument('-o', '--save_dir', type = str, required=True, help = "Path to save created data")
    parser.add_argument('--save_cnt_label', type = str, choices=['b', 't', 'bt', 'tb'], default = 'b', help = "Type of count label to save")
    parser.add_argument('--save_mol', action='store_true', help = "Save the created molecule data with RDKit")
    # parser.add_argument("-freq", "--frequency", type = int, default = 5,
    #                     help = "Threshold frequencies at decomposition")
    # parser.add_argument("-fpbit", type = int, default = 2048,
    #                     help = "Number of bits of ECFP")
    # parser.add_argument("-r", "--radius", type = int, default = 2,
    #                     help = "Effective radius of ECFP")
    # parser.add_argument("--save_path", type = str,
    #                     default = "./save_data", help = "Path to save created data")
    args = parser.parse_args()
    main(args)