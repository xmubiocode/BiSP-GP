from rdkit import Chem
from moses.utils import get_mol
from bisect import bisect_left
import pandas as pd

def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1
def check_novelty(gen_smiles, train_smiles): # gen: say 788, train: 120803
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  # [1]*45
        novel = len(gen_smiles) - sum(duplicates)  # 788-45=743
        print("novel:",novel)
        novel_ratio = novel/len(gen_smiles)  # 743*100/788=94.289
    print("novelty: {:.3f}%".format(novel_ratio))
    return novel_ratio

def check_novelty_1(gen_smiles, train_smiles):
    train_smiles=list(train_smiles)
    train_smiles.sort()
    count = 0
    for l in gen_smiles:
        if BinarySearch(train_smiles, l) < 0:
            count += 1
    print('novelty_1:', count / len(gen_smiles))
    return count / len(gen_smiles)

def canonic_smiles(gendata):
    lines=[]
    for l in gendata:
        mol=Chem.MolFromSmiles(l)
        if mol:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            lines.append(smiles)
    return lines

##计算得到唯一性（U）和新颖性(N)
def calculate_un(results, train_data):
    canon_smiles = canonic_smiles(results)
    v=len(canon_smiles)
    unique_smiles = list(set(canon_smiles))#
    novel_ratio = check_novelty(unique_smiles, set(train_data))   # replace 'source' with 'split' for moses

    return v/len(results),len(unique_smiles)/v, novel_ratio



# train_data="/home/syy/model_pretrain_CL_v2/data/data_char_1.csv"
# gen_data="/home/syy/model_pretrain_CL_v2/generated_molecules.txt"
# cand=pd.read_csv(gen_data,header=None)[0].values
# train_smile=pd.read_csv(train_data)
# va,un,na=calculate_un(cand,train_smile["SMILES"][0:20000000])
# print (f'validity:{va};uniqueness:{un};novelty:{na}')