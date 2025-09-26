from torch.utils.data import Dataset
import random
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

class Pretrain_Dataset(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        if data_length is not None:
            with open(data_path, 'r') as f:
                for _ in range(data_length[0]):
                    f.readline()
                lines = []
                for _ in range(data_length[1] - data_length[0]):
                    lines.append(f.readline())
        else:
            with open(data_path, 'r') as f:
                lines = f.readlines()
        self.data = [l.strip() for l in lines[1:]]
        if shuffle:
            random.shuffle(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        # print("________property_mean:", self.property_mean)
        char_all=(self.data[index]).split(',')
        if len(char_all)>7:
            print("index:",index)
        ss,prop_char=char_all[0],char_all[4]
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(ss), isomericSmiles=False, canonical=True)
        proper_char=prop_char
        scaffold_smiles=MurckoScaffoldSmiles(smiles)
        return '[CLS]' + smiles, '[CLS] '+proper_char , '[CLS]<scaffold>'+scaffold_smiles

    


    
