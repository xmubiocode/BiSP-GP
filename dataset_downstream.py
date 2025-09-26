from torch.utils.data import Dataset
import torch
import random
import pandas as pd
from rdkit import Chem
import pickle
from rdkit import RDLogger
from calc_property import calculate_property
from pysmilesutils.augment import MolAugmenter
RDLogger.DisableLog('rdApp.*')

def format_number(number):
    # 提取正负号
    sign = '_+_ ' if number[0] != '-' else '_-_ '
    number = number.lstrip('-')

    # 处理整数和小数部分
    integer_part, decimal_part = number.split('.')
    
    # 处理整数部分
    n=len(integer_part)
    result = sign + ' '.join(f'_{digit}_{n-i-1}_' for i, digit in enumerate(integer_part, start=0))
    
    # 处理小数点
    result += ' _._ '
    
    # 处理小数部分
    result += ' '.join(f'_{digit}_-{i}_' for i, digit in enumerate(decimal_part, start=1))
    
    return result


class SMILESDataset_BACEC(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['mol']), isomericSmiles=False, canonical=True)
        value = int(self.data[index]['Class'])

        return '[CLS]' + smiles, value


class SMILESDataset_BACER(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value = torch.tensor(self.data[index]['target'].item())
        return '[CLS]' + smiles, value


class SMILESDataset_LIPO(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['smiles'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        value = torch.tensor(self.data[index]['exp'].item())

        return '[CLS]' + smiles, value


class SMILESDataset_Clearance(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['smiles'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        value = torch.tensor(self.data[index]['target'].item())

        return '[CLS]' + smiles, value


class SMILESDataset_BBBP(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data)) if Chem.MolFromSmiles(data.iloc[i]['smiles'])]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        label = int(self.data[index]['p_np'])

        return '[CLS]' + smiles, label


class SMILESDataset_ESOL(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.data[index]['smiles'])
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        value = torch.tensor(self.data[index]['ESOL predicted log solubility in mols per litre'].item())

        return '[CLS]' + smiles, value


class SMILESDataset_Freesolv(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value = (self.data[index]['target'] - self.value_mean) / self.value_std

        return '[CLS]' + smiles, value

class SMILESDataset_Malaria(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        # value_char = format_number(str(round(self.data[index]['activity'],3)))
        value = torch.tensor(self.data[index]['activity'], dtype=torch.float32)  #(self.data[index]['activity'] - self.value_mean) / self.value_std
        return '[CLS]' + smiles, value


class SMILESDataset_CEP(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value = torch.tensor(self.data[index]['PCE'],dtype=torch.float32)  #(self.data[index]['PCE'] - self.value_mean) / self.value_std

        return '[CLS]' + smiles, value





class SMILESDataset_Clintox(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]
        self.n_output = 2

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False)
        value = torch.tensor([float(self.data[index]['FDA_APPROVED']), float(self.data[index]['CT_TOX'])])

        return '[CLS]' + smiles, value


class SMILESDataset_SIDER(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        data = pd.read_csv(data_path)
        self.data = [data.iloc[i] for i in range(len(data))]
        self.n_output = 27

        if shuffle: random.shuffle(self.data)
        if data_length is not None: self.data = self.data[data_length[0]:data_length[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]['smiles']), isomericSmiles=False, canonical=True, kekuleSmiles=False)
        value = self.data[index].values.tolist()[1:]
        value = torch.tensor([i.item() for i in value])
        return '[CLS]' + smiles, value







