from rdkit import Chem
import torch
import rdkit
from rdkit import RDLogger
from inspect import getmembers, isfunction
from rdkit.Chem import Descriptors
import time
from collections import OrderedDict
from rdkit.Contrib.SA_Score import sascorer
import numpy as np
import pandas as pd
from collections import OrderedDict
from rdkit.Chem import Descriptors
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import re
def calculate_property_v1(smiles,):
    with open('property_name.txt', 'r') as f:
        names = [n.strip() for n in f.readlines()]

    descriptor_dict = OrderedDict()
    for n in names:
        if n == 'QED':
            descriptor_dict[n] = lambda x: Chem.QED.qed(x)
        elif n == "SAS":
            descriptor_dict[n] = lambda x: sascorer.calculateScore(x)
        else:
            descriptor_dict[n] = getattr(Descriptors, n)
            
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    if smiles == '' or mol is None:
        return torch.zeros(len(descriptor_dict), dtype=torch.float)
    output = []
    for i, descriptor in enumerate(descriptor_dict):
        # print(descriptor)
        output.append(descriptor_dict[descriptor](mol))
    return torch.tensor(output, dtype=torch.float)


def format_number(number):

    sign = '_+_ ' if number[0] != '-' else '_-_ '
    number = number.lstrip('-')
    integer_part, decimal_part = number.split('.')
    n=len(integer_part)
    result = sign + ' '.join(f'_{digit}_{n-i-1}_' for i, digit in enumerate(integer_part, start=0))
    result += ' _._ '
    result += ' '.join(f'_{digit}_-{i}_' for i, digit in enumerate(decimal_part, start=1))
    
    return result

def transform_string(data):
    prop_char=""
    keys=data.keys()
    for key in keys:
        value=str(f"{data[key]:.3f}")
        xx_formatted=format_number(value)
        prop_char = " ".join([prop_char,"<"+key+">",xx_formatted])
    return prop_char

def transform_string_generation(data):
    prop_char=""
    keys=data.keys()
    count=0
    for key in keys:
        if data[key] == None:
            count+=1
    if count==3:
        prop_char=" <qed>"+" [UNK]"*20
        return prop_char
    else:
        for key in keys:
            if data[key] == None:
                prop_char=prop_char+" [UNK]"*7
            else:
                value=str(f"{data[key]:.3f}")
                xx_formatted=format_number(value)
                prop_char = " ".join([prop_char,"<"+key+">",xx_formatted])
    return prop_char

def extract_nums_from_text(texts):
    all_nums = []
    modified_text=[]
    for text in texts:
        nums = re.findall(r'-?\d+\.\d+', text)
        nums = list(map(float, nums)) 
        modified_text .append( re.sub(r'-?\d+\.\d+', '[NUM]', text))
        all_nums.append(nums)
    return all_nums,modified_text
def prop_token(input_strings,tokenizer_1):
    nums_list, modified_text= extract_nums_from_text(input_strings)
    prop_input = tokenizer_1(modified_text, padding='max_length', return_tensors='pt', truncation=True, max_length=108)
    input_ids = prop_input['input_ids']
    num_sequences = torch.ones_like(input_ids, dtype=torch.float)
    num_token_id = tokenizer_1.convert_tokens_to_ids("[NUM]")
    for i, input_id_row in enumerate(input_ids):
        num_index = 0
        for j, token_id in enumerate(input_id_row):
            if token_id == num_token_id and num_index < len(nums_list[i]):
                num_sequences[i, j] = nums_list[i][num_index]
                num_index += 1
    prop_input['num_sequence'] = num_sequences
    return prop_input
################################################################
if __name__ == '__main__':
    # st = time.time()
    # smiles='CCOC(=O)Cc1ccc(OC)c(C#N)c1O'#'Cc1cccc(CNNC(=O)C2(Cc3ccccc3CN=[N+]=[N-])N=C(c3ccc(OCCCO)cc3)OC2c2ccc(Br)cc2)c1'
    # output = calculate_property_v1(smiles)
    # print(output, output.size())
    # print(time.time() - st)
    # print(rdkit.__version__)
    prop={"qed":None,"logp":None,"sas":None}
    prop_char=transform_string_generation(prop)
    ix=prop_token
    print(prop_char)
    print(len(prop_char.split()))
