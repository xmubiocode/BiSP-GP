import argparse
import torch
from pretrain_model import BiSP_GP
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
from calc_property import calculate_property_v1,transform_string_generation,transform_string
from torch.distributions.categorical import Categorical
from rdkit import Chem
import random
import numpy as np
import pickle
import warnings
from tqdm import tqdm
from bisect import bisect_left
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import json
warnings.filterwarnings(action='ignore')
from unlti import calculate_un
#from dataset_CL import SMILESDataset_pretrain
import pandas as pd
def BinarySearch(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1


def generate(model, image_embeds, text, stochastic=True, prop_att_mask=None, k=None):
    text_atts = torch.where(text == 0, 0, 1)
    if prop_att_mask is None:   prop_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
    token_output = model.text_encoder(text,
                                      attention_mask=text_atts,
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=prop_att_mask,
                                      return_dict=True,
                                      is_decoder=True,
                                      return_logits=True,
                                      )[:, -1, :]  # batch*300
    
    if k:
        p = torch.softmax(token_output, dim=-1)
        if stochastic:#stochastic=True
            output = torch.multinomial(p, num_samples=k, replacement=False)
            return torch.log(torch.stack([p[i][output[i]] for i in range(output.size(0))])), output
        else:
            output = torch.topk(p, k=k, dim=-1)  # batch*k
            return torch.log(output.values), output.indices
    if stochastic:#
        p = torch.softmax(token_output, dim=-1)
        m = Categorical(p)
        token_output = m.sample()
    else:
        token_output = torch.argmax(token_output, dim=-1)
    return token_output.unsqueeze(1)  # batch*1


@torch.no_grad()
def generate_with_property(model, properties=None,scaffold=None, n_sample=None, k=2, stochastic=True, prop_len=None):
    
    print(properties, scaffold, n_sample, k, stochastic, prop_len)
    device = model.device
    tokenizer = model.tokenizer
    # test
    model.eval()
    print(f"PV-to-SMILES generation in {'stochastic' if stochastic else 'deterministic'} manner with k={k}...")

    if scaffold is not None and properties is not None:
        scaffold_input=tokenizer(scaffold,padding='longest',return_tensors='pt',truncation=True,max_length=prop_len).to(device)
        scaffold_emdeds=model.text_encoder.bert(scaffold_input.input_ids[:,1:], scaffold_input.attention_mask[:,1:],return_dict=True,mode='text').last_hidden_state
        print("scaffold_emdeds",scaffold_input)
        prop_input=tokenizer(properties,padding='longest',return_tensors='pt',truncation=True,max_length=prop_len).to(device)#padding='longest'表示填充到最长的序列，padding="max_length"表示填充到最大长度为120
        print("prop_input",prop_input.input_ids)
        prop_embeds = model.property_encoder.bert(prop_input.input_ids, prop_input.attention_mask,return_dict=True,mode='text').last_hidden_state
        print("prop_embeds",prop_embeds.size())
        ###将属性和支架的embedding拼接,prop_embeds[:,:-1]表示去掉[SEQ]的embedding，scaffold_emdeds[:,1:]表示去掉[CLS]的embedding
        condation_input=torch.cat((prop_embeds[:,:-1],scaffold_emdeds[:,1:]),dim=1)
        condation_atten_mask=torch.cat((prop_input.attention_mask[:,:-1],scaffold_input.attention_mask[:,2:]),dim=1)
        # condation_input=torch.cat((prop_embeds,scaffold_emdeds),dim=1)
        # condation_atten_mask=torch.cat((prop_input.attention_mask,scaffold_input.attention_mask[:,1:]),dim=1)
        print("condation_input",condation_input)
        print("condation_input",condation_input.size())
        print("condation_atten_mask",condation_atten_mask)
        print("condation_atten_mask",condation_atten_mask.size())
    elif scaffold is not None and properties is None:
        scaffold_input=tokenizer(scaffold,padding='longest',return_tensors='pt',truncation=True,max_length=prop_len).to(device)
        scaffold_emdeds=model.text_encoder.bert(scaffold_input.input_ids[:,1:], scaffold_input.attention_mask[:,1:],return_dict=True,mode='text').last_hidden_state
        condation_input=scaffold_emdeds
        condation_atten_mask=scaffold_input.attention_mask[:,1:]
    elif scaffold is None and properties is not None:
        prop_input=tokenizer(properties,padding='longest',return_tensors='pt',truncation=True,max_length=prop_len).to(device)
        prop_embeds = model.property_encoder.bert(prop_input.input_ids, prop_input.attention_mask,return_dict=True,mode='text').last_hidden_state
        print("prop_input",prop_input.input_ids)
        condation_input=prop_embeds
        condation_atten_mask=prop_input.attention_mask
    else:
        print("Please input scaffold or properties")"""
    
    if scaffold is not None:
        scaffold_input=tokenizer(scaffold,padding='longest',return_tensors='pt',truncation=True,max_length=prop_len).to(device)
        scaffold_emdeds=model.text_encoder.bert(scaffold_input.input_ids[:,1:], scaffold_input.attention_mask[:,1:],return_dict=True,mode='text').last_hidden_state
        print("scaffold_emdeds",scaffold_input)
        prop_input=tokenizer(properties,padding='longest',return_tensors='pt',truncation=True,max_length=prop_len).to(device)
        print("prop_input",prop_input.input_ids)
        prop_embeds = model.property_encoder.bert(prop_input.input_ids, prop_input.attention_mask,return_dict=True,mode='text').last_hidden_state
        print("prop_embeds",prop_embeds.size())
        condation_input=torch.cat((prop_embeds[:,:-1],scaffold_emdeds[:,1:]),dim=1)
        condation_atten_mask=torch.cat((prop_input.attention_mask[:,:-1],scaffold_input.attention_mask[:,2:]),dim=1)
        # condation_input=torch.cat((prop_embeds,scaffold_emdeds),dim=1)
        # condation_atten_mask=torch.cat((prop_input.attention_mask,scaffold_input.attention_mask[:,1:]),dim=1)
        print("condation_input",condation_input)
        print("condation_input",condation_input.size())
        print("condation_atten_mask",condation_atten_mask)
        print("condation_atten_mask",condation_atten_mask.size())
    elif scaffold is None:
        prop_input=tokenizer(properties,padding='longest',return_tensors='pt',truncation=True,max_length=prop_len).to(device)
        prop_embeds = model.property_encoder.bert(prop_input.input_ids, prop_input.attention_mask,return_dict=True,mode='text').last_hidden_state
        print("prop_input",prop_input.input_ids)
        condation_input=prop_embeds
        condation_atten_mask=prop_input.attention_mask
    else:
        print("Please input scaffold or properties")
        
    for n in tqdm(range(n_sample)):
        product_input = torch.tensor([tokenizer.cls_token_id]).expand(1, 1).to(device)
        values, indices = generate(model, condation_input, product_input, stochastic=stochastic, prop_att_mask=condation_atten_mask, k=k)
        #print(values, indices, values.size(), indices.size())
        product_input = torch.cat([torch.tensor([tokenizer.cls_token_id]).expand(k, 1).to(device), indices.squeeze(0).unsqueeze(-1)], dim=-1)#K*2
        #print(product_input, product_input.size())
        current_p = values.squeeze(0)
        final_output = []
        for _ in range(100):
            values, indices = generate(model, condation_input, product_input, stochastic=stochastic,prop_att_mask=condation_atten_mask, k=k)#indices[1,K]
            k2_p = current_p[:, None] + values
            product_input_k2 = torch.cat([product_input.unsqueeze(1).repeat(1, k, 1), indices.unsqueeze(-1)], dim=-1)
            if tokenizer.sep_token_id in indices:
                
                ends = (indices == tokenizer.sep_token_id).nonzero(as_tuple=False)
                
                for e in ends:
                    p = k2_p[e[0], e[1]].cpu().item()
                    final_output.append((p, product_input_k2[e[0], e[1]]))
                    k2_p[e[0], e[1]] = -1e5
                if len(final_output) >= k ** 2:
                    break
                
                
            current_p, i = torch.topk(k2_p.flatten(), k)
            next_indices = torch.from_numpy(np.array(np.unravel_index(i.cpu().numpy(), k2_p.shape))).T
            product_input = torch.stack([product_input_k2[i[0], i[1]] for i in next_indices], dim=0)

        candidate_k = []
        final_output = sorted(final_output, key=lambda x: x[0], reverse=True)[:k]#
        
        for p, sentence in final_output:
            cdd = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence[:-1])).replace('[CLS]', '')
            candidate_k.append(cdd)
        if not stochastic:
            candidate.append(candidate_k[0])
        else:
            candidate.append(random.choice(candidate_k))
    return candidate

@torch.no_grad()
def tanimoto_similarity(scaffold_o, smiles):
    smiles_sca=MurckoScaffoldSmiles(smiles)
    mol_scaffold=FingerprintMols.FingerprintMol(Chem.MolFromSmiles(smiles_sca))
    cond_fp=FingerprintMols.FingerprintMol(Chem.MolFromSmiles(scaffold_o))
    similarity=TanimotoSimilarity(mol_scaffold,cond_fp)
    return similarity


@torch.no_grad()
def metric_eval(prop_input, cand,prop_idx,train_smiles=None,mean=None,std=None,scaffold=None):
    
    if prop_input is not None:
        mean=mean[prop_idx]
        std=std[prop_idx]
        prop_input=torch.tensor(prop_input)
        random.shuffle(cand)
        mse = []
        mse_1=[]
        mad=[]
        for i in range(len(cand)):
            try:
                prop_cdd = calculate_property_v1(cand[i])
                prop_cdd=prop_cdd[prop_idx]
                n_ref = (prop_input - mean) /std
                n_cdd = (prop_cdd - mean) / std
                mse.append((n_ref - n_cdd) ** 2)
                mse_1.append((prop_input-prop_cdd)**2)
                mad.append(abs(prop_input-prop_cdd))
        
            except:
                continue
        
        mad=torch.stack(mad, dim=0)
        print("mad: ", torch.mean(mad, dim=0))
    
    lines=[]
    for l in cand:
        mol=Chem.MolFromSmiles(l)
        if mol:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            lines.append(smiles)
    #print("valids",lines)
    #valids = [Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False, canonical=True) for l in valids]

    #lines = valids
    if scaffold is not None:
        print("scaffold",scaffold)
        tanimoto = []
        for j in range(len(lines)):
            tanimoto_score=tanimoto_similarity(scaffold, lines[j])
            if tanimoto_score>0.8:                                 ####
                tanimoto.append(tanimoto_score)
        print("tanimoto",len(tanimoto))
        print("tanimoto_ratio",len(tanimoto)/len(lines))
        
    v = len(lines)
    print("v",v)
    print('validity:', v / len(cand))

    liness = [Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False) for l in lines]
    liness = list(set(liness))
    u = len(liness)
    print("u",u)
    print('uniqueness:', u / v)
    
    va,un,na=calculate_un(cand,train_smiles)
    print (f'validity:{va};uniqueness:{un};novelty:{na}')
    print('train_smiles:', len(train_smiles))
    # train_smiles.sort()
    # count = 0
    # for l in liness:
    #     if BinarySearch(train_smiles, l) < 0:
    #         count += 1
    # print('novelty_num:', count )
    # print('novelty:', count / u)
    if prop_input is not None and scaffold is not None:
        return f"validity:{va} uniqueness:{un} novelty:{na} n_rmse:{rmse}-{rmse.mean().item()} tanimoto_ratio:{len(tanimoto)/len(lines)} rmse:{rmse_1}-{rmse_1.mean().item()} mad:{torch.mean(mad, dim=0)} std {torch.std(mad, dim=0)} \n",[va,un,na,len(tanimoto)/len(lines)]+torch.mean(mad, dim=0).tolist()+torch.std(mad, dim=0).tolist()
    elif prop_input is not None and scaffold is None:
        return f"validity:{va} uniqueness:{un} novelty:{na}  n_rmse:{rmse}-{rmse.mean().item()} rmse:{rmse_1}-{rmse_1.mean().item()} mad:{torch.mean(mad, dim=0)} std {torch.std(mad, dim=0)} \n",[va,un,na]+torch.mean(mad, dim=0).tolist()+torch.std(mad, dim=0).tolist()
    elif prop_input is None and scaffold is not None:
        return f"validity:{va} uniqueness:{un} novelty:{na}  tanimoto_ratio:{len(tanimoto)/len(lines)}\n",[va,un,na,len(tanimoto)/len(lines)]
    else:   
        return f"validity:{va} uniqueness:{un} novelty:{na} \n",[va,un,na]
def process_line3(smiles):
    qed,logp,sas = calculate_property_v1(smiles)
    return {"qed": qed, "logp": logp, "sas": sas}
   


def main(args, config,prop_input):
    device = torch.device(args.device)
    train_data=pd.read_csv(args.train_smiles)
    train_smiles=train_data["SMILES"].tolist()
    train_smiles=train_smiles[0:args.train_smiles_num]
    
    with open('/home/syy/pretrain_syy/scaffold_model/normalize.pkl', 'rb') as w:
        norm = pickle.load(w)
    mean=torch.cat((norm[0][[52,30]],torch.tensor([3.03]))) 
    std=torch.cat((norm[1][[52,30]],torch.tensor([0.82])))
    
    with open(f"{args.output_dir}/scaffold_prop_output_709.txt", "a") as f, open(f"{args.output_dir}/scaffold_result_709.txt", "a") as r:#_{args.prop_name}
        f.write(f"----------------{args.prop_name}-{args.n_generate}-------------------\n")
        f.write("checkpoint:"+args.checkpoint+"\n")
        f.write(f"train_smile:{len(train_smiles)}\n")
        

        metric_list_all=[]
        test_data=pd.read_csv(args.test_scaffold)
        test_scaffold=test_data["scaffold"].iloc[0:10000].tolist()
        test_smiles=test_data["SMILES"].iloc[0:10000].tolist()
      
        if args.prop_bool and not args.scaffold:
            if args.prop_input_bool:
                prop_input_test=prop_input
            else:
                prop_input_test=test_smiles
                print("scaffold",len(test_scaffold))
            scaffold_input_test=None
        elif args.scaffold and not args.prop_bool:
            if arg.scaffold_input is not None:
                scaffold_input_test=args.scaffold_input
            else:
                scaffold_input_test=test_scaffold
            prop_input_test=None
  
        elif args.prop_bool and args.scaffold:
            if args.prop_input_bool:
                if len(prop_input)==len(test_smiles) or len(prop_input)==len(args.scaffold_input):
                    prop_input_test=prop_input
                else:
                    prop_input_test=[prop_input[0] for _ in range(len(test_smiles))
            else:
                prop_input_test=test_smiles
            if arg.scaffold_input is not None:
                scaffold_input_test=args.scaffold_input
            else:
                scaffold_input_test=test_scaffold
        elif  not args.prop_bool  and not args.scaffold:
            prop_input_test=[1]
            scaffold_input_test=None
        
        # print("prop_input_test",prop_input_test)
        print("scaffold_input_test",scaffold_input_test)
        
        if args.scaffold:
            epoch_len=len(scaffold_input_test)
        else:
            epoch_len=len(prop_input_test)
        for i in range(6273,10000):
            print("epoch:",i)
            prop={"qed":None,"logp":None,"sas":None}
            if args.prop_bool:
                if args.prop_input_bool:
                    prop_input_value=prop_input_test[i]   
                else:
                    prop_input_value=calculate_property_v1(prop_input_test[i])
                    prop_input_value=prop_input_value[args.prop_index]
                if len(args.prop_index)==1:
                    prop[args.prop_name_all[args.prop_index[0]]]=prop_input_value
                elif len(args.prop_index)>1:
                    for j,name in enumerate(args.prop_index):
                        prop[args.prop_name_all[int(name)]]=prop_input_value[int(j)]
            else:
                prop_input_value=None
            
            print("prop",prop)
            prop_input=transform_string_generation(prop)    
                
            ##是否有scaffold条件
            if args.scaffold:
                scaffold_smiles_s=scaffold_input_test[i]
                scaffold_smiles='[CLS]<scaffold>'+scaffold_smiles_s
            else:
                scaffold_smiles_s=None
                scaffold_smiles=None
            
            f.write(f"prop_input:{prop_input}  scaffold_smiles:{scaffold_smiles}\n")
            
            # seed = random.randint(0, 1000)
            seed=266
            print('seed:', seed, args.stochastic)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            cudnn.benchmark = True
            f.write(f"epoch:{i}  seed:{seed}\n")

            tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
            tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)

            # === Model === #
            print("Creating model")
            model = BiSP_GP(config=config, tokenizer=tokenizer, no_train=True)
            print("token file",args.vocab_filename)
            if args.checkpoint:
                print('LOADING PRETRAINED MODEL..')
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                state_dict = checkpoint['state_dict']

                for key in list(state_dict.keys()):
                    if 'word_embeddings' in key and 'property_encoder' in key:
                        del state_dict[key]
                    if 'queue' in key:
                        del state_dict[key]

                msg = model.load_state_dict(state_dict, strict=False)
                print('load checkpoint from %s' % args.checkpoint)
                print(msg)
            model = model.to(device)
            samples = generate_with_property(model, properties=prop_input,scaffold=scaffold_smiles,n_sample=args.n_generate, stochastic=args.stochastic, k=args.k,prop_len=arg.property_len)
            print('Generated molecules are saved in \'generated_molecules_1.txt\'')
            metric,metric_list=metric_eval(prop_input_value, samples,args.prop_index,train_smiles,mean,std,scaffold_smiles_s)#,train_smile["SMILES"][0:20000000])#评估生成的分子的性质[0:48000000]
            metric_list_all.append(metric_list)
            
            f.write(f"metric_eval:{metric}\n")
            print("=" * 50)
            if metric_list[-1]>0.90 and metric_list[1]>0.7:
                r.write(f"{i},{scaffold_smiles_s},{test_smiles[i]},")
                r.write(','.join(map(str,metric_list)))
                r.write("\n")
        array=np.array(metric_list_all)
        column_means=np.mean(array,axis=0)
        print("metric_list_all",column_means)
        f.write(f"mean:{column_means}\n")
        

import itertools
if __name__ == '__main__':
    # device = torch.device('cuda:0')
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/model_save/checkpoint_step.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab.txt')
    parser.add_argument('--train_smiles', default="./data/pretraining/pretraining_data.csv")
    parser.add_argument('--train_smiles_num', default=20000000)
    parser.add_argument('--test_scaffold', default=None)
    parser.add_argument('--output_dir', default="/output_smiles/generation_signprop_scaffld/")
    parser.add_argument('--smiles_save', default="scaffold_")
    parser.add_argument('--device', default='cuda:0')  
    parser.add_argument('--n_generate', default=1000, type=int)
    parser.add_argument('--k', default=2, type=int)
    parser.add_argument('--property_len', default=25,type=int)
    parser.add_argument('--stochastic', default=True, type=bool)#stochastic=True
    parser.add_argument('--prop_bool', default=False, type=bool)#False
    parser.add_argument('--prop_name', default=None)
    parser.add_argument('--prop_index', default=None,type=int, nargs='+')
    parser.add_argument('--scaffold', default=False, type=bool)
    parser.add_argument('--scaffold_input', default=None, type=json.loads)
    parser.add_argument('--scaffold_prop_input', default=None,type=float, nargs='+')
    parser.add_argument('--prop_input_bool', default=False)
    parser.add_argument('--prop_name_all', default=["qed","logp","sas"])
    arg = parser.parse_args()
    
    
    print("arg",arg.prop_bool)
    print("arg",arg.scaffold_prop_input)
    if arg.prop_bool:
        if arg.prop_input_bool:
            if arg.scaffold_prop_input is not None:
                if len(arg.prop_index)>1:
                    prop_input=[arg.scaffold_prop_input]
                else:
                    prop_input=arg.scaffold_prop_input
            else:
                prop_qed=[0.5,0.7,0.9]#[0.3,0.6,0.9]
                prop_logp=[2.0,3.0,4.0]#[2,4,6]
                prop_sas=[2.0,2.5,3.0]#[2,3,4]
                prop_list={"qed":prop_qed,"logp":prop_logp,"sas":prop_sas}
                prop_input_name=arg.prop_name.split("_")
                prop_input=[]
                if len(prop_input_name)==1:
                    prop_input1=prop_list[prop_input_name[0]]
                    prop_input=[[i] for i in prop_input1]
                elif len(prop_input_name)==2:
                    list1=prop_list[prop_input_name[0]]
                    list2=prop_list[prop_input_name[1]]
                    prop_input=[list(pair) for pair in itertools.product(list1, list2)]
                elif len(prop_input_name)==3:
                    list1=prop_list[prop_input_name[0]]
                    list2=prop_list[prop_input_name[1]]
                    list3=prop_list[prop_input_name[2]]
                    prop_input=[list(pair) for pair in itertools.product(list1, list2,list3)]
                else:
                    prop_input=None
        else:
            prop_input=None
    else:
        prop_input=None
    print("prop_input",prop_input)
    configs = {
        'embed_dim': 256,
        'bert_config_text': './config_bert_smiles.json',
        'bert_config_property': './config_bert_property.json',
    }
    main(arg, configs,prop_input)



#

