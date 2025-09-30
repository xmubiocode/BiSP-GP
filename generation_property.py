import argparse
import torch
import numpy as np
from pretrain_model import BiSP_GP
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer, WordpieceTokenizer
from dataset import Pretrain_Dataset
from torch.utils.data import DataLoader
import random
import pickle
from sklearn.metrics import r2_score
import re

############分子属性预测

def generate(model, prop_input, text_embeds, text_atts):
    prop_atts = torch.ones(prop_input.size(), dtype=torch.long).to(prop_input.device)
    prop_embeds=model.property_encoder.bert(prop_input,prop_atts,is_decoder=True, return_dict=True,).last_hidden_state
    token_output = model.text_encoder(encoder_embeds=prop_embeds,
                                        attention_mask=prop_atts,
                                        encoder_hidden_states=text_embeds,
                                        encoder_attention_mask=text_atts,
                                        return_dict=True,
                                        is_decoder=True,
                                        return_logits=True,
                                        mode='fusion',
                                        )[:, -1, :]  # batch*300
    
    token_output = torch.argmax(token_output, dim=-1)#返回最大值的索引
    return token_output.unsqueeze(1)  #


@torch.no_grad()
def pv_generate(model, data_loader):
    # test
    with open('./normalize.pkl', 'rb') as w:
        mean, std = pickle.load(w)
    device = model.device
    tokenizer = model.tokenizer
    model.eval()
    print("SMILES-to-PV generation...")
    # convert list of string to dataloader
    
    if isinstance(data_loader, list):#判断data_loader是否是list类型
        if data_loader[0][5] != "[CLS]":#判断第一个元素的第一个字符是否是"[CLS]"
            data_loader = ['[CLS]'+d for d in data_loader]
        gather = []
        text_input = tokenizer(data_loader, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)
        text_embeds = model.text_encoder.bert(text_input.input_ids[:, 1:], attention_mask=text_input.attention_mask[:, 1:],
                                              return_dict=True, mode='text').last_hidden_state
        prop_input = torch.tensor([tokenizer.cls_token_id]).expand(len(data_loader), -1, -1)#获得一个全是property_cls的tensor，大小和text一样表示起始字符
        prediction = []
        for _ in range(20):
            output = generate(model, prop_input, text_embeds, text_input.attention_mask[:, 1:])
            prediction.append(output)
            prop_input = torch.cat([prop_input, output], dim=1)

        prediction = torch.stack(prediction, dim=-1)
        for i in range(len(data_loader)):
            gather.append(prediction[i].cpu()*std + mean)
        return gather

    reference, candidate = [], []
    for (prop, text) in data_loader:
        
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=100, return_tensors="pt").to(device)
        text_embeds = model.text_encoder.bert(text_input.input_ids[:, 1:], attention_mask=text_input.attention_mask[:, 1:],
                                              return_dict=True, mode='text').last_hidden_state
        prop_input = torch.tensor([tokenizer.cls_token_id]).expand(len(text), 1).to(device)#获得一个全是property_cls的tensor，大小和text一样
        prediction = []
        for i in range(20):
            #print ("i",i)
            output = generate(model, prop_input, text_embeds, text_input.attention_mask[:, 1:])
            prediction.append(output)
            prop_input = torch.cat([prop_input, output], dim=1)
            
                
        prediction = torch.stack(prediction, dim=-1)#将prediction的值按照dim=-1的维度进行堆叠
        for i in range(prop.size(0)):
            reference.append(prop[i].cpu())
            candidate.append(prediction[i].cpu())
        
    candidate_p=[]      
    for sentence in candidate:
        p=[]
        for i in sentence:
            cdd = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(i))#将token转化为字符串
            p.append(cdd)
        candidate_p.append(p)
    condidate_num=[]
    for identifier in candidate_p:
        try:
            string=identifier[0].split("[sep]")[0]
            reference_num=parse_multiple_formatted_strings(string)
            if len(reference_num)==3:
                condidate_num.append(list(reference_num.values()))
            else:
                print("error",reference_num)
        except ValueError as e:
            print(e)
    
    print('SMILES-to-PV generation done')
    return reference, torch.tensor(condidate_num)#reference是真实的物性值，candidate是生成的物性值

def parse_multiple_formatted_strings(string):
    # 定义通用的正则表达式来匹配 <标识符> 和其后的正则化数字
    pattern = r"""
        <(?P<identifier>\w+)>\s      # 捕获 <标识符>，如 <qed>, <logp> 等
        (?P<sign>_[+-]_\s)           # 捕获正负号
        ((?:_\d+_\d+_\s)+)           # 捕获整数部分
        _\._\s                       # 匹配小数点
        ((?:_\d+_-\d+_\s)+)          # 捕获小数部分
    """
    
    matches = re.finditer(pattern, string, re.VERBOSE)

    results = {}

    # 提取数字并转化为浮点数，添加检查逻辑
    def extract_number(sign, int_part, dec_part):
        # 提取整数部分
        int_numbers = re.findall(r'_(\d+)_(\d+)_', int_part)
        integer = ''.join(digit for digit, _ in int_numbers)

        # 检查整数部分索引递增性
        int_indices = [int(index) for _, index in int_numbers]
        if int_indices != sorted(int_indices):
            raise ValueError("整数部分的索引没有递增")

        # 提取小数部分
        dec_numbers = re.findall(r'_(\d+)_-(\d+)_', dec_part)
        decimal = ''.join(digit for digit, _ in dec_numbers)

        # 检查小数部分索引递减性
        dec_indices = [-int(index) for _, index in dec_numbers]
        if dec_indices != sorted(dec_indices, reverse=True):
            raise ValueError("小数部分的索引没有递减")

        # 组合整数和小数，形成浮点数
        number = float(f"{integer}.{decimal}")
        
        # 根据正负号调整
        if sign == '_-_ ':
            number = -number
        return number
    
    # 遍历所有的匹配项
    for match in matches:
        identifier = match.group('identifier')
        sign = match.group('sign')
        int_part = match.group(3)
        dec_part = match.group(4)

        # 解析并获取数值
        number = extract_number(sign, int_part, dec_part)
        results[identifier] = number

    return results

@torch.no_grad()
def metric_eval(ref, cand):
    
    mse = []
    n_mse = []
    rs, cs = [], []
    for i in range(len(ref)):
        # r = (ref[i] * std) + mean#数据标准化
        # c = (cand[i] * std) + mean
        r=ref[i]
        c=cand[i]
        rs.append(r)
        cs.append(c)
        mse.append((r - c) ** 2)
        n_mse.append((ref[i] - cand[i]) ** 2)
    mse = torch.stack(mse, dim=0)
    rmse = torch.sqrt(torch.mean(mse, dim=0)).squeeze()
    n_mse = torch.stack(n_mse, dim=0)
    n_rmse = torch.sqrt(torch.mean(n_mse, dim=0))
    print("RMSE:", n_rmse)
    print('mean of 3 properties\' normalized RMSE:', n_rmse.mean().item())

    rs = torch.stack(rs)
    cs = torch.stack(cs).squeeze()
    r2 = []
    for i in range(rs.size(1)):
        r2.append(r2_score(rs[:, i].cpu().numpy(), cs[:, i].cpu().numpy()))#计算r^2系数
    r2 = np.array(r2)
    print("R^2:", r2)
    # print("ref:",rs)
    # print("cand:",cs)
    print('mean r^2 coefficient of determination:', r2.mean().item())
    return n_rmse, r2

def main(args, config):
    device = torch.device(args.device)
    #seed = random.randint(0, 1000)#
    seed=986
    #print('seed:', seed)
    torch.manual_seed(seed)#
    np.random.seed(seed)#
    random.seed(seed)
    cudnn.benchmark = True

    # === Dataset === #
    print("Creating dataset")
    dataset_test = Pretrain_Dataset(args.input_file)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], pin_memory=True, drop_last=False)

    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)
    

    # === Model === #
    print("Creating model")
    model = BiSP_GP(config=config, tokenizer=tokenizer, no_train=True)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'queue' in key:
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    model = model.to(device)

    print("=" * 50)
    r_test, c_test = pv_generate(model, test_loader)
    
    with open(args.output_file, 'w') as f:
        f.write("qed,logp,sas\n")
        for i in range(len(c_test)):
            f.write(f"{c_test[i][0]:.4f},{c_test[i][1]:.4f},{c_test[i][2]:.4f}\n")  
    n_rmse, r2 = metric_eval(r_test, c_test)
    with open("abaly/unseen_prediction.txt", 'a') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"n_rmse: {n_rmse}, r2: {r2}\n")
        f.write("=" * 50 + "\n")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./model_save/checkpoint_step=260000.ckpt')
    parser.add_argument('--vocab_filename', default='./vocab.txt')
    parser.add_argument('--input_file', default='./data/SMILES-property/zinc15_1k_unseen_pp.csv')
    parser.add_argument('--output_file', default='zinc15_1k_unseen_prediction.csv')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    config = {
        'embed_dim': 256,
        'batch_size_test': 64,
        'bert_config_text': './config_bert_smiles.json',
        'bert_config_property': './config_bert_property.json',
    }
    main(args, config)

