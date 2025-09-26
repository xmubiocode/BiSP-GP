import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
import pytorch_lightning as pl
from xbert import BertConfig, BertForMaskedLM

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        


class PSBPGM(pl.LightningModule):
    def __init__(self, tokenizer=None, no_train=False, loader_len=100, config=None):
        super(PSBPGM, self).__init__()
        self.automatic_optimization = False
        self.tokenizer = tokenizer
        self.config=config
        self.use_cuda=torch.cuda.is_available()
        self.hidden_en_size=256
        embed_dim = config['embed_dim']
        
        bert_config = BertConfig.from_json_file(config['bert_config_text'])
        self.text_encoder=BertForMaskedLM(config=bert_config)
        text_width = self.text_encoder.config.hidden_size
        property_width=text_width
 
        bert_config2=BertConfig.from_json_file(config['bert_config_property'])
        self.property_encoder=BertForMaskedLM(config=bert_config2)
        

        self.property_proj = nn.Linear(property_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.star=torch.tensor([300, 301, 302])
        self.end=torch.tensor([301, 302, 3])
        #构建教师动量模型
        self.property_encoder_m = BertForMaskedLM(config=bert_config2)
        self.property_proj_m = nn.Linear(property_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM(config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        for p in self.property_encoder_m.parameters():  p.requires_grad = False
        for p in self.property_proj_m.parameters():     p.requires_grad = False
        for p in self.text_encoder_m.parameters():      p.requires_grad = False
        for p in self.text_proj_m.parameters():         p.requires_grad = False

        self.model_pairs = [[self.property_encoder, self.property_encoder_m],
                            [self.property_proj, self.property_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()
        
        if not no_train:
            self.temp = nn.Parameter(torch.ones([]) * config['temp'])

            self.momentum = config['momentum']
            self.queue_size = config['queue_size']
            self.register_buffer("prop_queue", torch.randn(self.hidden_en_size, self.queue_size))
            self.register_buffer("text_queue", torch.randn(self.hidden_en_size, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.prop_queue = nn.functional.normalize(self.prop_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
         

    
    def forward(self,text_input_ids,text_attention_mask,prop_input_ids,prop_attention_mask,scaffold_smiles,scaffold_attention_mask,alpha=0.4):
        with torch.no_grad():
            self.temp.clamp_(0.01, 0.5)
        device=self.device
        text_embeds = self.text_encoder.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
        text_feat= F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        scaffold_embeds=self.text_encoder.bert(scaffold_smiles, attention_mask=scaffold_attention_mask, return_dict=True,mode='text').last_hidden_state

        prop_mask = torch.zeros_like(prop_input_ids)
        mpm_mask = torch.bernoulli(torch.ones(prop_input_ids.size(0), self.star.size(0)) * 0.5)
        unk_tokens_star = (mpm_mask * self.star.unsqueeze(0)).long().to(device)
        unk_tokens_end = (mpm_mask * self.end.unsqueeze(0)).long().to(device)
        start_indices = (prop_input_ids.unsqueeze(1) == unk_tokens_star.unsqueeze(2)).nonzero(as_tuple=True)
        end_indices = (prop_input_ids.unsqueeze(1) == unk_tokens_end.unsqueeze(2)).nonzero(as_tuple=True)

        for i in range(start_indices[0].size(0)):
            batch_id = start_indices[0][i]
            start_pos = start_indices[2][i]
            end_pos = end_indices[2][i]
            if start_pos < end_pos: 
                prop_mask[batch_id, start_pos:end_pos ] = 1
        unk_prop_input_ids=prop_input_ids*(1-prop_mask)+prop_mask*torch.tensor(self.tokenizer.convert_tokens_to_ids('[UNK]')).to(device)

        ####################################################################################################################
        prop_embeds = self.property_encoder.bert(unk_prop_input_ids, attention_mask=prop_attention_mask, return_dict=True,mode='text').last_hidden_state
        prop_feat= F.normalize(self.property_proj(prop_embeds[:, 0, :]), dim=-1)
        with torch.no_grad():
            self._momentum_update()
            prop_embeds_m = self.property_encoder_m.bert(prop_input_ids, attention_mask=prop_attention_mask, return_dict=True,mode='text').last_hidden_state
            prop_feat_m= F.normalize(self.property_proj(prop_embeds_m[:, 0, :]), dim=-1)
            prop_feat_all = torch.cat([prop_feat_m.t(), self.prop_queue.clone().detach()], dim=1)
            
            text_embeds_m = self.text_encoder_m.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
            text_feat_m= F.normalize(self.text_proj(text_embeds_m[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
            
            
            sim_i2t_m = prop_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ prop_feat_all / self.temp
            sim_i2i_m = prop_feat_m @ prop_feat_all / self.temp
            sim_t2t_m = text_feat_m @ text_feat_all / self.temp
            

            sim_targets = torch.zeros(sim_i2t_m.size()).to(device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_i2i_targets = alpha * F.softmax(sim_i2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2t_targets = alpha * F.softmax(sim_t2t_m, dim=1) + (1 - alpha) * sim_targets
          
            
        sim_p2s=prop_feat @ text_feat_all / self.temp
        sim_s2s=text_feat @ text_feat_all / self.temp
        sim_s2p=text_feat @ prop_feat_all / self.temp
        sim_p2p=prop_feat @ prop_feat_all / self.temp

        
        loss_p2s=-torch.sum(F.log_softmax(sim_p2s, dim=1)*sim_i2t_targets, dim=1).mean()
        loss_s2s=-torch.sum(F.log_softmax(sim_s2s, dim=1)*sim_t2t_targets, dim=1).mean()
        loss_s2p=-torch.sum(F.log_softmax(sim_s2p, dim=1)*sim_t2i_targets, dim=1).mean()
        loss_p2p=-torch.sum(F.log_softmax(sim_p2p, dim=1)*sim_i2i_targets, dim=1).mean()
        
        loss_cl=(loss_p2s+loss_s2s+loss_s2p+loss_p2p)/2
        self._dequeue_and_enqueue(prop_feat, text_feat) 
        
        #####################################################
        input_ids = text_input_ids.clone()
        labels = input_ids.clone()[:, 1:]
        attention_mask=torch.ones(text_feat.size(0),1,1).to(prop_feat.device)
        hidden_states=(text_embeds[:, 0, :]).unsqueeze(1)
        hidden_states=hidden_states.to(prop_feat.device)
        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=hidden_states,
                                       encoder_attention_mask=attention_mask,
                                       return_dict=True,
                                       is_decoder=True,
                                       return_logits=True,
                                       )[:, :-1, :]
        
        loss_fct = nn.CrossEntropyLoss()
        loss_sre = loss_fct(mlm_output.permute((0, 2, 1)), labels)

        ##################################################
        input_ids2=prop_input_ids.clone()
        labels2=prop_input_ids.clone()[:, 1:]
        attention_mask1=torch.ones(prop_feat.size(0),1,1).to(prop_feat.device)
        hidden_states1=(prop_embeds[:, 0, :]).unsqueeze(1)
        hidden_states1=hidden_states1.to(prop_feat.device)
        prop_embeds_causal=self.property_encoder.bert(input_ids2, attention_mask=prop_attention_mask,is_decoder=True, return_dict=True).last_hidden_state
        prop_output=self.text_encoder(encoder_embeds=prop_embeds_causal,
                                    attention_mask=prop_attention_mask,
                                    encoder_hidden_states=hidden_states1,
                                    encoder_attention_mask=attention_mask1,
                                    return_dict=True,
                                    is_decoder=True,
                                    return_logits=True,
                                    mode='fusion',
                                    )[:, :-1, :]
        
        prop_predict=prop_output[(1 - prop_mask[:,1:]).to(bool)]

        loss_pre=loss_fct(prop_predict.permute((0,1)),labels2[(1 - prop_mask[:,1:]).to(bool)])
     
        #####################################################################
        input_ids = text_input_ids.clone()
        labels = input_ids.clone()[:, 1:]
        
        scaffold_do = torch.bernoulli(torch.tensor(0.5)).int().item()

        if scaffold_do:
            condation_emdedss=torch.cat([prop_embeds[:,:-1,:], scaffold_embeds[:,1::,:]], dim=1)
            condation_attention_mask=torch.cat([prop_attention_mask[:,:-1], scaffold_attention_mask[:,1::]], dim=1)
            with torch.no_grad():
            # self._momentum_update()
                prop_embeds_m = self.property_encoder_m.bert(unk_prop_input_ids, attention_mask=prop_attention_mask, return_dict=True,mode='text').last_hidden_state
                scaffold_embeds_m=self.text_encoder_m.bert(scaffold_smiles, attention_mask=scaffold_attention_mask, return_dict=True,mode='text').last_hidden_state
                condation_emdedss_m=torch.cat([prop_embeds_m[:,:-1,:], scaffold_embeds_m[:,1::,:]], dim=1)
                condation_attention_mask_m=torch.cat([prop_attention_mask[:,:-1], scaffold_attention_mask[:,1::]], dim=1)
                logits_m = self.text_encoder_m(input_ids,
                                            attention_mask=text_attention_mask,
                                            encoder_hidden_states=condation_emdedss_m,
                                            encoder_attention_mask=condation_attention_mask_m,
                                            return_dict=True,
                                            is_decoder=True,
                                            return_logits=True,
                                            )[:, :-1, :]

        else:
            condation_emdedss=prop_embeds
            condation_attention_mask=prop_attention_mask
            with torch.no_grad():
                # self._momentum_update()
                prop_embeds_m = self.property_encoder_m.bert(unk_prop_input_ids, attention_mask=prop_attention_mask, return_dict=True,mode='text').last_hidden_state
                condation_emdedss_m=prop_embeds_m
                condation_attention_mask_m=prop_attention_mask
                logits_m = self.text_encoder_m(input_ids,
                                            attention_mask=text_attention_mask,
                                            encoder_hidden_states=condation_emdedss_m,
                                            encoder_attention_mask=condation_attention_mask_m,
                                            return_dict=True,
                                            is_decoder=True,
                                            return_logits=True,
                                            )[:, :-1, :]
            
        mlm_output = self.text_encoder(input_ids,
                                    attention_mask=text_attention_mask,
                                    encoder_hidden_states=condation_emdedss,
                                    encoder_attention_mask=condation_attention_mask,
                                    return_dict=True,
                                    is_decoder=True,
                                    return_logits=True,
                                    )[:, :-1, :]#[batch_size,seq_len,vocab_size]
        
        loss_fct = nn.CrossEntropyLoss()
        loss_mlm = loss_fct(mlm_output.permute((0, 2, 1)), labels)
        
        loss_distill_text = -torch.sum(F.log_softmax(mlm_output, dim=-1) * F.softmax(logits_m, dim=-1), dim=-1)
        loss_distill_text = loss_distill_text[labels != 0].mean()
        loss_mlm = (1 - alpha) * loss_mlm + alpha * loss_distill_text
        
        ##########################################################
        input_ids2=unk_prop_input_ids.clone()
        labels2=prop_input_ids.clone()[:, 1:]
        prop_embeds_causal=self.property_encoder.bert(input_ids2, attention_mask=prop_attention_mask,is_decoder=True, return_dict=True).last_hidden_state
        prop_output=self.text_encoder(encoder_embeds=prop_embeds_causal,
                                    attention_mask=prop_attention_mask,
                                    encoder_hidden_states=text_embeds,
                                    encoder_attention_mask=text_attention_mask,
                                    return_dict=True,
                                    is_decoder=True,
                                    return_logits=True,
                                    mode='fusion',
                                    )[:,:-1,:]
        prop_predict=prop_output[(1 - prop_mask[:,1:]).to(bool)]
        loss_mpm=loss_fct(prop_predict.permute((0,1)),labels2[(1 - prop_mask[:,1:]).to(bool)])
    
    
    
        return loss_cl,loss_mlm,loss_mpm,loss_sre,loss_pre
    

    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_feat, text_feat):
        img_feats = concat_all_gather(img_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = img_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.prop_queue[:, ptr:ptr + batch_size] = img_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def configure_optimizers(self):
        arg_opt = self.config['optimizer']
        optimizer = torch.optim.AdamW(self.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])
        return [optimizer]
        
    def training_step(self,train_bath,batch_idx):
        train_loss=0
        # print(f"cuda: {self.device}") 
        device=self.device
        optimizer=self.optimizers()
        optimizer.zero_grad()
        text,prop_char,scaffolds= train_bath
         
        text_input=self.tokenizer(text,padding='longest',return_tensors='pt',truncation=True,max_length=120).to(device)
        prop_input=self.tokenizer(prop_char,padding='longest',return_tensors='pt',truncation=True,max_length=40).to(device)
        scaffolds_input=self.tokenizer(scaffolds,padding='longest',return_tensors='pt',truncation=True,max_length=120).to(device)
        del text, prop_char,scaffolds
        loss_cl,loss_mlm,loss_mpm,loss_sre,loss_pre=self(text_input.input_ids[:, 1:], text_input.attention_mask[:, 1:],prop_input.input_ids[:, 1:], prop_input.attention_mask[:, 1:],scaffolds_input.input_ids[:, 1:], scaffolds_input.attention_mask[:, 1:])
        loss=loss_cl+loss_mlm+loss_mpm+loss_sre+loss_pre
        
        if loss!=torch.tensor(0.):
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            optimizer.step()
            
            if batch_idx%10000==0:
                print('###########loss is',loss)
                print('###########loss_cl is',loss_cl)
                print('###########loss_mlm is',loss_mlm)
                print('###########loss_mpm is',loss_mpm)
                print('###########loss_sre is',loss_sre)
                print('###########loss_pre is',loss_pre)
        else:
            print('###########loss is 0')
        if self.global_rank == 0:
            self.log('train_loss', loss)
            self.log("lr",optimizer.param_groups[0]['lr'])
        train_loss+=loss.item()
        if batch_idx%10000==0:
            print('train_loss:',train_loss/10000)
            train_loss=0
        return loss
    
    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient   
    
    
              
    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)        
                
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

