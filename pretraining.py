from torch.utils.data import DataLoader

from pretrain_model import PSBPGM
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from dataset_pretraining import Pretrain_Dataset
from transformers import BertTokenizer, WordpieceTokenizer
print("torch.cuda.is_available()",torch.cuda.is_available())
def main(args,config):
    ngpu=4
    print("Creating dataset")
    dataset=Pretrain_Dataset(args.data_path)
    print('#data:', len(dataset), torch.cuda.is_available())
    print(dataset.data[0])
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=0, shuffle=False, pin_memory=True, drop_last=True)
    tokenizer = BertTokenizer(vocab_file=args.vocab_filename, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=250)
    
    #model
    model= PSBPGM(config=config,tokenizer=tokenizer,loader_len=len(data_loader) // torch.cuda.device_count())
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        _ = model.load_state_dict(checkpoint['state_dict'], strict=False)
    #training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, filename='checkpoint_{step}',
                                                       every_n_train_steps=10000,
                                                        )
    trainer = pl.Trainer(accelerator='gpu', devices=ngpu, precision='16-mixed', max_epochs=config['schedular']['epochs'],
                         callbacks=[checkpoint_callback], strategy=DDPStrategy(find_unused_parameters=True), limit_val_batches=0.)
    print("model",model.device)
    trainer.fit(model, data_loader, None, ckpt_path=args.checkpoint if args.checkpoint else None)

    
if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default="./model_save")
    parser.add_argument('--vocab_filename', default="./vocab.txt")
    parser.add_argument('--data_path', default="./data/pretraining/pretraining_data.csv")
    parser.add_argument('--gpus', default=4, type=int)
    args = parser.parse_args()
    config = {
        'embed_dim': 256,
        'batch_size': 96,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'queue_size': 12288,
        'momentum': 0.995,
        'alpha': 0.4,
        'bert_config_smiles': './config_bert_smiles.json',
        'bert_config_property': './config_bert_property.json',
        'schedular': {'sched': 'cosine', 'lr': 5e-5, 'epochs': 5, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 5e-5, 'weight_decay': 0.02}
    }
    
    main(args,config)
    main(args,config)
   