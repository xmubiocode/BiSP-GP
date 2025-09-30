**Requirements**
---------
python==3.9
rdkit==2023.3.1
pandas==2.0.2

**Data**
-------

**Training**
--------
python pretraining.py 

**generation**
----------
Unconditional SMILES generation

python generation_proper2smiles.py 

Generation-based on single properties
python generation_proper2smiles.py   --output_dir ./output_smiles/generation/ --prop_bool True --prop_input_bool True --prop_name logp --prop_index 1  --n_generate 1000 --checkpoint ./model_save/checkpoint_step.ckpt
p
python 

Generation-based on multiple properties






