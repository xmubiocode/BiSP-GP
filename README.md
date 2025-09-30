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

`python generation_proper2smiles.py`

Generation-based on single/multiple properties

`python generation_proper2smiles.py   --output_dir ./output_smiles/generation/ --prop_bool True --prop_input_bool True --prop_name logp --prop_index 1  --n_generate 1000 --checkpoint ./model_save/checkpoint_step.ckpt`

Generation-based on scaffold

`python generation_proper-scaffold2smiles.py --output_dir ./output_smiles/generation/ --prop_bool True --prop_name logp --prop_index 1 --prop_input_bool True --scaffold True  --scaffold_prop_input 2 3 4  --scaffold_input '["O=C(N1CCCCC1)n1cnc2ccccc21","O=C(N1CCCCC1)n1cnc2ccccc21","O=C(N1CCCCC1)n1cnc2ccccc21"]' --n_generate 1000 --checkpoint ./model_save/checkpoint_step.ckpt`

**Property prediction**
------
`python generation_property.py`

**downstream tasks**
------
`python downstream_classification.py`
`python downstream_classification_multilabel.py`
`python downstream_regression.py`









