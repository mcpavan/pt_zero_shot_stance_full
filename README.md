# pt_zero_shot_stance_full
A benchmark for Portuguese zero-shot stance detection

# Requirements:

- python==3.7.10
- numpy==1.19.2
- matplotlib==3.5.2
- pandas==1.2.4
- seaborn==0.11.2
- tqdm==4.59.0
- scikit-learn==0.24.2
- pytorch==1.13.1 (Cuda 11.7)
- transformers==4.28.0

- faiss-gpu==1.7.1 (conda install -c conda-forge faiss-gpu==1.7.1) - only for JointCL

- transformers==4.28.0 - only for llm
- sentencepiece==0.1.98 - only for llm
- accelerate==0.18.0 - only for llm
- bitsandbytes==0.38.1 - only for llm
- protobuf==3.20.0 - only for llm

- llama-cpp-python==0.1.48 - only for llama_cpp

- sentence-transformers==2.2.2

# Creating the config files:

Go to the folder [`config`](https://github.com/mcpavan/pt_zero_shot_stance_full/tree/main/config) and execute the .py file for the respective approach (`in-domain`, `cross-target`, `llm` or `distant`). This will generate a set of subfolders and files that correspond to the grid seach executed in the PhD Thesis.

# Training a model:

Go to the folder [`src/py_util`](https://github.com/mcpavan/pt_zero_shot_stance_full/tree/main/src/py_util) and run a command with the following template:

General syntax:
```
python train_model.py -m train -c [config_file] -t [train_data] -v [valid_data] -p [test_data] -n [model_name] -e [early_stopping] -s [save_checkpoints]
```

Example:
```
python train_model.py -m train -c ../../config/ustancebr/indomain/BiLSTM_example.txt -t ../../data/UStanceBR/v2/final_bo_train.csv -v ../../data/UStanceBR/v2/final_bo_valid.csv -n bo -e 5 -s 1
```

# References:

This code is inspired by https://github.com/emilyallaway/zero-shot-stance and https://github.com/MalavikaSrikanth16/adversarial-learning-for-stance, from which the implementations in the present repository of the models BiCond, CrossNet, and TOAD were reused. The implementation of the model JointCL is reused from https://github.com/HITSZ-HLT/JointCL/tree/main.

# Contact:

[Matheus Camasmie Pavan](linkedin.com/in/matheus-camasmie-pavan) ([matheus.pavan@usp.br](matheus.pavan@usp.br))