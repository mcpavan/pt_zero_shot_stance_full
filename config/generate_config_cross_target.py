from itertools import product
import os

first_v_config = 1
folder = "cross_target"
dataset_list = ["ustancebr", "semeval"]

modelname2example = {
    "BertAAD": f"./example_{folder}/BertAAD_example.txt",
    "BiCond": f"./example_{folder}/BiCond_example.txt",
    "BiLSTM": f"./example_{folder}/BiLstm_example.txt",
    "BiLSTMJoint": f"./example_{folder}/BiLstmJoint_example.txt",
    "CrossNet": f"./example_{folder}/CrossNet_example.txt",
    "JointCL": f"./example_{folder}/JointCL_example.txt",
    "TOAD": f"./example_{folder}/TOAD_example.txt",
}

default_params_ustancebr = {
    "text_col":"Text",
    "topic_col":"Target",
    "label_col":"Polarity",
    "sample_weights": 1,
    "n_output_classes": 2,
}

default_params_semeval = {
    "text_col":"Tweet",
    "topic_col":"Target",
    "label_col":"Stance",
    "sample_weights": 0,
    "n_output_classes": 3,
    "alpha_load_classes": 1,
}

values_dict_ustancebr = {
    "BertAAD": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "learning_rate": [
            1e-7,
        ],
        "discriminator_learning_rate": [
            1e-7,
        ],
        "discriminator_dim": [
            1024,
            3072,
        ]
    },
    "BiCond": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "learning_rate": [
            1e-7,
        ],
        # "batch_size": [
        #     96
        # ],
    },
    "BiLSTM": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "32",
            "64",
        ],
        "attention_heads": [
            "1",
            "16",
        ],
        "batch_size": [
            "192",
        ]
    },
    "BiLSTMJoint": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "32",
            "64",
        ],
        "attention_heads": [
            "1",
            "16",
        ],
        "batch_size": [
            "152",
        ]
    },
    "CrossNet": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "100",
            "200",
        ],
        "learning_rate": [
            1e-7,
        ],
        # "batch_size": [
        #     64
        # ],
    },
    "JointCL": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            "-1", # only 1 layer allowed
            # "-4,-3,-2,-1",
        ],
        "gnn_dims": [
            "64,64",
            # "128,128",
            "192,192",
        ],
        "att_heads": [
            "12,12",
            # "6,6",
            "4,4",
        ],
        "learning_rate": [
            1e-7,
        ],
        # "batch_size": [
        #     32
        # ],
    },
    "TOAD": {
        "bert_pretrained_model": [
            # "neuralmind/bert-base-portuguese-cased",
            "pablocosta/bertabaporu-base-uncased",
        ],
        "bert_layers": [
            "-1",
            # "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            # "2",
        ],
        "lstm_hidden_dim": [
            128,
            # 226
        ],
        "stance_classifier_dimension": [
            201,
            402,
            # 804,
        ],
        "topic_classifier_dimension": [
            140,
            280,
            # 420,
        ],
        "learning_rate": [
            1e-5,
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
}

values_dict_semeval = {
    "BertAAD": {
        "bert_pretrained_model": [
            "bert-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "learning_rate": [
            1e-7,
        ],
        "discriminator_learning_rate": [
            1e-7,
        ],
        "discriminator_dim": [
            1024,
            3072,
        ]

    },
    "BiCond": {
        "bert_pretrained_model": [
            "bert-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "learning_rate": [
            1e-7,
        ],
        "epochs":[
            20,
        ],
    },
    "BiLSTM": {
        "bert_pretrained_model": [
            "bert-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "32",
            "64",
        ],
        "attention_heads": [
            "1",
            "16",
        ],
        # "learning_rate": [
        #     1e-10,
        # ],
        "epochs":[
            20,
        ],
    },
    "BiLSTMJoint": {
        "bert_pretrained_model": [
            "bert-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "32",
            "64",
        ],
        "attention_heads": [
            "1",
            "16",
        ],
        # "learning_rate": [
        #     1e-10,
        # ],
        "epochs":[
            20,
        ],
    },
    "CrossNet": {
        "bert_pretrained_model": [
            "bert-base-uncased",
        ],
        "bert_layers": [
            # "-1",
            "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            "2",
        ],
        "lstm_hidden_dim": [
            "16",
            "128",
        ],
        "attention_density": [
            "100",
            "200",
        ],
        "learning_rate": [
            1e-7,
        ],
        "epochs":[
            20,
        ],
        # "batch_size": [
        #     64
        # ],
    },
    "JointCL": {
        "bert_pretrained_model": [
            "bert-base-uncased",
        ],
        "bert_layers": [
            "-1", # only 1 layer allowed
            # "-4,-3,-2,-1",
        ],
        "gnn_dims": [
            "64,64",
            # "128,128",
            "192,192",
        ],
        "att_heads": [
            "12,12",
            # "6,6",
            "4,4",
        ],
        "learning_rate": [
            1e-7,
        ],
        "epochs":[
            20,
        ],
        # "batch_size": [
        #     32
        # ],
    },
    "TOAD": {
        "bert_pretrained_model": [
            "bert-base-uncased",
        ],
        "bert_layers": [
            "-1",
            # "-4,-3,-2,-1",
        ],
        "lstm_layers": [
            "1",
            # "2",
        ],
        "lstm_hidden_dim": [
            128,
            # 226
        ],
        "stance_classifier_dimension": [
            201,
            402,
            # 804,
        ],
        "topic_classifier_dimension": [
            140,
            280,
            # 420,
        ],
        "learning_rate": [
            1e-5,
        ],
        "epochs":[
            20,
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
}

def load_config_file(config_file_path):
    with open(config_file_path, 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    
    return config

for dataset in dataset_list:
    out_path = "./{dataset}/{folder}/{model_name_out_file}_v{k}.txt"
    ckp_path = "../../checkpoints/{dataset}/{folder}/{name}/V{k}/"
    
    if dataset == "semeval":
        values_dict = values_dict_semeval
        default_params = default_params_semeval
    elif dataset == "ustancebr":
        values_dict = values_dict_ustancebr
        default_params = default_params_ustancebr

    for model_name_out_file, example_file in modelname2example.items():
        os.makedirs(
            "/".join(out_path.split("/")[:-1]) \
            .replace("{dataset}", dataset) \
            .replace("{folder}", folder),
            exist_ok=True
        )

        base_config_dict = load_config_file(example_file)
        base_config_dict["name"] = f"{model_name_out_file}_{dataset}_"
        k = first_v_config
        for combination in product(*list(values_dict[model_name_out_file].values())):

            if model_name_out_file == "JointCL":
                comb_dict = {}
                for key, value in zip(values_dict[model_name_out_file].keys(), combination):
                    comb_dict[key] = value

                bert_out_dim = 768 # it comes from the pooler output not from the layer hidden states
                gnn_dim = int(comb_dict["gnn_dims"].split(",")[0])
                att_heads = int(comb_dict["att_heads"].split(",")[0])

                if gnn_dim * att_heads != bert_out_dim:
                    continue
            
            new_config_dict = base_config_dict
            # setting default params
            for key, value in default_params.items():
                new_config_dict[key] = value

            # setting combination specific params
            for key, value in zip(values_dict[model_name_out_file].keys(), combination):
                new_config_dict[key] = value
            
            new_config_dict["ckp_path"] = ckp_path \
                .replace("{dataset}", dataset) \
                .replace("{folder}", folder) \
                .replace(
                    "{name}",
                    model_name_out_file.lower().replace("bert", ""),
                ) \
                .replace("{k}", str(k))             
            current_out_path = out_path \
                .replace("{dataset}", dataset) \
                .replace("{folder}", folder) \
                .replace("{model_name_out_file}", model_name_out_file) \
                .replace("{k}", str(k))
            
            with open(current_out_path, "w") as f_:
                new_config_str = "\n".join(f"{k}:{v}" for k, v in new_config_dict.items())
                print(new_config_str, end="", file=f_, flush=True)
            
            k += 1
        