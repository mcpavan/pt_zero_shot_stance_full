from itertools import product
import os

first_v_config = 1
folder = "distant"
dataset_list = ["ustancebr", "semeval", "mtwitter", "brmoral"]

modelname2example = {
    "BiCond": f"./example_{folder}/BiCond_example.txt",
    "BiLSTM": f"./example_{folder}/BiLstm_example.txt",
    "BiLSTMJoint": f"./example_{folder}/BiLstmJoint_example.txt",
    "CrossNet": f"./example_{folder}/CrossNet_example.txt",
}

default_params_ustancebr_distant = {
    "text_col":"Text",
    "topic_col":"Target",
    "label_col":"Polarity",
    "sample_weights": 1,
    "n_output_classes": 2,
}

default_params_semeval_distant = {
    "text_col":"Tweet",
    "topic_col":"Target",
    "label_col":"Stance",
    "sample_weights": 1,
    "n_output_classes": 2,
    "alpha_load_classes": 1,
}

default_params_brmoral_mtwitter_distant = {
    "text_col":"Text",
    "topic_col":"Target",
    "label_col":"Polarity",
    "sample_weights": 1,
    "n_output_classes": 2,
    "alpha_load_classes": 1,
}

values_dict_default = {
    "BiCondLstm": {
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
        "epochs": [
            50
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
        "epochs": [
            50
        ],
        # "batch_size": [
        #     "192",
        # ]
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
        "epochs": [
            50
        ],
        # "batch_size": [
        #     "152",
        # ]
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
        "epochs": [
            50
        ],
        # "batch_size": [
        #     64
        # ],
    },
}

def load_config_file(config_file_path):
    with open(config_file_path, 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]
    
    return config

values_dict = values_dict_default
for dataset in dataset_list:
    out_path = "./{dataset}/{folder}/{model_name_out_file}_v{k}.txt"
    ckp_path = "../../checkpoints/{dataset}/{folder}/{name}/V{k}/"
    
    if dataset == "ustancebr":
        default_params = default_params_ustancebr_distant
    elif dataset == "semeval":
        default_params = default_params_semeval_distant
    elif dataset in ["brmoral", "mtwitter"]:
        default_params = default_params_brmoral_mtwitter_distant

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
        