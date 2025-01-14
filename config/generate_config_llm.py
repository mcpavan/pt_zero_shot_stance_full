from itertools import product
import os

first_v_config = 1
folder = "llm"
dataset_list = ["ustancebr", "semeval"]

modelname2example = {
    "Llama_4bit": f"./example_{folder}/Llama_4bit_example.txt",
    "Llama_8bit": f"./example_{folder}/Llama_8bit_example.txt",
    "Bloom": f"./example_{folder}/HF_API_example.txt",
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
    "Llama_4bit": {
        "pretrained_model_name": [
            "../../data/LLMs/ggml-alpaca-7b-q4.bin",
        ],
        "prompt": [
            {
                "prompt_template_file": "./example_llm/ustancebr_prompts/stance_prompt_alpaca_score10_0.md",
                "output_max_score": 10,
            }
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
    "Llama_8bit": {
        "model": [
            {
                "pretrained_model_name": "pablocosta/llama-7b",
                "hf_model_load_in_8bit":True,
                "hf_model_use_auth_token": "",
                "hf_tokenizer_use_auth_token": "",
            },
        ],
        "prompt": [
            {
                "prompt_template_file": "./example_llm/ustancebr_prompts/stance_prompt_alpaca_score10_0.md",
                "output_max_score": 10,
            },
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
    "Bloom": {
        "model": [
            {
                "auth_token": "",
            },
        ],
        "prompt": [
            {
                "prompt_template_file": "./example_llm/ustancebr_prompts/stance_prompt_alpaca_score10_0.md",
                "output_max_score": 10,
            },
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
}

values_dict_semeval = {
    "Llama_4bit": {
        "pretrained_model_name": [
            "../../data/LLMs/ggml-alpaca-7b-q4.bin",
        ],
        "prompt": [
            {
                "prompt_template_file": "./example_llm/semeval_prompts/stance_prompt_alpaca_score10_0.md",
                "output_max_score": 10,
            }
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
    "Llama_8bit": {
        "model": [
            {
                "pretrained_model_name": "pablocosta/llama-7b",
                "hf_model_load_in_8bit":True,
                "hf_model_use_auth_token": "",
                "hf_tokenizer_use_auth_token": "",
            },
        ],
        "prompt": [
            {
                "prompt_template_file": "./example_llm/semeval_prompts/stance_prompt_alpaca_score10_0.md",
                "output_max_score": 10,
            }
        ],
        # "batch_size": [
        #     # 80,
        #     16,
        # ],
    },
    "Bloom": {
        "model": [
            {
                "auth_token": "",
            },
        ],
        "prompt": [
            {
                "prompt_template_file": "./example_llm/semeval_prompts/stance_prompt_alpaca_score10_0.md",
                "output_max_score": 10,
            },
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

    if dataset  == "semeval":
        values_dict = values_dict_semeval
        default_params = default_params_semeval
    elif dataset  == "ustancebr":
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
            new_config_dict = base_config_dict
            # setting default params
            for key, value in default_params.items():
                new_config_dict[key] = value

            # setting combination specific params
            for key, value in zip(values_dict[model_name_out_file].keys(), combination):
                if key in ["prompt", "model"]:
                    for prompt_key, prompt_value in value.items():
                        new_config_dict[prompt_key] = prompt_value
                else:
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
        