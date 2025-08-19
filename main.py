from model_manager import ModelManager
import yaml
import json

import torch


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    manager = ModelManager(config)
    manager.generate_categories(config["categories"])
    print(manager.categories)
    prompts = None
    with open("prompts.json", "r") as prompts_file:
        prompts = json.load(prompts_file)
        prompts = prompts["answers"]
    results = []
    for prompt in prompts:
        result = manager.prompt_model_multi(prompt, top_k=3, threshold=0.15)
        results.append(result)
        print(f"{prompt} ---> {result}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
