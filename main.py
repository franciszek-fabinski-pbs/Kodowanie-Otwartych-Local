from model_manager import ModelManager
import yaml
import json

import torch


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    manager = ModelManager(config)

    with open("data/categories.json", "r") as categories_file:
        categories = json.load(categories_file)
        categories = categories["categories"]
    manager.pull_categories(categories)
    prompts = None
    with open("data/prompts.json", "r") as prompts_file:
        prompts = json.load(prompts_file)
        prompts = prompts["answers"]
    results = []

    for prompt in prompts:
        result = manager.prompt_model_multi(prompt, **config["classification"])
        results.append(result)
        print("-" * 40)
        print(f"{prompt}, entropia: {manager.calculate_entropy(prompt)}:")
        print("-" * 10)
        for idx, (cat, score, score_norm) in enumerate(result, start=1):
            print(f"\t{idx:<4}: {cat:<55} {score:>18.16f} {score_norm:>18.16f}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
