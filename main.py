from model_manager import ModelManager
import yaml

import torch


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    categories = config["categories"]
    manager = ModelManager(config)
    manager.generate_system_prompt(categories)
    manager.prompt_model("1574 - ...")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
