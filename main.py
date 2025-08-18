from model_manager import ModelManager
import yaml

import torch


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    # categories = config["categories"]
    manager = ModelManager(config)
    # manager.generate_system_prompt(categories)
    manager.categories = manager.generate_categories(config["categories"])
    print(manager.categories)
    # print(manager.prompt_model("przyjazna atmosfera").shape)
    # print(manager.categories[torch.argmax(manager.prompt_model("bulka"))])
    print(manager.categories[manager.prompt_model("query: ok")])


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
