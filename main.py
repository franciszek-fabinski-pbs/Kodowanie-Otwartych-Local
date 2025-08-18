from model_manager import ModelManager
import yaml

import torch


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    manager = ModelManager(config)
    manager.categories = manager.generate_categories(config["categories"])
    print(manager.categories)
    prompt = " - warunki pracy - elastyczne godziny pracy,  możliwość pracy zdalnej  - wynagrodzenie (uwzględniające poza wynagrodzeniem zasadniczym nagrody, 13stki)  - lokalizacja gmachu  - prestiż instytucji"
    result = manager.categories[manager.prompt_model(prompt)]
    print(f"{prompt} ---> {result}")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
