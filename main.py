import json
import yaml

import numpy as np

from data_manager import CategoryManager
from data_util import calculate_entropy
from data_util import extract_column_from_csv
from data_util import extract_columns
from model_manager import ModelManager


def analyze_results(
    results: list[list[tuple[int, float]]],
    human_classification,
    cat_manager: CategoryManager,
    kept,
    flagged,
) -> None:
    ai_len_mean = np.mean([len(classification) for classification in results])
    human_len = []
    for idx, result in enumerate(results):
        prompt = kept[idx]
        ai_classification = [res[0] for res in result]
        hc_row = human_classification[idx]
        hc = hc_row[~np.isnan(hc_row)].astype(int).tolist()
        human_len.append(len(hc))
        if not hc:
            continue
        K = min(len(hc), len(ai_classification))
        ai_top_k = ai_classification
        hc = hc[:K]
        print("-" * 40)
        print(f"{prompt} (entropy: {calculate_entropy(prompt)})")
        print(f"Human(K={K}): {hc}")
        for i, cat in enumerate(cat_manager.get_by_id(hc)):
            print(f"{i}: {cat.name} --- {None}")

        print(f"AI(K={K}): {ai_top_k}")
        for i, cat in enumerate(cat_manager.get_by_id(ai_top_k)):
            print(f"{i}: {cat.name} --- {results[idx][i][1]}")
            cat_manager.classification_counter[str(cat.id)] += 1
        print("-" * 40)

    cat_manager.classification_counter[str(99)] += len(flagged)

    human_len_mean = np.mean(human_len)
    print(f"AI AVERAGE CLASS COUNT: {ai_len_mean}")
    print(f"HUMAN AVERAGE CLASS COUNT: {human_len_mean}")
    print("Classification counter: ")
    for k, v in cat_manager.classification_counter.items():
        print(f"{k}: {v}")


def pull_data():
    prompts = None
    prompts = extract_column_from_csv(
        "./data/2024-u8.csv", column_idx=1, data_type=str
    ).tolist()
    # prompts = extract_column_from_csv(
    #     "./data/2023-u8.csv", column_idx=1, data_type=str
    # ).tolist()
    with open("data/2024-kat.json", "r") as categories_file:
        categories = json.load(categories_file)
        categories = categories["categories"]
        print(f"ilosc kat: {len(categories)}")

    return prompts, categories


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    cls_cfg = config.get("classification", {})
    # t = float(cls_cfg.get("threshold", 0.003))
    # m = float(cls_cfg.get("margin", 0.025))
    # s = float(cls_cfg.get("min_similiarity", 0.857))

    prompts, categories = pull_data()

    model_manager = ModelManager(config)
    cat_manager = CategoryManager(categories)

    model_manager.pull_categories(cat_manager.categories)

    # clean out prompts from low entropy answers, put them in flagged category
    thr = 2.5
    triples = [(i, p, calculate_entropy(p)) for i, p in enumerate(prompts)]

    flagged_idx = [i for i, p, e in triples if e < thr]
    flagged = [p for i, p, e in triples if e < thr]
    kept_idx = [i for i, p, e in triples if e >= thr]
    kept = [p for i, p, e in triples if e >= thr]

    human_classification = extract_columns(
        "./data/2024-u8.csv", columns=slice(2, None, 1), return_numpy=True
    )
    human_classification = human_classification[kept_idx]

    results = model_manager.classify(kept)
    analyze_results(results, human_classification, cat_manager, kept, flagged)


if __name__ == "__main__":
    main()
