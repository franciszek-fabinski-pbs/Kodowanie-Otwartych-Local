from model_manager import ModelManager
import yaml
import json

import numpy as np
import pandas as pd
import torch


def pretty_print_sim_idx(  # {{{
    names, S, decimals=3, hide_diag=True, start=0, pad=2, show_map=True, maxlen=60
):
    """
    names  : lista pełnych nazw kategorii (np. ["passage: ...", ...])
    S      : macierz podobieństw (NxN) - torch.Tensor lub np.ndarray
    decimals: liczba miejsc po przecinku
    hide_diag: zamień diagonalę na '—' dla czytelności
    start  : od jakiego indeksu numerować (np. 0 lub 1)
    pad    : ile cyfr w indeksie (np. 2 -> 00, 01, 02)
    show_map: czy wydrukować mapowanie indeks -> nazwa
    maxlen : maks. długość wiersza w mapowaniu (tylko do wydruku)
    """
    # 1) oczyść nazwy (opcjonalnie usuń "passage: ")
    clean = [n.removeprefix("passage: ").strip() for n in names]

    # 2) przygotuj macierz jako numpy
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    S = np.asarray(S)

    # 3) etykiety = same indeksy
    labels = [f"{i + start:0{pad}d}" for i in range(len(clean))]

    # 4) DataFrame
    df = pd.DataFrame(S, index=labels, columns=labels)

    # 5) schowaj diagonalę (ładniejszy podgląd)
    if hide_diag:
        np.fill_diagonal(df.values, np.nan)

    # 6) wydruk tabeli
    print(
        df.to_string(float_format=lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "—")
    )

    # 7) opcjonalne mapowanie indeks → pełna nazwa
    if show_map:
        print("\nMapowanie indeksów:")
        for idx, name in zip(labels, clean):
            t = name if len(name) <= maxlen else name[: maxlen - 1] + "…"
            print(f"  {idx}  {t}")


# }}}


def extract_column_from_csv(
    filename: str, column_idx: int, data_type: type
) -> pd.DataFrame:
    df = pd.read_csv(
        filename, sep=";", header=0, usecols=[column_idx], skip_blank_lines=False
    )
    results = df.iloc[:, 0].astype(data_type)
    return results


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    manager = ModelManager(config)

    with open("data/2023-kat.json", "r") as categories_file:
        categories = json.load(categories_file)
        categories = categories["categories"]
        print(f"ilosc kat: {len(categories)}")
    manager.pull_categories(categories)
    prompts = None

    prompts = extract_column_from_csv(
        "./data/2023-u8.csv", column_idx=1, data_type=str
    ).tolist()  # TODO

    manager.prompt_model_multi_batch(
        prompts, show_all_sims=True, **config["classification"]
    )
    results, self_sim = manager.get_results()

    # for idx, result in enumerate(results):
    #     prompt = prompts[idx]
    #     print("-" * 40)
    #     print(f"{prompt}, entropia: {manager.calculate_entropy(prompt)}:")
    #     print("-" * 10)
    #     for idx, (cat_idx, cat, score, score_norm) in enumerate(result, start=1):
    #         print(f"\t{idx:<4}: {cat:<55} {score:>18.16f} {score_norm:>18.16f}")

    human_classification = extract_column_from_csv(
        "./data/2023-u8.csv", 2, int
    ).tolist()

    for idx, result in enumerate(results):
        prompt = prompts[idx]
        ai_classification = result[0][0]
        hc = human_classification[idx]
        print("-" * 40)
        print(f"{prompt}:")
        print(f"Human: {hc}-{categories[hc]}")
        print(f"AI: {ai_classification}-{categories[ai_classification]}")
        print("-" * 40)

    # pretty_print_sim_idx(prompts, self_sim, decimals=5)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
