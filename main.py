from model_manager import ModelManager
import yaml
import json

import numpy as np
import pandas as pd
import torch


def pretty_print_sim_idx(
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


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    manager = ModelManager(config)

    with open("data/categories.json", "r") as categories_file:
        categories = json.load(categories_file)
        categories = categories["categories"]
        print(f"ilosc kat: {len(categories)}")
    manager.pull_categories(categories)
    prompts = None
    # with open("data/prompts.json", "r") as prompts_file:
    #     prompts = json.load(prompts_file)
    #     prompts = prompts["answers"]

    df = pd.read_csv("data/zbior.csv", sep=";", header=None, usecols=[2])  # 3. kolumna
    prompts = df.iloc[:, 0].astype(str).str.strip().tolist()

    manager.prompt_model_multi_batch(prompts, show_all_sims=True, **config["classification"])
    results, self_sim = manager.get_results()

    for idx, result in enumerate(results):
        prompt = prompts[idx]
        print("-" * 40)
        print(f"{prompt}, entropia: {manager.calculate_entropy(prompt)}:")
        print("-" * 10)
        for idx, (cat, score, score_norm) in enumerate(result, start=1):
            print(f"\t{idx:<4}: {cat:<55} {score:>18.16f} {score_norm:>18.16f}")

    # pretty_print_sim_idx(prompts, self_sim, decimals=5)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
