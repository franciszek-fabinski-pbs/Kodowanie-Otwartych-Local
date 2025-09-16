from typing import Iterable, Optional, Union
from model_manager import ModelManager
from data_manager import CategoryManager
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


def extract_columns(
    filename: str,
    columns: Optional[Union[Iterable[int], slice, range]] = None,
    dtype: Optional[Union[type, dict]] = None,
    sep: str = ";",
    header: int = 0,
    skip_blank_lines: bool = False,
    return_numpy: bool = False,
):
    """
    Wczytuje wskazane kolumny z CSV.
    - columns: lista/zbiór indeksów, slice (np. slice(3, None, 2)), albo range.
               Jeśli None -> wszystkie kolumny.
    - dtype: pojedynczy typ (np. float) lub słownik {kolumna: typ}.
             Działa jak DataFrame.astype(...).
    - return_numpy: True -> zwróci macierz (ndarray), False -> DataFrame.
    """
    # Jeśli columns to slice bez stopu, potrzebujemy znać liczbę kolumn:
    if isinstance(columns, slice) and columns.stop is None:
        header_only = pd.read_csv(filename, sep=sep, header=header, nrows=0)
        ncols = header_only.shape[1]
        usecols = list(range(columns.start or 0, ncols, columns.step or 1))
    elif isinstance(columns, (range, list, tuple, set)):
        usecols = list(columns)  # pandas wymaga listy/tupli
    else:
        usecols = columns  # None lub pełny slice z .stop zdefiniowanym

    df = pd.read_csv(
        filename,
        sep=sep,
        header=header,
        usecols=usecols,
        skip_blank_lines=skip_blank_lines,
    )

    if dtype is not None:
        df = df.astype(dtype)

    return df.to_numpy() if return_numpy else df


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    model_manager = ModelManager(config)

    with open("data/2023-kat.json", "r") as categories_file:
        categories = json.load(categories_file)
        categories = categories["categories"]
        print(f"ilosc kat: {len(categories)}")
    model_manager.pull_categories(categories)

    cat_manager = CategoryManager(categories)
    cat_manager.encode(model_manager.encode)

    prompts = None
    prompts = extract_column_from_csv(
        "./data/2023-u8.csv", column_idx=1, data_type=str
    ).tolist()

    model_manager.prompt_model_multi_batch(
        prompts,
        cat_manager.categories,
        cat_manager.categories_encoded,
        show_all_sims=True,
        **config["classification"],
    )
    results, self_sim = model_manager.get_results()

    human_classification = extract_columns(
        "./data/2023-u8.csv", columns=slice(2, None, 2), return_numpy=True
    )

    for idx, result in enumerate(results):
        prompt = prompts[idx]
        ai_classification = [res[0] for res in result]
        hc = human_classification[idx]
        print("-" * 40)
        print(f"{prompt}:")

        print(f"Human: {hc}-{cat_manager.get_by_id(hc[0])[0].name}")
        print(
            f"AI: {ai_classification[:7]}-{cat_manager.get_by_id([ai_classification[0]])[0].name}"
        )
        print("-" * 40)

    # pretty_print_sim_idx(prompts, self_sim, decimals=5)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
