import json
from typing import Iterable
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import torch
import yaml

from data_manager import CategoryManager
from model_manager import ModelManager


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


def optimize_classification_params(  # {{{
    model_manager: ModelManager,
    cat_manager: CategoryManager,
    prompts: list[str],
    human_classification: np.ndarray,
    config: dict,
    subset_n: int | None = None,
    max_iters: int = 12,
):
    from sentence_transformers import util

    # Compute category self-similarity once (for loss)
    cat_self_sim = util.cos_sim(
        cat_manager.categories_encoded, cat_manager.categories_encoded
    )

    def compute_loss(
        threshold: float, margin: float, min_sim: float, use_n: int | None = None
    ) -> float:
        use_prompts = prompts if use_n is None else prompts[:use_n]
        # Determine per-prompt K (number of human labels)
        k_list = []
        for idx in range(len(use_prompts)):
            hc_row = human_classification[idx]
            k = np.count_nonzero(~np.isnan(hc_row))
            k_list.append(int(k))
        k_max = max([k for k in k_list if k > 0] or [1])

        model_manager.prompt_model_multi_batch(
            use_prompts,
            cat_manager.categories,
            cat_manager.categories_encoded,
            threshold=threshold,
            margin=margin,
            min_similiarity=min_sim,
            show_all_sims=False,
            return_top_n=k_max,
            **{
                k: v
                for k, v in config.get("classification", {}).items()
                if k in ["top_k", "batch_size", "intro"]
            },
        )
        results, _ = model_manager.get_results()
        total = 0.0
        count = 0
        for idx, result in enumerate(results):
            hc_row = human_classification[idx]
            hc = hc_row[~np.isnan(hc_row)].astype(int).tolist()
            K = len(hc)
            if K == 0:
                continue
            ai_ids = [res[0] for res in result][:K]
            # Average pairwise rank-aligned distance across K
            diffs = []
            for h_id, a_id in zip(hc, ai_ids):
                h_idx = cat_manager.get_index_by_id(h_id)
                a_idx = cat_manager.get_index_by_id(a_id)
                diffs.append(1 - float(cat_self_sim[h_idx][a_idx]))
            total += float(np.mean(diffs))
            count += 1
        return total / max(count, 1)

    def clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 0.9999))

    # Initialize from config
    cls_cfg = config.get("classification", {})
    t = float(cls_cfg.get("threshold", 0.003))
    m = float(cls_cfg.get("margin", 0.025))
    s = float(cls_cfg.get("min_similiarity", 0.857))

    # Hyperparameters for optimization
    subset_n = (
        min(len(prompts), 512) if subset_n is None else min(len(prompts), subset_n)
    )
    lr_t, lr_m, lr_s = 0.008, 0.003, 0.01
    eps = 1e-2

    best_t, best_m, best_s = t, m, s
    best_loss = compute_loss(best_t, best_m, best_s, use_n=subset_n)
    print(
        f"[opt] init loss={best_loss:.6f} with t={best_t:.4f}, m={best_m:.4f}, s={best_s:.4f}"
    )

    for it in range(1, max_iters + 1):
        base_loss = compute_loss(t, m, s, use_n=subset_n)

        l_p = compute_loss(clip01(t + eps), m, s, use_n=subset_n)
        l_n = compute_loss(clip01(t - eps), m, s, use_n=subset_n)
        g_t = (l_p - l_n) / (2 * eps)

        l_p = compute_loss(t, clip01(m + eps), s, use_n=subset_n)
        l_n = compute_loss(t, clip01(m - eps), s, use_n=subset_n)
        g_m = (l_p - l_n) / (2 * eps)

        l_p = compute_loss(t, m, clip01(s + eps), use_n=subset_n)
        l_n = compute_loss(t, m, clip01(s - eps), use_n=subset_n)
        g_s = (l_p - l_n) / (2 * eps)

        # gradient step
        t_new = clip01(t - lr_t * g_t)
        m_new = clip01(m - lr_m * g_m)
        s_new = clip01(s - lr_s * g_s)

        new_loss = compute_loss(t_new, m_new, s_new, use_n=subset_n)
        improved = new_loss < best_loss - 1e-6
        if improved:
            best_t, best_m, best_s = t_new, m_new, s_new
            best_loss = new_loss

        print(
            f"[opt] it={it:02d} base_loss={base_loss:.6f} new_loss={new_loss:.6f} | t={
                t_new:.4f} m={m_new:.4f} s={s_new:.4f} | grad=({g_t:.4f},{g_m:.4f},{
                g_s:.4f})"
        )

        if new_loss <= base_loss:
            t, m, s = t_new, m_new, s_new
        else:
            lr_t *= 0.8
            lr_m *= 0.8
            lr_s *= 0.8

    print(
        f"[opt] best_loss={best_loss:.6f} with threshold={best_t:.6f}, margin={best_m:.6f}, min_similiarity={best_s:.6f}"
    )
    final_loss = compute_loss(best_t, best_m, best_s, use_n=None)
    print(f"[opt] final full loss={final_loss:.6f}")
    return best_t, best_m, best_s, best_loss, final_loss


# }}}


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    model_manager = ModelManager(config)

    with open("data/2024-kat.json", "r") as categories_file:
        categories = json.load(categories_file)
        categories = categories["categories"]
        print(f"ilosc kat: {len(categories)}")
    model_manager.pull_categories(categories)

    cat_manager = CategoryManager(categories)
    cat_manager.encode(model_manager.encode)

    prompts = None
    # prompts = extract_column_from_csv(
    #     "./data/2023-u8.csv", column_idx=1, data_type=str
    # ).tolist()

    prompts = extract_column_from_csv(
        "./data/2024-u8.csv", column_idx=1, data_type=str
    ).tolist()

    from sentence_transformers import util

    cat_self_sim = util.cos_sim(
        cat_manager.categories_encoded, cat_manager.categories_encoded
    )

    human_classification = extract_columns(
        "./data/2024-u8.csv", columns=slice(2, None, 1), return_numpy=True
    )

    # --- Optimize params ---
    best_t, best_m, best_s, best_loss, final_loss = optimize_classification_params(
        model_manager,
        cat_manager,
        prompts,
        human_classification,
        config,
        subset_n=None,
        max_iters=20,
    )

    # --- Final evaluation on full set with best params ---
    model_manager.prompt_model_multi_batch(
        prompts,
        cat_manager.categories,
        cat_manager.categories_encoded,
        threshold=best_t,
        margin=best_m,
        min_similiarity=best_s,
        show_all_sims=False,
        **{
            k: v
            for k, v in config.get("classification", {}).items()
            if k in ["top_k", "batch_size", "intro"]
        },
    )
    results, _ = model_manager.get_results()

    for idx, result in enumerate(results):
        prompt = prompts[idx]
        ai_classification = [res[0] for res in result]
        hc_row = human_classification[idx]
        hc = hc_row[~np.isnan(hc_row)].astype(int).tolist()
        if not hc:
            continue
        K = min(len(hc), len(ai_classification))
        ai_top_k = sorted(ai_classification)
        hc = sorted(hc[:K])
        print("-" * 40)
        print(f"{prompt} (entropy: {model_manager.calculate_entropy(prompt)})")
        print(f"Human(K={K}): {hc}-{cat_manager.get_by_id(hc)[0].name}")
        print(f"AI(K={K}): {ai_top_k}-{cat_manager.get_by_id([ai_top_k])[0].name}")
        diffs = [
            1
            - cat_self_sim[cat_manager.get_index_by_id(h)][
                cat_manager.get_index_by_id(a)
            ]
            for h, a in zip(hc[:K], ai_top_k)
        ]
        diffs = [d.item() for d in diffs]
        print(f"avg diff over top-{K}: {float(np.mean(diffs[0]))}")
        print("-" * 40)

    # Second: category self-similarity table for reference (unchanged)
    model_manager.prompt_model_multi_batch(
        [cat.name for cat in cat_manager.categories],
        cat_manager.categories,
        cat_manager.categories_encoded,
        show_all_sims=True,
        **config["classification"],
    )
    results, self_sim = model_manager.get_results()
    # pretty_print_sim_idx(cat_manager.cat_names, self_sim, decimals=5)
    print(best_t, best_m, best_s, best_loss, final_loss)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
