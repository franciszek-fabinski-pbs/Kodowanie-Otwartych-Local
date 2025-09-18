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


def pretty_print_sim_idx(
    names, S, decimals=3, hide_diag=True, start=0, pad=2, show_map=True, maxlen=60
):
    """
    names  : list of categories full names (np. ["passage: ...", ...])
    S      : similiarity matrix (NxN) - torch.Tensor or np.ndarray
    decimals: liczba miejsc po przecinku
    hide_diag: replace diagonal by '—' for easier reading
    start  : indexing starting index
    pad    : number of digits in the index (e.g. 2 -> 00, 01, 02)
    show_map: whether to print index -> name mapping
    maxlen : max. row length in mapping (print only)
    """
    # 1) clean out category names
    clean = [n.removeprefix("passage: ").strip() for n in names]

    # 2) prepare the matrix for numpy
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    S = np.asarray(S)

    # 3) labels = index only
    labels = [f"{i + start:0{pad}d}" for i in range(len(clean))]

    # 4) DataFrame
    df = pd.DataFrame(S, index=labels, columns=labels)

    # 5) hide diagonal (better readability)
    if hide_diag:
        np.fill_diagonal(df.values, np.nan)

    # 6) print out the table
    print(
        df.to_string(float_format=lambda x: f"{x:.{decimals}f}" if pd.notna(x) else "—")
    )

    # 7) optional: index → full name mapping
    if show_map:
        print("\nMapowanie indeksów:")
        for idx, name in zip(labels, clean):
            t = name if len(name) <= maxlen else name[: maxlen - 1] + "…"
            print(f"  {idx}  {t}")


def extract_column_from_csv(
    filename: str, column_idx: int, data_type: type
) -> pd.DataFrame:
    """
    extracting a single @filename.csv column at index @column_idx as @data_type
    """
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
    Reads selected columns from CSV file.
    - columns: index list/set, slice (e.g. slice(3, None, 2)) or range.
               if None -> all columns.
    - dtype: single type (e.g. float) or dictionary {column: type}.
             works like DataFrame.astype(...).
    - return_numpy: True -> returns (np.ndarray), False -> pd.DataFrame.
    """
    # if columns is a slice with no .stop set,
    # we need to know the number of columns
    if isinstance(columns, slice) and columns.stop is None:
        header_only = pd.read_csv(filename, sep=sep, header=header, nrows=0)
        ncols = header_only.shape[1]
        usecols = list(range(columns.start or 0, ncols, columns.step or 1))
    elif isinstance(columns, (range, list, tuple, set)):
        usecols = list(columns)  # pandas requires list/tuple
    else:
        usecols = columns  # None or full slice with set .stop

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


def optimize_classification_params(
    model_manager: ModelManager,
    cat_manager: CategoryManager,
    prompts: list[str],
    human_classification: np.ndarray,
    config: dict,
    subset_n: int | None = None,
    max_iters: int = 12,
):
    # Order-insensitive optimization using set overlap (Jaccard-based)
    def compute_loss(
        threshold: float, margin: float, min_sim: float, use_n: int | None = None
    ) -> float:
        use_prompts = prompts if use_n is None else prompts[:use_n]
        model_manager.prompt_model_multi_batch(
            use_prompts,
            cat_manager.categories,
            cat_manager.categories_encoded,
            threshold=threshold,
            margin=margin,
            min_similiarity=min_sim,
            show_all_sims=False,
            top_k=None,
            **{
                k: v
                for k, v in config.get("classification", {}).items()
                if k in ["batch_size", "intro"]
            },
        )
        results, _ = model_manager.get_results()
        total = 0.0
        count = 0
        for idx, result in enumerate(results):
            hc_row = human_classification[idx]
            human_ids = set(map(str, hc_row[~np.isnan(hc_row)].astype(int).tolist()))
            if not human_ids:
                continue
            ai_ids = set(str(res[0]) for res in result)
            union = human_ids | ai_ids
            inter = human_ids & ai_ids
            jaccard = len(inter) / max(len(union), 1)
            diff = 1.0 * (1 - jaccard)
            total += diff
            count += 1
            # print(f"ai len: {len(ai_ids)}, human len: {len(human_ids)}")
            # print(f"union len: {len(union)}, inter len: {len(inter)}")
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
    lr_t, lr_m, lr_s = 0.2, 0.2, 0.2
    eps = 8e-2
    eps_t_rate = 0.005
    eps_m_rate = 0.006
    eps_s_rate = 0.008

    best_t, best_m, best_s = t, m, s
    best_loss = compute_loss(best_t, best_m, best_s, use_n=subset_n)
    loss_history: list[float] = []
    print(
        f"[opt] init loss={best_loss:.6f} with t={best_t:.4f}, m={best_m:.4f}, s={best_s:.4f}"
    )

    for it in range(1, max_iters + 1):
        print("=" * 20)
        base_loss = compute_loss(t, m, s, use_n=subset_n)
        loss_history.append(float(base_loss) * 100)
        print("=" * 20)

        l_p = compute_loss(clip01(t + eps_t_rate * eps), m, s, use_n=subset_n)
        l_n = compute_loss(clip01(t - eps_t_rate * eps), m, s, use_n=subset_n)
        g_t = (l_p - l_n) / (2 * eps)
        print(f"threshold gain: {g_t}")

        l_p = compute_loss(t, clip01(m + eps_m_rate * eps), s, use_n=subset_n)
        l_n = compute_loss(t, clip01(m - eps_m_rate * eps), s, use_n=subset_n)
        g_m = (l_p - l_n) / (2 * eps)
        print(f"margin gain: {g_m}")

        l_p = compute_loss(t, m, clip01(s + eps_s_rate * eps), use_n=subset_n)
        l_n = compute_loss(t, m, clip01(s - eps_s_rate * eps), use_n=subset_n)
        g_s = (l_p - l_n) / (2 * eps)
        print(f"minimum similiarity gain: {g_s}")

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
            eps_t_rate *= 0.95
            eps_m_rate *= 0.95
            eps_s_rate *= 0.95

    print(
        f"[opt] best_loss={best_loss:.6f} with threshold={best_t:.6f}, margin={best_m:.6f}, min_similiarity={best_s:.6f}"
    )
    final_loss = compute_loss(best_t, best_m, best_s, use_n=None)
    print(f"[opt] final full loss={final_loss:.6f}")

    # Save loss curve to file
    try:
        import matplotlib.pyplot as plt

        if loss_history:
            xs = list(range(1, len(loss_history) + 1))
            plt.figure(figsize=(6, 4))
            plt.plot(xs, loss_history, marker="o", linewidth=2)
            plt.xlabel("Iteracja")
            plt.ylabel("Loss [%]")
            plt.title("Loss na iterację (subset)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("loss.png", dpi=150)
            plt.close()
            print("[opt] Zapisano wykres loss do pliku: loss.png")
    except Exception as e:
        print(f"[opt] Nie udało się zapisać wykresu loss: {e}")
    return best_t, best_m, best_s, best_loss, final_loss


def main():
    with open("config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    model_manager = ModelManager(config)

    cls_cfg = config.get("classification", {})
    t = float(cls_cfg.get("threshold", 0.003))
    m = float(cls_cfg.get("margin", 0.025))
    s = float(cls_cfg.get("min_similiarity", 0.857))

    with open("data/2024-kat.json", "r") as categories_file:
        categories = json.load(categories_file)
        categories = categories["categories"]
        print(f"ilosc kat: {len(categories)}")

    cat_manager = CategoryManager(categories)
    # model_manager.pull_categories(categories)
    cat_manager.encode(model_manager.encode)

    prompts = None
    # prompts = extract_column_from_csv(
    #     "./data/2023-u8.csv", column_idx=1, data_type=str
    # ).tolist()

    prompts = extract_column_from_csv(
        "./data/2024-u8.csv", column_idx=1, data_type=str
    ).tolist()

    # clean out prompts from low entropy answers, put them in flagged category
    thr = 2.5
    triples = [
        (i, p, model_manager.calculate_entropy(p)) for i, p in enumerate(prompts)
    ]

    flagged_idx = [i for i, p, e in triples if e < thr]
    flagged = [p for i, p, e in triples if e < thr]

    kept_idx = [i for i, p, e in triples if e >= thr]
    kept = [p for i, p, e in triples if e >= thr]

    from sentence_transformers import util

    cat_self_sim = util.cos_sim(
        cat_manager.categories_encoded, cat_manager.categories_encoded
    )

    human_classification = extract_columns(
        "./data/2024-u8.csv", columns=slice(2, None, 1), return_numpy=True
    )
    human_classification = human_classification[kept_idx]

    # --- Optimize params ---
    # best_t, best_m, best_s, best_loss, final_loss = optimize_classification_params(
    #     model_manager,
    #     cat_manager,
    #     kept,
    #     human_classification,
    #     config,
    #     subset_n=None,
    #     max_iters=25,
    # )

    best_t = t
    best_m = m
    best_s = s

    # --- Final evaluation on full set with best params ---
    model_manager.prompt_model_multi_batch(
        kept,
        cat_manager.categories,
        cat_manager.categories_encoded,
        threshold=best_t,
        margin=best_m,
        min_similiarity=best_s,
        show_all_sims=False,
        top_k=None,
        **{
            k: v
            for k, v in config.get("classification", {}).items()
            if k in ["batch_size", "intro"]
        },
    )
    results, _ = model_manager.get_results()
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
        print(f"{prompt} (entropy: {model_manager.calculate_entropy(prompt)})")
        print(f"Human(K={K}): {hc}")
        for i, cat in enumerate(cat_manager.get_by_id(hc)):
            print(f"{i}: {cat.name} --- {None}")

        print(f"AI(K={K}): {ai_top_k}")
        for i, cat in enumerate(cat_manager.get_by_id(ai_top_k)):
            print(f"{i}: {cat.name} --- {results[idx][i]}")
            cat_manager.classification_counter[str(cat.id)] += 1
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

    cat_manager.classification_counter[str(99)] += len(flagged)

    human_len_mean = np.mean(human_len)
    print(f"AI AVERAGE CLASS COUNT: {ai_len_mean}")
    print(f"HUMAN AVERAGE CLASS COUNT: {human_len_mean}")
    print("Classification counter: ")
    for k, v in cat_manager.classification_counter.items():
        print(f"{k}: {v}")

    # Second: category self-similarity table for reference (unchanged)
    # model_manager.prompt_model_multi_batch(
    #     [cat.name for cat in cat_manager.categories],
    #     cat_manager.categories,
    #     cat_manager.categories_encoded,
    #     show_all_sims=True,
    #     **config["classification"],
    # )
    # results, self_sim = model_manager.get_results()
    # pretty_print_sim_idx(cat_manager.cat_names, self_sim, decimals=5)
    # print(best_t, best_m, best_s, best_loss, final_loss)

    # print("flagged low-entropy prompts:")
    # for idx, p in zip(flagged_idx, flagged):
    #     print(f"{idx}: {p}")
    #
    # print("non-flagged low-entropy prompts:")
    # for idx, p in zip(kept_idx, kept):
    #     print(f"{idx}: {p}")


if __name__ == "__main__":
    # torch.cuda.empty_cache()
    main()
