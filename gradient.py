from model_manager import ModelManager
from data_manager import CategoryManager
import numpy as np


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
