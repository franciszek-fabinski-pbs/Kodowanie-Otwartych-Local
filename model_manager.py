import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from data_types import Category


class ModelManager:
    """
    Classification Model Manager for loading models and accepting prompts.
    """

    def __init__(self, config: dict):
        """
        config: dictionary with properties:
            device: torch.device, "cuda"/"cpu" etc
            model: model directory path
            categories: array of dicts:
                id: id of the category
                name: name of the category
        """
        model_name: str = config["model"]
        self.model = SentenceTransformer(model_name)
        self.device: torch.device | str = config["device"]
        self.categories_meta: list[dict] = None
        self.categories_names: list[str] = None
        self.category_embeddings: torch.Tensor = None
        self.prompt_embeddings: torch.Tensor = None
        self.sim_results = None

        self.model.to(self.device)

    def prompt_model(self, prompt: str) -> int:
        """
        Prompt model with an answer/sentence, no special formatting needed.

        Returns index of the most fitting category.
        """
        prompt = prompt
        # prompt = "query: " + prompt
        prompt_embedding = self.model.encode(
            [prompt], convert_to_tensor=True, normalize_embeddings=True
        )

        sims = util.cos_sim(prompt_embedding, self.category_embeddings).squeeze(0)
        best_idx = int(torch.argmax(sims))

        return best_idx

    def prompt_model_multi_batch(
        self,
        prompts: list[str],
        categories: list[Category],
        categories_encoded: torch.Tensor,
        threshold: float | None = 0.35,
        margin: float | None = 0.02,
        top_k: int | None = None,
        batch_size: int = 32,
        min_similiarity: float | None = 0.857,
        min_local_similiarity: float | None = 0.85,
        show_all_sims: bool = False,
        intro: str = "",
        return_top_n: int | None = None,
    ) -> list[tuple[int, str, float, float]]:
        q_prompts = [f"query: {intro} {p}" for p in prompts]

        Q = self.model.encode(
            q_prompts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=batch_size,
        )  # shape: (M, d)
        self.prompt_embeddings = Q

        S = util.cos_sim(Q, categories_encoded)
        result = []
        print(
            f"calling prompt with args: threshold-{threshold}, margin-{margin}, min_similiarity-{min_similiarity}"
        )
        for m in range(S.size(0)):
            row = S[m]
            s_min = row.min().item()
            s_max = row.max().item()

            row_norm = (row - s_min) / (s_max - s_min + 1e-9)
            # Sort all categories by score descending
            top_idx, top_vals = map(
                list,
                zip(*sorted(enumerate(row), key=lambda val: val[1], reverse=True)),
            )
            top_norm = row_norm[top_idx]
            best = float(top_vals[0])

            # Apply threshold-based selection when top_k is None
            picked = []
            for score_tensor, score_norm, idx in zip(top_vals, top_norm, top_idx):
                score = float(score_tensor)
                score_n = float(score_norm)
                # All provided constraints must pass; unspecified ones are ignored
                if threshold is not None and score < float(threshold):
                    continue
                if margin is not None and score < best - float(margin):
                    continue
                if min_similiarity is not None and score < float(min_similiarity):
                    continue
                if min_local_similiarity is not None and score_n < float(min_local_similiarity):
                    continue
                picked.append((idx, score))

            # If nothing matched constraints, fall back to top-1
            if not picked:
                picked = [(int(top_idx[0]), float(top_vals[0]))]

            # If explicit top_k is requested, cap the number of picks after filtering
            if top_k is not None:
                k = max(0, min(int(top_k), len(picked)))
                picked = picked[:k] if k > 0 else [(int(top_idx[0]), float(top_vals[0]))]

            if show_all_sims:
                order = torch.argsort(row, descending=True)
            else:
                order = [res[0] for res in sorted(picked, key=lambda pick: pick[1], reverse=True)]

            row_norm = row_norm[order]
            row = row[order]
            row_list = [
                (
                    categories[i].id,
                    categories[i].name.removeprefix("passage: ").strip(),
                    s.item(),
                    s_norm.item(),
                )
                for i, s, s_norm in zip(order, row, row_norm)
            ]

            if return_top_n is not None:
                row_list = row_list[: max(0, min(return_top_n, len(row_list)))]

            result.append(row_list)

        self.sim_results = result
        return result

    def pull_categories(self, categories: list[Category]) -> None:
        """
        Generate a list of category names without ids from dict list and format
        it for the model.
        """
        result = []
        self.categories_meta = categories
        for cat in categories:
            tags = cat.keywords
            result.append("passage: " + tags)
            # result.append(name)
        self.categories_names = result
        # self.category_embeddings = self.model.encode(
        #     self.categories_names, convert_to_tensor=True, normalize_embeddings=True
        # )

    def encode(
        self, data: list[str], prefix: str | None = None, batch_size: int = 32
    ) -> torch.Tensor:
        if prefix is not None:
            data = [prefix + d for d in data]
        return self.model.encode(
            data,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=batch_size,
        )

    def get_results(self):
        sims_self = util.cos_sim(self.prompt_embeddings, self.prompt_embeddings)
        return self.sim_results, sims_self

    @staticmethod
    def calculate_entropy(prompt: str):
        entropy = 0
        letters = "".join(set(prompt))

        for i in letters:
            x = prompt.count(i)
            y = x = x / len(prompt)
            x = np.log2(x)
            x = x * y
            entropy = entropy + x
        return -entropy
