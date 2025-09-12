from sentence_transformers import SentenceTransformer, util
from data_types import Category
import torch
import numpy as np


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
        show_all_sims: bool = False,
    ):
        q_prompts = [f"query: {p}" for p in prompts]

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
            if top_k is not None:
                k = min(top_k, row.size(0))
                top_vals, top_idx = torch.topk(row, k=k)
                best = top_vals[0].item()
                top_vals = top_vals.tolist()
                top_idx = top_idx.tolist()

            else:
                top_idx, top_vals = map(
                    list,
                    zip(*sorted(enumerate(row), key=lambda val: val[1], reverse=True)),
                )
                best = top_vals[0]

            picked = []
            for score, idx in zip(top_vals, top_idx):
                ok_threshold = threshold is not None and score >= threshold
                ok_margin = margin is not None and score >= best - margin
                ok_minimum = min_similiarity is not None and score > min_similiarity
                if (
                    threshold is None and margin is None and min_similiarity is not None
                ) or (ok_threshold and ok_margin and ok_minimum):
                    picked.append((idx, score))
            if not picked:
                picked = [(int(top_idx[0]), float(top_vals[0]))]

            if show_all_sims:
                order = torch.argsort(row, descending=True)
            else:
                order = [
                    res[0]
                    for res in sorted(picked, key=lambda pick: pick[1], reverse=True)
                ]

            row_norm = row_norm[order]
            row = row[order]
            result.append(
                [
                    (
                        categories[i].id,
                        self.categories_names[i].removeprefix("passage: ").strip(),
                        s.item(),
                        s_norm.item(),
                    )
                    for i, s, s_norm in zip(order, row, row_norm)
                ]
            )

        self.sim_results = result
        return result

    def pull_categories(self, categories: list[dict]) -> None:
        """
        Generate a list of category names without ids from dict list and format
        it for the model.
        """
        result = []
        self.categories_meta = categories
        for cat in categories:
            name = cat["name"]
            result.append("passage: " + name)
            # result.append(name)
        self.categories_names = result
        self.category_embeddings = self.model.encode(
            self.categories_names, convert_to_tensor=True, normalize_embeddings=True
        )

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
