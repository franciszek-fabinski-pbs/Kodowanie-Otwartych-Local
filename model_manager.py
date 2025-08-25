from sentence_transformers import SentenceTransformer, util
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

        self.model.to(self.device)

    def prompt_model(self, prompt: str) -> int:
        """
        Prompt model with an answer/sentence, no special formatting needed.

        Returns index of the most fitting category.
        """
        prompt = prompt.lower()
        # prompt = "query: " + prompt
        prompt_embedding = self.model.encode(
            [prompt], convert_to_tensor=True, normalize_embeddings=True
        )

        sims = util.cos_sim(prompt_embedding, self.category_embeddings).squeeze(0)
        best_idx = int(torch.argmax(sims))

        return best_idx

    def prompt_model_multi(
        self,
        prompt: str,
        top_k: int = 3,
        threshold: float | None = 0.35,
        margin: float | None = 0.02,
    ) -> list[int]:
        prompt = "query: " + prompt
        prompt_embedding = self.model.encode_query(
            [prompt], convert_to_tensor=True, normalize_embeddings=True
        )
        print(f"ilosc kat emb: {len(self.category_embeddings)}")
        print(f"ilosc kat names: {len(self.categories_names)}")

        sims = util.cos_sim(prompt_embedding, self.category_embeddings).squeeze(0)
        sims_min = sims.min()
        sims_max = sims.max()

        sims_norm = (sims - sims_min) / (sims_max - sims_min + 1e-9)
        k = min(top_k, sims.size(0))
        top_vals, top_idx = torch.topk(sims, k=k)
        best = top_vals[0].item()

        picked = []
        for score, idx in zip(top_vals.tolist(), top_idx.tolist()):
            ok_threshold = threshold is not None and score >= threshold
            ok_margin = margin is not None and score >= best - margin
            if (threshold is None and margin is None) or (ok_threshold and ok_margin):
                picked.append((idx, score))
        if not picked:
            picked = [(int(top_idx[0]), float(top_vals[0]))]

        order = torch.argsort(sims, descending=True)
        sims_norm = sims_norm[order]
        sims = sims[order]
        return [
            (
                self.categories_names[i].removeprefix("passage: ").strip(),
                s.item(),
                s_norm.item(),
            )
            for i, s, s_norm in zip(order.tolist(), sims, sims_norm)
        ]

    def prompt_model_multi_batch(
        self,
        prompts: list[str],
        threshold: float = 0.35,
        margin: float = 0.02,
        top_k: int = 3,
        batch_size: int = 32,
    ):
        q_prompts = [f"query: {p}" for p in prompts]

        Q = self.model.encode(
            q_prompts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=batch_size,
        )  # shape: (M, d)
        S = util.cos_sim(Q, self.category_embeddings)
        print(f"ilosc kat emb: {len(self.category_embeddings)}")
        print(f"ilosc kat names: {len(self.categories_names)}")
        result = []
        for m in range(S.size(0)):
            row = S[m]
            s_min = row.min().item()
            s_max = row.max().item()

            row_norm = (row - s_min) / (s_max - s_min + 1e-9)
            k = min(top_k, row.size(0))
            top_vals, top_idx = torch.topk(row, k=k)
            best = top_vals[0].item()

            picked = []
            for score, idx in zip(top_vals.tolist(), top_idx.tolist()):
                ok_threshold = threshold is not None and score >= threshold
                ok_margin = margin is not None and score >= best - margin
                if (threshold is None and margin is None) or (
                    ok_threshold and ok_margin
                ):
                    picked.append((idx, score))
            if not picked:
                picked = [(int(top_idx[0]), float(top_vals[0]))]

            order = torch.argsort(row, descending=True)
            row_norm = row_norm[order]
            row = row[order]
            result.append(
                [
                    (
                        self.categories_names[i].removeprefix("passage: ").strip(),
                        s.item(),
                        s_norm.item(),
                    )
                    for i, s, s_norm in zip(order.tolist(), row, row_norm)
                ]
            )

        return result

    def pull_categories(self, categories: list[dict]) -> None:
        """
        Generate a list of category names without ids from dict list and format
        it for the model.
        """
        result = []
        self.categories_meta = categories
        for cat in categories:
            name = cat["name"].lower()
            result.append("passage: " + name)
            # result.append(name)
        self.categories_names = result
        self.category_embeddings = self.model.encode(
            self.categories_names, convert_to_tensor=True, normalize_embeddings=True
        )

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
