import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from sentence_transformers import util
from typing import Literal

from data_types import Category


_MODEL_TYPE = Literal["Sentence Transformer", "ReRanker"]


class ModelManager:
    """
    Classification Model Manager for loading models and accepting prompts.
    """

    def __init__(self, config: dict):
        """
        config: dictionary with properties:
            device: torch.device, "cuda"/"cpu" etc
            model: model directory path
            model_type: type of loaded model (currently Sentence Transformer or
                                              ReRanker)
            categories: array of dicts:
                id: id of the category
                name: name of the category
        """
        model_name: str = config["model"]
        self.model: SentenceTransformer | CrossEncoder | None = None
        self.device: torch.device | str = config["device"]
        self.model_type: _MODEL_TYPE = config["model_type"]
        match self.model_type:
            case "Sentence Transformer":
                self.model = SentenceTransformerManager(
                    model_name,
                    device=self.device if self.device is not None else "auto",
                )
            case "ReRanker":
                self.model = ReRankerManager(
                    model_name,
                    device=self.device if self.device is not None else "auto",
                )
            case _:
                raise Exception("Unknown model type!")
        self.categories: list[Category] = None

    def get_results(self):
        return self._sim_results

    def pull_categories(self, categories: list[Category]) -> None:
        match self.model_type:
            case "Sentence Transformer":
                self.model.pull_categories(categories, prefix="passage: ")
            case "ReRanker":
                self.model.pull_categories(categories)
            case _:
                raise Exception("Unknown model type!")

    def classify(self, answers: list[str]) -> list[list[tuple[int, float]]]:
        """
        Classify a series of answers.
        Returns a matrix of (category.id, similiarity) tuples.
        Return structure: result[answer_index][category_index][id, similiarity]
        """
        result = self.model.classify(answers)
        return result


class SentenceTransformerManager:
    def __init__(
        self,
        model_name: str,
        device: torch.device | str,
        categories: list[Category] | None = None,
    ):
        self._prompt_embeddings: torch.Tensor = None

        if categories is not None:
            self.pull_categories(categories)
        else:
            self.categories = None
            self._category_embeddings = None

        self._device = device
        self.model = SentenceTransformer(model_name, device=self._device)
        self.id_idx_map = None

    def pull_categories(
        self,
        data: list[Category],
        prefix: str | None = None,
        batch_size: int = 32,
        id_to_index: dict = {},
    ) -> None:
        cat_names = None
        if prefix is not None:
            cat_names = [prefix + d.name for d in data]
        else:
            cat_names = [d.name for d in data]
        self.categories = data
        self.id_idx_map = {int(c.id): i for i, c in enumerate(data)}
        self._category_embeddings = self.model.encode(
            cat_names,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=batch_size,
        )

    def classify(
        self,
        prompts: list[str],
        categories: list[Category] | None = None,
        threshold: float | None = 0.35,
        margin: float | None = 0.02,
        top_k: int | None = None,
        batch_size: int = 32,
        min_similiarity: float | None = 0.857,
        min_local_similiarity: float | None = 0.85,
        show_all_sims: bool = False,
        intro: str = "",
        return_top_n: int | None = None,
    ) -> list[list[tuple[int, float]]]:
        """
        Classify multiple answers to multiple categories.
        Returns a matrix of (index, similiarity) tuples (Tensor).
        """
        q_prompts = [f"query: {intro} {p}" for p in prompts]

        Q = self.model.encode(
            q_prompts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=batch_size,
        )  # shape: (M, d)
        self.prompt_embeddings = Q

        if categories is None and self._category_embeddings is None:
            raise Exception(
                "No categories to use. To use cached categories, pull them beforehand"
            )
        elif categories is not None:
            self.pull_categories(categories, prefix="passage: ")

        S = util.cos_sim(Q, self._category_embeddings)
        result = []
        for m in range(S.size(0)):
            row = S[m]
            s_min = row.min().item()
            s_max = row.max().item()

            row_norm = (row - s_min) / (s_max - s_min + 1e-9)
            # Sort all categories by score descending
            ids = torch.tensor([c.id for c in self.categories], device=row.device)

            vals, order = torch.sort(row, descending=True)

            top_idx = ids[order]
            top_vals = vals
            top_norm = row_norm[order]
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
                if min_local_similiarity is not None and score_n < float(
                    min_local_similiarity
                ):
                    continue
                picked.append((idx, score))

            # If nothing matched constraints, fall back to top-1
            if not picked:
                picked = torch.Tensor([(int(top_idx[0]), float(top_vals[0]))])

            # If explicit top_k is requested, cap the number of picks after filtering
            if top_k is not None:
                k = max(0, min(int(top_k), len(picked)))
                picked = (
                    picked[:k] if k > 0 else [(int(top_idx[0]), float(top_vals[0]))]
                )

            if show_all_sims:
                order = torch.argsort(row, descending=True)
            else:
                order = [
                    torch.as_tensor(res[0], device=self._device)
                    for res in sorted(picked, key=lambda pick: pick[1], reverse=True)
                ]
            o = np.array([v.cpu().item() for v in order])

            sel_idx = [self.id_idx_map[cid] for cid in o]
            row = row[sel_idx]
            row_list = [
                (
                    i,
                    s.item(),
                )
                for i, s in zip(o, row)
            ]

            if return_top_n is not None:
                row_list = row_list[: max(0, min(return_top_n, len(row_list)))]

            result.append(row_list)

        self._sim_results = result
        return result


class ReRankerManager:
    def __init__(
        self,
        model_name: str,
        device: torch.device | str,
        categories: list[Category] | None = None,
    ):
        self.categories = categories
        self._device = device
        self.model = CrossEncoder(
            model_name,
            default_activation_function=torch.nn.Identity(),
            max_length=512,
            device=self._device,
        )

    def pull_categories(self, categories: list[Category]) -> None:
        self.categories = categories

    def classify_single(self, answer: str) -> list[tuple[int, float]]:
        """
        Classify a single answer to existing categories
        Returns a list of (id, similiarity) tuples
        """
        sim = self.model.predict([[answer, cat.name] for cat in self.categories])
        result = sorted(
            zip([c.id for c in self.categories], sim), key=lambda d: d[1], reverse=True
        )
        return result

    def classify(self, answers: list[str]) -> list[list[tuple[int, float]]]:
        result = []
        for ans in answers:
            result.append(self.classify_single(ans))
        return result
