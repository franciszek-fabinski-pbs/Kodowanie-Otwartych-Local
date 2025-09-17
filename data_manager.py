from typing import Callable

import numpy as np
import torch

from data_types import Category


class CategoryManager:
    def __init__(
        self,
        categories: list[dict] = [],
    ):
        self.categories: list[Category] = [
            Category(name=c["name"], id=c["id"]) for c in categories
        ]
        self.categories_encoded: torch.Tensor = None
        self.cat_names = [c.name for c in self.categories]
        self.id_idx_map = {c["id"]: i for i, c in enumerate(categories)}
        self.sims = None

    def update_categories(
        self, categories: list[dict], encoder: Callable | None = None
    ):
        self.__init__(categories)
        if encoder is not None:
            self.encode(encoder)

    def encode(self, encoder: Callable):
        self.categories_encoded = encoder(
            [c.name for c in self.categories], prefix="passage: "
        )

    def get_by_id(self, id: list[int]) -> list[Category]:
        arr = np.asarray(id, dtype=float)
        idxs = arr[~np.isnan(arr)].astype(int).tolist()
        return [self.categories[self.id_idx_map[str(i)]] for i in idxs]

    def get_index_by_id(self, id: str) -> Category:
        if type(id) is not str:
            id = str(id)
        return self.id_idx_map[id]

    def get_cat_sim(self):
        return self.sim

    def set_cat_sim(self, sims: torch.Tensor):
        self.sim = sims
        return
