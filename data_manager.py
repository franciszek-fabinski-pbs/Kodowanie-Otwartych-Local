import torch
from typing import Callable
import numpy as np

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
        # idxs = [int(i) for i in id if i != float("nan")]
        arr = np.asarray(id, dtype=float)  # wymusi NaN dla braków
        idxs = arr[~np.isnan(arr)].astype(int).tolist()

        # for item in idxs:
        #     print(f"{item}-{type(item)}")

        return [self.categories[self.id_idx_map[i]] for i in idxs]
