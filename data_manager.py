import numpy as np
import torch

from data_types import Category


class CategoryManager:
    def __init__(
        self,
        categories: list[dict] = [],
    ):
        self.categories: list[Category] = [
            Category(name=c["name"], id=int(c["id"]), keywords=c["keywords"])
            for c in categories
        ]
        self.cat_names = [c.name for c in self.categories]
        self.id_idx_map = {int(c["id"]): i for i, c in enumerate(categories)}
        self.classification_counter = {c["id"]: 0 for c in categories}
        self.sims = None

    def update_categories(
        self, categories: list[dict]
    ):
        self.__init__(categories)

    def get_by_id(self, id: list[int] | int) -> list[Category]:
        arr = np.asarray(id, dtype=float)
        idxs = arr[~np.isnan(arr)].astype(int).tolist()
        return [self.categories[self.id_idx_map[i]] for i in idxs]

    def get_index_by_id(self, id: str) -> Category:
        if type(id) is not str:
            id = id
        return self.id_idx_map[id]

    def get_cat_sim(self):
        return self.sim

    def set_cat_sim(self, sims: torch.Tensor):
        self.sim = sims
        return
