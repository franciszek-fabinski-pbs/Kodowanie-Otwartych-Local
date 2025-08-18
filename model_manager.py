from sentence_transformers import SentenceTransformer, util
import torch


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
        model_name = config["model"]
        self.model = SentenceTransformer(model_name)
        self.device = config["device"]
        self.categories = None

        self.model.to(self.device)

    def prompt_model(self, prompt: str) -> int:
        """
        Prompt model with an answer/sentence, no special formatting needed.

        Returns index of the most fitting category.
        """
        prompt = "query: " + prompt
        prompt_embedding = self.model.encode(
            [prompt], convert_to_tensor=True, normalize_embeddings=True
        )
        category_embeddings = self.model.encode(
            self.categories, convert_to_tensor=True, normalize_embeddings=True
        )

        sims = util.cos_sim(prompt_embedding, category_embeddings).squeeze(0)
        print(sims)
        best_idx = int(torch.argmax(sims))

        return best_idx

    @staticmethod
    def generate_categories(categories: list[dict]) -> list[str]:
        """
        Generate a list of category names without ids from dict list and format
        it for the model.
        """
        result = []
        for cat in categories:
            name = cat["name"]
            result.append("passage: " + name)
        return result
