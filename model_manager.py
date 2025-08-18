from sentence_transformers import SentenceTransformer, util
import torch


class ModelManager:
    def __init__(self, config: dict):
        model_name = config["model"]
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name, torch_dtype=torch.bfloat16, local_files_only=True
        # )
        self.model = SentenceTransformer(model_name)
        self.system_prompt = config["system_prompt"]
        self.messages = [config["system_prompt"]]
        self.device = config["device"]
        self.max_new_tokens = config["max_new_tokens"]
        self.categories = None

        self.model.to(self.device)

    def prompt_model(self, prompt: str):
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
        result = []
        for cat in categories:
            name = cat["name"]
            result.append("passage: " + name)
        return result
