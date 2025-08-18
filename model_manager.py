from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    attention_mask = attention_mask.to(token_embeddings.device)
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class ModelManager:
    def __init__(self, config: dict):
        model_name = config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True
        )
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name, torch_dtype=torch.bfloat16, local_files_only=True
        # )
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, local_files_only=True
        )
        self.system_prompt = config["system_prompt"]
        self.messages = [config["system_prompt"]]
        self.device = config["device"]
        self.max_new_tokens = config["max_new_tokens"]
        self.categories = None

        self.model.to(self.device)

    def prompt_model(self, prompt: str):
        all_sentences = self.categories.copy()
        all_sentences.append(prompt)
        print(
            f"typ kategorii: {type(self.categories)}, typ prompta: {type(prompt)}, typ all_categories: {type(all_sentences)}"
        )
        encoded_input = self.tokenizer(
            all_sentences, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        return sentence_embeddings

    @staticmethod
    def generate_categories(categories: list[dict]) -> list[str]:
        result = []
        for cat in categories:
            name = cat["name"]
            result.append(name)
        return result
