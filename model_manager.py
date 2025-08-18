import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelManager:
    def __init__(self, config: dict):
        model_name = config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, local_files_only=True
        )
        self.system_prompt = config["system_prompt"]
        self.messages = [config["system_prompt"]]
        self.device = config["device"]
        self.max_new_tokens = config["max_new_tokens"]

        self.model.to(self.device)

    def prompt_model(self, prompt: str):
        self.messages.append({"role": "user", "content": prompt})
        input_ids = self.tokenizer.apply_chat_template(
            self.messages, return_tensors="pt", add_generation_prompt=True
        )
        attention_mask = torch.ones_like(input_ids)
        model_inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=self.max_new_tokens, do_sample=True
        )
        new_tokens = generated_ids[:, input_ids.shape[1] :]
        decoded = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": decoded})
        print(decoded)

    def generate_system_prompt(self, categories: list[dict]) -> None:
        for cat in categories:
            id = cat["id"]
            name = cat["name"]
            self.system_prompt["content"] += f"\nid: {id}, nazwa: {name}"
        print(self.system_prompt["content"])
