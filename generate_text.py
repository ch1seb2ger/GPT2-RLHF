import argparse
from pathlib import Path

import torch

from gpt2_rlhf.models import Policy, load_backbone, load_tokenizer
from gpt2_rlhf.utils import generate, generation_config


def load_model(path, base_model="models/gpt2"):
    legacy = Path(path).is_file()
    tokenizer = load_tokenizer(base_model if legacy else path)
    model = Policy.from_pretrained(path, base_model).lm if legacy else load_backbone(path)
    return model.to("cuda").eval().requires_grad_(False), tokenizer


@torch.no_grad()
def continue_text(model, tokenizer, text, response_length=128, temperature=0.7):
    queries = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    context_length = queries.shape[1]
    config = generation_config(response_length, temperature)
    tokens = generate(model, queries, tokenizer.pad_token_id, config)
    response = tokenizer.decode(tokens[0, context_length:].tolist(), skip_special_tokens=True)
    return text + response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Text to continue.")
    parser.add_argument("--model", default="checkpoints/policy", help="Local model directory or legacy policy .pt file.")
    parser.add_argument("--base-model", default="models/gpt2", help="Base model and tokenizer for a .pt checkpoint.")
    parser.add_argument("--response-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    model, tokenizer = load_model(args.model, args.base_model)
    print(continue_text(model, tokenizer, args.text, args.response_length, args.temperature))


if __name__ == "__main__":
    main()
