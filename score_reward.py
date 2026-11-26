import argparse
from pathlib import Path

import torch

from gpt2_rlhf.models import RewardModel, load_tokenizer
from gpt2_rlhf.utils import forward_model


def load_model(path="checkpoints/reward", base_model="models/gpt2"):
    tokenizer = load_tokenizer(base_model if Path(path).is_file() else path)
    model = RewardModel.from_pretrained(path, base_model)
    return model.to("cuda").requires_grad_(False), tokenizer


@torch.no_grad()
def score_text(model, tokenizer, text, reward_sign=1.0):
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    reward = forward_model(model, tokens, tokenizer.pad_token_id, reward_sign=reward_sign)[1]
    return reward.float().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Complete text to score, including the prompt if applicable.")
    parser.add_argument("--model", default="checkpoints/reward", help="Reward checkpoint directory or legacy .pt file.")
    parser.add_argument("--base-model", default="models/gpt2", help="Base model and tokenizer for a .pt checkpoint.")
    parser.add_argument("--reward-sign", type=float, default=1.0, help="Multiplier for the RM output.")
    args = parser.parse_args()
    model, tokenizer = load_model(args.model, args.base_model)
    print(score_text(model, tokenizer, args.text, args.reward_sign))


if __name__ == "__main__":
    main()
