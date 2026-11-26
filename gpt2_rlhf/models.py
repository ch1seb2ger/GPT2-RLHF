import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2_rlhf.utils import forward_model, init_linear, right_padding_to_left_padding


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side="right", local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def load_backbone(path):
    torch.set_float32_matmul_precision("high")
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = None
    return model


class RLHFModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.lm = backbone

    def save_pretrained(self, path):
        path = Path(path)
        self.lm.save_pretrained(path)
        heads = {
            name: tensor.cpu().contiguous()
            for name, tensor in self.state_dict().items()
            if not name.startswith("lm.")
        }
        save_file(heads, path / "rlhf_heads.safetensors")

    @classmethod
    def from_pretrained(cls, path, base_model="models/gpt2"):
        path = Path(path)
        if path.is_file():
            model = cls(load_backbone(base_model))
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        else:
            heads = load_file(path / "rlhf_heads.safetensors")
            model = cls(load_backbone(path))
            state = model.state_dict()
            expected = {name for name in state if not name.startswith("lm.")}
            state.update(heads)
            model.load_state_dict(state)
        return model.eval()


class RewardModel(RLHFModel):
    def __init__(self, backbone):
        super().__init__(backbone)
        hidden = backbone.config.hidden_size
        self.scalar_head = init_linear(nn.Linear(hidden, 1), std=1 / math.sqrt(hidden + 1))
        self.reward_gain = nn.Parameter(torch.tensor(1.0))
        self.reward_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids, attention_mask, position_ids, reward_sign=1.0):
        output = self.lm.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        last = output.last_hidden_state[:, -1, :]
        reward = self.reward_gain * self.scalar_head(last) + self.reward_bias
        return output, reward * reward_sign

    @torch.no_grad()
    def score(self, queries, responses, pad_id, reward_sign=1.0):
        tokens = torch.cat((queries, responses), dim=1)
        tokens = right_padding_to_left_padding(tokens, pad_id)
        return forward_model(self, tokens, pad_id, reward_sign=reward_sign)[1].float().flatten()


class Policy(RLHFModel):
    def __init__(self, backbone):
        super().__init__(backbone)
        hidden = backbone.config.hidden_size
        self.value_head = init_linear(nn.Linear(hidden, 1), std=0.0)

    def forward(self, input_ids, attention_mask, position_ids, context_length):
        output = self.lm.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden = output.last_hidden_state[:, context_length - 1 : -1]
        logits = self.lm.get_output_embeddings()(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values
