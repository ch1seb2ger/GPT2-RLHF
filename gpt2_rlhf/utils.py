import random
from pathlib import Path

import torch
from transformers import GenerationConfig


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, tokenizer, optimizer, path):
    path = Path(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")


def init_linear(layer, std):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.zeros_(layer.bias)
    return layer


def right_padding_to_left_padding(tokens, pad_id):
    """Move pad tokens left while preserving the order of all other tokens."""
    order = (tokens != pad_id).long().argsort(dim=1, stable=True)
    return tokens.gather(1, order)


def mask_tokens(token_ids, pad_id):
    attention_mask = (token_ids != pad_id).long()
    position_ids = attention_mask.cumsum(1) - attention_mask
    input_ids = token_ids.masked_fill(attention_mask == 0, 0)
    return input_ids, attention_mask, position_ids


def forward_model(model, token_ids, pad_id, **kwargs):
    input_ids, attention_mask, position_ids = mask_tokens(token_ids, pad_id)
    return model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs)


def generation_config(response_length, temperature):
    return GenerationConfig(
        max_new_tokens=response_length,
        min_new_tokens=response_length,
        temperature=temperature,
        top_k=0,
        top_p=1.0,
        do_sample=True,
        eos_token_id=None,
        pad_token_id=None,
        return_dict_in_generate=True,
    )


def generate(backbone, queries, pad_id, config):
    context_length = queries.shape[1]
    input_ids, attention_mask, _ = mask_tokens(queries, pad_id)
    output = backbone.generate(
        input_ids=input_ids, attention_mask=attention_mask, generation_config=config,
    )
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


def standardize(values):
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    return (values - mean) * torch.rsqrt(var + 1e-8)


def truncate_responses(responses, truncate_token, truncate_after, pad_id):
    after = (responses == truncate_token).long()
    after[:, :truncate_after] = 0
    drop = (torch.cumsum(after, dim=1) - after).bool()
    kept = responses.masked_fill(drop, pad_id)
    has_token = (kept[:, truncate_after:] == truncate_token).any(dim=-1)
    return kept, has_token
