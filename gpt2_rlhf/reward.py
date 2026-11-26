import json
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from gpt2_rlhf.data import load_json, preference_batch, query_loader
from gpt2_rlhf.models import RewardModel, load_backbone, load_tokenizer
from gpt2_rlhf.utils import (
    forward_model, generate, generation_config, right_padding_to_left_padding,
    save_checkpoint, set_seed,
)


@torch.no_grad()
def _sample_rewards(reward_model, ref_lm, queries, tokenizer, config):
    pad_id = tokenizer.pad_token_id
    queries = queries.to(reward_model.reward_gain.device)
    queries = right_padding_to_left_padding(queries, pad_id)
    query_responses = generate(ref_lm, queries, pad_id, config)
    reward = forward_model(reward_model, query_responses, pad_id)[1]
    return reward.float().squeeze(-1)


@torch.no_grad()
def normalize(reward_model, ref_lm, query_iter, tokenizer, config, n_samples):
    reward_model.eval()
    reward_model.reward_gain.data.fill_(1.0)
    reward_model.reward_bias.data.fill_(0.0)
    rewards = _collect(reward_model, ref_lm, query_iter, tokenizer, config, n_samples)
    mean, std = rewards.mean(), rewards.std()
    gain = 1.0 / std
    bias = -gain * mean
    reward_model.reward_gain.data.copy_(gain)
    reward_model.reward_bias.data.copy_(bias)


def _collect(reward_model, ref_lm, query_iter, tokenizer, config, n_samples):
    chunks = []
    seen = 0
    while seen < n_samples:
        queries = next(query_iter)
        reward = _sample_rewards(reward_model, ref_lm, queries, tokenizer, config)
        chunks.append(reward)
        seen += reward.shape[0]
    return torch.cat(chunks)[:n_samples]


def preference_loss(reward_model, query, responses, best, pad_id):
    # Vectorize the four candidates while preserving [query, candidate] order.
    responses = torch.stack(responses, dim=1)
    queries = query[:, None, :].expand(-1, responses.shape[1], -1)
    tokens = torch.cat((queries, responses), dim=2).flatten(0, 1)
    tokens = right_padding_to_left_padding(tokens, pad_id)
    predicted = forward_model(reward_model, tokens, pad_id)[1].float().view(query.shape[0], -1)
    loss = torch.nn.functional.cross_entropy(predicted, best)
    accuracy = (predicted.argmax(1) == best).float().mean()
    return loss, accuracy


def reward_update(reward_model, optimizer, label, indices, tokenizer, args):
    optimizer.zero_grad()
    query, responses, best = preference_batch(
        label, indices, args.num_labels, tokenizer.pad_token_id, "cuda",
    )
    loss, accuracy = preference_loss(reward_model, query, responses, best, tokenizer.pad_token_id)
    loss.backward()
    optimizer.step()
    return torch.stack((loss.detach(), accuracy.detach())).tolist()


def train(args):
    set_seed(args.seed)
    label = load_json(args.label_path)
    num_train = len(label)
    device = torch.device("cuda")
    tokenizer = load_tokenizer(args.base_model)
    reward_model = RewardModel(load_backbone(args.base_model)).to(device).eval()
    reward_model.compile()
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=args.lr, fused=True)
    # These indices select records from the CPU-side JSON list.
    order = torch.randperm(num_train, device="cpu")
    with SummaryWriter(f"{args.log_dir}/reward-{time.time_ns()}") as writer:
        writer.add_text("config", "```json\n" + json.dumps(vars(args), indent=2) + "\n```", 0)
        batches = tqdm(range(0, num_train, args.batch_size), desc="Reward", unit="batch")
        for step, start in enumerate(batches, start=1):
            lr = (1 - start / num_train) * args.lr
            optimizer.param_groups[0]["lr"] = lr
            indices = order[start : start + args.batch_size].tolist()
            loss, accuracy = reward_update(reward_model, optimizer, label, indices, tokenizer, args)
            metrics = {"loss/reward": loss, "reward/accuracy": accuracy, "train/lr": lr}
            for name, value in metrics.items():
                writer.add_scalar(name, value, step)

    ref_lm = load_backbone(args.base_model).to(device).eval().requires_grad_(False)
    config = generation_config(args.response_length, args.temperature)
    query_iter = iter(
        query_loader(tokenizer, args.query_length, args.normalize_batch_size, args.seed)
    )
    normalize(reward_model, ref_lm, query_iter, tokenizer, config, args.normalize_samples)

    save_checkpoint(reward_model, tokenizer, optimizer, args.save_path)
