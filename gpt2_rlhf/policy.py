import json
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from gpt2_rlhf.data import query_loader
from gpt2_rlhf.models import Policy, RewardModel, load_backbone, load_tokenizer
from gpt2_rlhf.utils import (
    forward_model, generate, generation_config, right_padding_to_left_padding,
    save_checkpoint, set_seed, standardize, truncate_responses,
)


def gae(rewards, values, gamma, lam):
    advantages = torch.empty_like(rewards)
    next_value = 0.0
    next_advantage = 0.0

    for t in reversed(range(rewards.shape[1])):
        td_error = rewards[:, t] + gamma * next_value - values[:, t]
        advantage = td_error + gamma * lam * next_advantage
        advantages[:, t] = advantage

        next_value = values[:, t]
        next_advantage = advantage

    return advantages


class AdaptiveKLController:
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        proportional_error = min(max(current / self.target - 1, -0.2), 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


def _token_logprobs(logits, tokens):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    indices = tokens.unsqueeze(-1)
    selected = torch.gather(log_probs, dim=2, index=indices)
    return selected.squeeze(-1)

def policy_predictions(policy, tokens, pad_id, context_length, temperature):
    logits, values = forward_model(policy, tokens, pad_id, context_length=context_length)
    # Logits are not reused: scale in place to avoid another full vocabulary tensor.
    logprobs = _token_logprobs(logits.float().div_(temperature), tokens[:, context_length:])
    return logprobs, values.float()


@torch.no_grad()
def score_rollout(ref_policy, reward_model, tokens, kept, pad_id, context_length, temperature, reward_sign=1.0):
    ref_logprobs, _ = policy_predictions(ref_policy, tokens, pad_id, context_length, temperature)
    rm_rewards = reward_model.score(tokens[:, :context_length], kept, pad_id, reward_sign=reward_sign)
    return ref_logprobs, rm_rewards


@torch.no_grad()
def generate_rollout_chunk(policy, queries, pad_id, config, args):
    context_length = queries.shape[1]
    queries = right_padding_to_left_padding(queries, pad_id)
    tokens = generate(policy.lm, queries, pad_id, config)
    logprobs, values = policy_predictions(policy, tokens, pad_id, context_length, args.temperature)
    kept, has_token = truncate_responses(
        tokens[:, context_length:], args.truncate_token, args.truncate_after, pad_id,
    )
    chunk = {
        "tokens": tokens,
        "logprobs": logprobs,
        "values": values,
        "has_token": has_token,
    }
    return chunk, kept


@torch.no_grad()
def collect_rollout(policy, ref_policy, reward_model, queries, pad_id, config, args):
    context_length = queries.shape[1]
    chunks = []
    for query_chunk in queries.split(args.rollout_batch_size):
        chunk, kept = generate_rollout_chunk(policy, query_chunk, pad_id, config, args)
        ref_logprobs, rm_rewards = score_rollout(
            ref_policy, reward_model, chunk["tokens"], kept, pad_id, context_length, args.temperature,
            reward_sign=args.reward_sign,
        )
        penalty = torch.full_like(rm_rewards, args.penalty_reward_value)
        chunk["ref_logprobs"] = ref_logprobs
        chunk["terminal_rewards"] = torch.where(
            chunk["has_token"], rm_rewards, torch.minimum(rm_rewards, penalty),
        )
        chunks.append(chunk)

    return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}


@torch.no_grad()
def rollout_targets(rollout, kl_coef, args):
    kl = rollout["logprobs"] - rollout["ref_logprobs"]  # This MSTC conserves vram and computation
    rewards = -kl_coef * kl
    rewards[:, -1] += rollout["terminal_rewards"]
    advantages = gae(rewards, rollout["values"], args.gamma, args.lam)
    rollout["returns"] = advantages + rollout["values"]
    # Normalize advantages across the complete rollout.
    rollout["advantages"] = standardize(advantages)
    return kl


def update(policy, optimizer, rollout, pad_id, context_length, args):
    """One optimizer step per minibatch, with sample-weighted gradient accumulation."""
    totals = rollout["values"].new_zeros(4)
    for _ in range(args.noptepochs):
        indices = torch.randperm(args.batch_size, device=rollout["tokens"].device)
        for minibatch in torch.tensor_split(indices, args.nminibatches):
            optimizer.zero_grad()
            for start in range(0, len(minibatch), args.micro_batch_size):
                mb = minibatch[start : start + args.micro_batch_size]
                new_logprobs, vpred = policy_predictions(
                    policy, rollout["tokens"][mb], pad_id, context_length, args.temperature
                )
                vf_loss = 0.5 * F.mse_loss(vpred, rollout["returns"][mb])
                logprob_diff = new_logprobs - rollout["logprobs"][mb]
                ratio = torch.exp(logprob_diff)
                pg_losses1 = -rollout["advantages"][mb] * ratio
                pg_losses2 = -rollout["advantages"][mb] * torch.clamp(
                    ratio, 1.0 - args.eps, 1.0 + args.eps
                )
                pg_loss = torch.max(pg_losses1, pg_losses2).mean()
                loss = pg_loss + args.vf_coef * vf_loss
                (loss * (len(mb) / len(minibatch))).backward()
                with torch.no_grad():
                    approxkl = 0.5 * logprob_diff.square().mean()
                    clipfrac = ((ratio - 1).abs() > args.eps).float().mean()
                    totals += torch.stack((pg_loss.detach(), vf_loss.detach(), approxkl, clipfrac)) * len(mb)
            optimizer.step()
    return (totals / (args.batch_size * args.noptepochs)).tolist()


def load_models(args, device):
    policy = Policy(load_backbone(args.base_model)).to(device).eval()
    policy.compile()
    ref_policy = Policy(load_backbone(args.base_model)).to(device).eval().requires_grad_(False)
    reward_model = RewardModel.from_pretrained(args.reward_path, args.base_model)
    reward_model = reward_model.to(device).requires_grad_(False)
    return policy, ref_policy, reward_model


def log_training_step(writer, step, rollout, losses, mean_kl, kl_coef, lr, args):
    pg_loss, vf_loss, approxkl, clipfrac = losses
    entropy, mean_reward = torch.stack((
        (-rollout["logprobs"]).sum(1).mean(),
        rollout["terminal_rewards"].mean(),
    )).tolist()
    metrics = {
        "reward/terminal": mean_reward,
        "policy/kl": mean_kl,
        "policy/kl_per_token": mean_kl / args.response_length,
        "policy/entropy": entropy,
        "policy/kl_coef": kl_coef,
        "policy/approxkl": approxkl,
        "policy/clipfrac": clipfrac,
        "loss/policy": pg_loss,
        "loss/value": vf_loss,
        "loss/total": pg_loss + args.vf_coef * vf_loss,
        "train/lr": lr,
    }
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda")
    tokenizer = load_tokenizer(args.base_model)
    pad_id = tokenizer.pad_token_id
    assert args.batch_size % args.nminibatches == 0, "batch_size must be divisible by nminibatches"
    policy, ref_policy, reward_model = load_models(args, device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, fused=True)
    config = generation_config(args.response_length, args.temperature)
    query_iter = iter(query_loader(tokenizer, args.query_length, args.batch_size, args.seed))
    kl_controller = AdaptiveKLController(args.kl_coef, args.kl_target, args.kl_horizon)
    num_updates = args.total_episodes // args.batch_size
    with SummaryWriter(f"{args.log_dir}/policy-{time.time_ns()}") as writer:
        writer.add_text("config", "```json\n" + json.dumps(vars(args), indent=2) + "\n```", 0)
        for step in tqdm(range(1, num_updates + 1), desc="Policy", unit="update"):
            lr = 0.5 * (1 + math.cos(math.pi * (step - 1) / num_updates)) * args.lr
            optimizer.param_groups[0]["lr"] = lr
            queries = next(query_iter).to(device)
            rollout = collect_rollout(policy, ref_policy, reward_model, queries, pad_id, config, args)
            kl = rollout_targets(rollout, kl_controller.value, args)
            losses = update(policy, optimizer, rollout, pad_id, queries.shape[1], args)
            mean_kl = kl.sum(1).mean().item()
            log_training_step(writer, step, rollout, losses, mean_kl, kl_controller.value, lr, args)
            kl_controller.update(mean_kl, args.batch_size)
    save_checkpoint(policy, tokenizer, optimizer, args.save_path)
