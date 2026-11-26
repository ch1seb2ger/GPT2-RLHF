# GPT2-RLHF

## Setup

Run this from the project root to install the dependencies:

```bash
uv sync
```

## Download the model and training data

After `uv sync`, run these commands from the project root. They use the project's Hugging Face CLI and save files to the default paths used by training.

```bash
# GPT-2 weights, configuration, and tokenizer
uv run hf download openai-community/gpt2 \
  config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt \
  --local-dir models/gpt2

# Sentiment preference labels for reward model training
uv run hf download vwxyzjn/lm-human-preferences sentiment/offline_5k.json \
  --repo-type dataset --local-dir datasets/preferences

# BookCorpus for reward normalization and PPO sampling (10 Parquet shards, about 3.02 GB)
uv run hf download bookcorpus/bookcorpus \
  --repo-type dataset --revision refs/convert/parquet \
  --include 'plain_text/train/*.parquet' --local-dir datasets/books
```

## Training

After downloading the model and data, run these commands from the project root on a machine with a CUDA-capable GPU:

```bash
uv run python -m gpt2_rlhf.train reward && \
uv run python -m gpt2_rlhf.train policy
```

The first command trains the reward model and saves it to `checkpoints/reward/`. If it succeeds, the second command loads that checkpoint, trains the policy with PPO, and saves it to `checkpoints/policy/`.

## Score and generate text

After training, use the reward model to score a sentence and the policy to continue a prompt:

```bash
uv run python score_reward.py "When the earthquake stopped, he crawled through the rubble, dust stinging his eyes. Then he heard his sister call his name."
uv run python generate_text.py "When the earthquake stopped, he crawled through the rubble, dust stinging his eyes. Then he heard"
```

## Reward sign

`--reward-sign` multiplies the reward model's output. The default, `1`, keeps the score unchanged. Use `-1` to negate it, which reverses the ranking of unequal scores without changing the reward checkpoint.

```bash
uv run python score_reward.py --reward-sign -1 \
  "When the earthquake stopped, he crawled through the rubble, dust stinging his eyes. Then he heard his sister call his name."
```

PPO accepts the same flag. It flips the reward model score before the missing-period penalty and does not flip the KL penalty. This example saves the new policy separately.

```bash
uv run python -m gpt2_rlhf.train policy --reward-sign -1. --save-path checkpoints/policy-negated
```
