import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="Train a reward model or PPO policy from local files.")
    commands = parser.add_subparsers(dest="model", required=True)
    reward = commands.add_parser("reward", help="Train the reward model on all preference records.")
    policy = commands.add_parser("policy", help="Train a PPO policy using the local reward checkpoint.")

    for command in (reward, policy):
        command.add_argument("--base-model", default="models/gpt2", help="Local pretrained model directory.")
        command.add_argument("--query-length", type=int, default=64)
        command.add_argument("--response-length", type=int, default=128)
        command.add_argument("--temperature", type=float, default=0.7)
        command.add_argument("--seed", type=int, default=1)
        command.add_argument("--log-dir", default="logs", help="Directory for TensorBoard runs.")

    reward.add_argument("--label-path", default="datasets/preferences/sentiment/offline_5k.json")
    reward.add_argument("--batch-size", type=int, default=32, help="Preference queries per optimizer step.")
    reward.add_argument("--lr", type=float, default=5e-5)
    reward.add_argument("--num-labels", type=int, default=4, help="Candidates per preference query.")
    reward.add_argument("--normalize-samples", type=int, default=2048)
    reward.add_argument("--normalize-batch-size", type=int, default=512)
    reward.add_argument("--save-path", default="checkpoints/reward", help="Output checkpoint directory.")

    policy.add_argument("--reward-path", default="checkpoints/reward", help="Reward checkpoint directory or legacy .pt file.")
    policy.add_argument("--truncate-token", type=int, default=13)
    policy.add_argument("--truncate-after", type=int, default=16)
    policy.add_argument("--penalty-reward-value", type=float, default=-1.0)
    policy.add_argument("--batch-size", type=int, default=512, help="Sequences per rollout.")
    policy.add_argument("--rollout-batch-size", type=int, default=64)
    policy.add_argument("--micro-batch-size", type=int, default=32)
    policy.add_argument("--nminibatches", type=int, default=1)
    policy.add_argument("--noptepochs", type=int, default=4)
    policy.add_argument("--lr", type=float, default=1e-5)
    policy.add_argument("--vf-coef", type=float, default=0.1)
    policy.add_argument("--eps", type=float, default=0.2, help="PPO probability-ratio clipping range.")
    policy.add_argument("--gamma", type=float, default=1.0)
    policy.add_argument("--lam", type=float, default=0.95)
    policy.add_argument("--kl-coef", type=float, default=0.15)
    policy.add_argument("--reward-sign", "--score-sign", type=float, default=1.0,
                        help="Multiplier for the RM output before applying the missing-period penalty.")
    policy.add_argument("--kl-target", type=float, default=6.0)
    policy.add_argument("--kl-horizon", type=int, default=10000)
    policy.add_argument("--total-episodes", type=int, default=1_000_000)
    policy.add_argument("--save-path", default="checkpoints/policy", help="Output checkpoint directory.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.model == "reward":
        from gpt2_rlhf.reward import train
    else:
        from gpt2_rlhf.policy import train
    train(args)


if __name__ == "__main__":
    main()
