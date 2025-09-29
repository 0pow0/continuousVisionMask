import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal


def build_mlp_network(sizes):
    """
    Build a multi-layer perceptron (MLP) neural network.

    This function constructs an MLP network with the specified layer sizes and activation functions.

    Args:
        sizes (list of int): List of integers representing the sizes of each layer in the network.

    Returns:
        nn.Sequential: An instance of PyTorch's Sequential module representing the constructed MLP.
    """
    layers = list()
    for j in range(len(sizes) - 1):
        act = nn.Tanh if j < len(sizes) - 2 else nn.Identity
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        nn.init.kaiming_uniform_(affine_layer.weight, a=np.sqrt(5))
        layers += [affine_layer, act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Actor network for policy-based reinforcement learning.

    This class represents an actor network that outputs a distribution over actions given observations.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        act_dim (int): Dimensionality of the action space.

    Attributes:
        mean (nn.Sequential): MLP network representing the mean of the action distribution.
        log_std (nn.Parameter): Learnable parameter representing the log standard deviation of the action distribution.

    Example:
        obs_dim = 10
        act_dim = 2
        actor = Actor(obs_dim, act_dim)
        observation = torch.randn(1, obs_dim)
        action_distribution = actor(observation)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list = [64, 64]):
        super().__init__()
        self.mean = build_mlp_network([obs_dim] + hidden_sizes + [act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=True)

    def forward(self, obs: torch.Tensor):
        mean = self.mean(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)


def main():
    # ckpt = "/home/rzuo02/work/Safe-Policy-Optimization/runs/single_agent_exp/SafetyPointGoal1-v0/ppo/seed-000-2025-09-26-16-59-37/torch_save/model499.pt"
    # print(f"Loading checkpoint: {ckpt}")
    # checkpoint = torch.load(ckpt, map_location="cpu")
    # actor = Actor(obs_dim=60, act_dim=2)
    # actor.load_state_dict(checkpoint)
    # print(actor)
    obs_dim = 10
    act_dim = 2
    actor = Actor(obs_dim, act_dim)
    observation = torch.randn(1, obs_dim)
    action_distribution = actor(observation)
    print(action_distribution)
    print(action_distribution.sample())
    print(action_distribution.rsample())


if __name__ == "__main__":
    main()
