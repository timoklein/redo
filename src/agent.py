import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Basic nature DQN agent."""

    def __init__(self, env):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.q = nn.Linear(512, int(env.single_action_space.n))

    def forward(self, x):
        x1 = F.relu(self.conv1(x / 255.0))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x3 = torch.flatten(x3, start_dim=1)
        x4 = F.relu(self.fc1(x3))
        q_vals = self.q(x4)
        return q_vals, (x1, x2, x3, x4)


def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
