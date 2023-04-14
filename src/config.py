from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "ReDo DQN"
    seed: int = 1
    torch_deterministic: bool = False
    gpu: Optional[int] = 0
    track: bool = False
    wandb_project_name: str = "Atari_feature_regularizers"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = False

    # Environment settings
    env_id: str = "PongNoFrameskip-v4"
    total_timesteps: int = 10000000
    num_envs: int = 1

    # DQN settings
    buffer_size: int = 1000000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 32
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 800
    train_frequency: int = 4

    # ReDo settings
    enable_redo: bool = False
    redo_tau: float = 0.1
    redo_check_interval: int = 10000
