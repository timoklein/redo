from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "ReDo DQN"
    seed: int = 0
    torch_deterministic: bool = False
    gpu: Optional[int] = 1
    track: bool = False
    wandb_project_name: str = "Atari_feature_regularizers"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = False

    # Environment settings
    env_id: str = "DemonAttack-v4"
    total_timesteps: int = 6000000
    num_envs: int = 1

    # DQN settings
    buffer_size: int = 1000000
    batch_size: int = 32
    learning_rate: float = 6.25 * 1e-5  # cleanRL default: 1e-4
    adam_eps: float = 1.5 * 1e-4
    use_lecun_init: bool = False  # ReDO uses lecun_normal initializer, cleanRL uses the pytorch default (kaiming_uniform)
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000  # cleanRL default: 8000
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 20000  # cleanRL default: 80000
    train_frequency: int = 0.25  # cleanRL default: 4

    # ReDo settings
    enable_redo: bool = True
    redo_tau: float = 0.1
    redo_check_interval: int = 1000
    redo_bs: int = 64
