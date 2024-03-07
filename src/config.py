from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "ReDo DQN"
    tags: tuple[str, ...] | str | None = ("fixed_nonzero_bias",)
    seed: int = 0
    torch_deterministic: bool = False
    gpu: int | None = 0
    track: bool = False
    wandb_project_name: str = "ReDo"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = False

    # Environment settings
    env_id: str = "DemonAttack-v4"
    total_timesteps: int = 10_000_000
    num_envs: int = 1

    # DQN settings
    buffer_size: int = 1_000_000
    batch_size: int = 32
    learning_rate: float = 6.25 * 1e-5  # cleanRL default: 1e-4, theirs: 6.25 * 1e-5
    adam_eps: float = 1.5 * 1e-4
    use_lecun_init: bool = False  # ReDO uses lecun_normal initializer, cleanRL uses the pytorch default (kaiming_uniform)
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000  # cleanRL default: 8000, 4 freq -> 8000, 0.5 freq -> 1000, 1 -> 2000
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 20_000  # cleanRL default: 80000, theirs 20000
    train_frequency: int = 0.5  # cleanRL default: 4, theirs 1

    # ReDo settings
    enable_redo: bool = False
    redo_tau: float = 0.1  # 0.025 for default, else 0.1
    redo_check_interval: int = 1000
    redo_bs: int = 64
