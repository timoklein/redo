# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from src.buffer import ReplayBuffer
from src.config import Config
from src.utils import set_cuda_configuration
from src.wrappers import *


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, int(env.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv1(x / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def linear_schedule(start_e: float, end_e: float, duration: float, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# TODO: How can I script these operations? Maybe torch compile if it works for 3.11?
def dqn_loss(
    q_network: QNetwork,
    target_network: QNetwork,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        # Get value estimates from the target network
        target_vals = target_network.forward(next_obs)
        # Select actions through the policy network
        policy_actions = q_network(next_obs).argmax(dim=1)
        target_max = target_vals[range(len(target_vals)), policy_actions]
        # Calculate Q-target
        td_target = rewards.flatten() + gamma * target_max * (1 - dones.flatten())

    old_val = q_network(obs).gather(1, actions).squeeze()
    return F.mse_loss(td_target, old_val), old_val


def main(cfg: Config) -> None:
    run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"

    wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        config=vars(cfg),
        name=run_name,
        monitor_gym=True,
        save_code=True,
        mode="online" if cfg.track else "disabled",
    )

    if cfg.save_model:
        evaluation_episode = 0
        wandb.define_metric("evaluation_episode")
        wandb.define_metric("eval/episodic_return", step_metric="evaluation_episode")

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.torch_deterministic)

    device = set_cuda_configuration(cfg.gpu)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.env_id, cfg.seed + i, i, cfg.capture_video, run_name) for i in range(cfg.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=cfg.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    q_network = torch.jit.script(q_network)
    target_network = torch.jit.script(target_network)

    rb = ReplayBuffer(
        cfg.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=cfg.seed)
    for global_step in range(cfg.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(cfg.start_e, cfg.end_e, cfg.exploration_fraction * cfg.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # REVIEW: Must probably be fixed for envpool
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                epi_return = info["episode"]["r"].item()
                print(f"global_step={global_step}, episodic_return={epi_return}")
                wandb.log(
                    {
                        "charts/episodic_return": epi_return,
                        "charts/episodic_length": info["episode"]["l"].item(),
                        "charts/epsilon": epsilon,
                    },
                    step=global_step,
                )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # REVIEW: Must probably be fixed for envpool
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if global_step % cfg.train_frequency == 0:
                data = rb.sample(cfg.batch_size)
                loss, old_val = dqn_loss(
                    q_network=q_network,
                    target_network=target_network,
                    obs=data.observations,
                    next_obs=data.next_observations,
                    actions=data.actions,
                    rewards=data.rewards,
                    dones=data.dones,
                    gamma=cfg.gamma,
                )

                if global_step % 100 == 0:
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    wandb.log(
                        {
                            "losses/td_loss": loss,
                            "losses/q_values": old_val.mean().item(),
                            "charts/SPS": int(global_step / (time.time() - start_time)),
                        },
                        step=global_step,
                    )

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % cfg.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        cfg.tau * q_network_param.data + (1.0 - cfg.tau) * target_network_param.data
                    )

    if cfg.save_model:
        model_path = Path(f"runs/{run_name}/{cfg.exp_name}")
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(q_network.state_dict(), model_path / ".cleanrl_model")
        print(f"model saved to {model_path}")
        from src.evaluate import evaluate

        episodic_returns = evaluate(
            model_path=model_path,
            make_env=make_env,
            env_id=cfg.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
            capture_video=False,
        )
        for episodic_return in episodic_returns:
            wandb.log({"evaluation_episode": evaluation_episode, "eval/episodic_return": episodic_return})
            evaluation_episode += 1

    envs.close()
    wandb.finish()


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=Config)
    main(cfg)
