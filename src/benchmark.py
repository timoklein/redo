"""
Simplified version of cleanRL's benchmarking script.
The original version can be found here: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/benchmark.py
"""

from dataclasses import dataclass
import tyro
import shlex
import subprocess

@dataclass
class BenchmarkConfig:
    env_ids: tuple[str] = ("CartPole-v1", "Acrobot-v1", "MountainCar-v0")
    command: str = "python redo_dqn.py"
    start_seed: int = 1
    num_seeds: int = 3
    workers: int = 3


def run_experiment(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    fd = subprocess.Popen(command_list)
    return_code = fd.wait()
    assert return_code == 0


if __name__ == "__main__":
    args = tyro.cli(BenchmarkConfig)

    commands = []
    for seed in range(0, args.num_seeds):
        for env_id in args.env_ids:
            commands += [" ".join([args.command, "--env_id", env_id, "--seed", str(args.start_seed + seed)])]

    print("======= commands to run:")
    for command in commands:
        print(command)

    if args.workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="cleanrl-benchmark-worker-")
        for command in commands:
            executor.submit(run_experiment, command)
        executor.shutdown(wait=True)
    else:
        print("not running the experiments because --workers is set to 0; just printing the commands to run")
