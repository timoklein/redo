"""
Simplified version of cleanRL's benchmarking script.
The original version can be found here: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/benchmark.py
"""

import argparse
import shlex
import subprocess


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="+", default=["CartPole-v1", "Acrobot-v1", "MountainCar-v0"],
        help="the ids of the environment to benchmark")
    parser.add_argument("--command", type=str, default="python redo_dqn.py",
        help="the command to run")
    parser.add_argument("--num-seeds", type=int, default=3,
        help="the number of random seeds")
    parser.add_argument("--start-seed", type=int, default=1,
        help="the number of the starting seed")
    parser.add_argument("--workers", type=int, default=0,
        help="the number of workers to run benchmark experiments")
    args = parser.parse_args()
    # fmt: on
    return args


def run_experiment(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    fd = subprocess.Popen(command_list)
    return_code = fd.wait()
    assert return_code == 0


if __name__ == "__main__":
    args = parse_args()

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
