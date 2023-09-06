# Simplified benchmarking script from cleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/dqn.sh

OMP_NUM_THREADS=1 python -m src.benchmark \
    --env-ids PongNoFrameskip-v4 \
    --command "python redo_dqn.py --track True --use_infer False" \
    --num-seeds 4 \
    --workers 2