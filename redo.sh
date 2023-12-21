# Simplified benchmarking script from cleanRL: https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/dqn.sh

OMP_NUM_THREADS=1 python -m src.benchmark \
    --env-ids DemonAttack-v4 \
    --command "python redo_dqn.py --track --enable_redo" \
    --num-seeds 3 \
    --workers 1