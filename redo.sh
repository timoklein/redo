OMP_NUM_THREADS=1 python -m src.benchmark \
    --env-ids PongNoFrameskip-v4 BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 \
    --command "python redo_dqn.py --track False" \
    --num-seeds 3 \
    --workers 1
