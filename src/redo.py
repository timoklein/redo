# TODO: Add function that checks for dormant neurons
# TODO: Add function that re-initializes a dormant neuron according to the redo specs
# TODO: Function should return the number of dormant neurons so they can be logged
# TODO: Add option to perform re-init or just check for dormant neurons

import torch


def run_redo(batch, model, tau, re_initialize) -> int:
    ...
