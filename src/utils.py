import typing

import torch


def set_cuda_configuration(gpu: typing.Any) -> torch.device:
    """Set up the device for the desired GPU or all GPUs."""
    if gpu is None or gpu == -1 or gpu is False:
        device = torch.device("cpu")
    elif isinstance(gpu, int):
        assert gpu <= torch.cuda.device_count(), "Invalid CUDA index specified."
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cuda")

    return device
