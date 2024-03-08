import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .agent import QNetwork
from .buffer import ReplayBufferSamples


@torch.no_grad()
def _kaiming_uniform_reinit(layer: nn.Linear | nn.Conv2d, mask: torch.Tensor) -> None:
    """Partially re-initializes the bias of a layer according to the Kaiming uniform scheme."""

    # This is adapted from https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
    fan_in = nn.init._calculate_correct_fan(tensor=layer.weight, mode="fan_in")
    gain = nn.init.calculate_gain(nonlinearity="relu", param=math.sqrt(5))
    std = gain / math.sqrt(fan_in)
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0) * std
    layer.weight.data[mask, ...] = torch.empty_like(layer.weight.data[mask, ...]).uniform_(-bound, bound)

    if layer.bias is not None:
        # NOTE: The original code resets the bias to 0.0
        # layer.bias.data[mask] = 0.0
        if isinstance(layer, nn.Conv2d):
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                layer.bias.data[mask, ...] = torch.empty_like(layer.bias.data[mask, ...]).uniform_(-bound, bound)
        else:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            layer.bias.data[mask, ...] = torch.empty_like(layer.bias.data[mask, ...]).uniform_(-bound, bound)


@torch.no_grad()
def _lecun_normal_reinit(layer: nn.Linear | nn.Conv2d, mask: torch.Tensor) -> None:
    """Partially re-initializes the bias of a layer according to the Lecun normal scheme."""

    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)

    # This implementation follows the jax one
    # https://github.com/google/jax/blob/366a16f8ba59fe1ab59acede7efd160174134e01/jax/_src/nn/initializers.py#L260
    variance = 1.0 / fan_in
    stddev = math.sqrt(variance) / 0.87962566103423978
    layer.weight[mask] = nn.init._no_grad_trunc_normal_(layer.weight[mask], mean=0.0, std=1.0, a=-2.0, b=2.0)
    layer.weight[mask] *= stddev
    if layer.bias is not None:
        layer.bias.data[mask] = 0.0


@torch.inference_mode()
def _get_activation(name: str, activations: dict[str, torch.Tensor]):
    """Fetches and stores the activations of a network layer."""

    def hook(layer: nn.Linear | nn.Conv2d, input: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        Get the activations of a layer with relu nonlinearity.
        ReLU has to be called explicitly here because the hook is attached to the conv/linear layer.
        """
        activations[name] = F.relu(output)

    return hook


@torch.inference_mode()
def _get_redo_masks(activations: dict[str, torch.Tensor], tau: float) -> torch.Tensor:
    """
    Computes the ReDo mask for a given set of activations.
    The returned mask has True where neurons are dormant and False where they are active.
    """
    masks = []

    # Last activation are the q-values, which are never reset
    for name, activation in list(activations.items())[:-1]:
        # Taking the mean here conforms to the expectation under D in the main paper's formula
        if activation.ndim == 4:
            # Conv layer
            score = activation.abs().mean(dim=(0, 2, 3))
        else:
            # Linear layer
            score = activation.abs().mean(dim=0)

        # Divide by activation mean to make the threshold independent of the layer size
        # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
        # https://github.com/google/dopamine/issues/209
        normalized_score = score / (score.mean() + 1e-9)

        layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
        if tau > 0.0:
            layer_mask[normalized_score <= tau] = 1
        else:
            layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = 1
        masks.append(layer_mask)
    return masks


@torch.no_grad()
def _reset_dormant_neurons(model: QNetwork, redo_masks: torch.Tensor, use_lecun_init: bool) -> QNetwork:
    """Re-initializes the dormant neurons of a model."""
    # NOTE: This code only works for the Nature-DQN architecture in this repo

    layers = [layer for _, layer in list(model.named_modules())[1:]]
    assert len(redo_masks) == len(layers) - 1, "Number of masks must match the number of layers"

    # Reset the ingoing weights
    # Here the mask size always matches the layer weight size
    for i in range(len(layers[:-1])):
        mask = redo_masks[i]
        layer = layers[i]
        next_layer = layers[i + 1]

        # Skip if there are no dead neurons
        if torch.all(~mask):
            # No dormant neurons in this layer
            continue

        # The initialization scheme is the same for conv2d and linear
        # 1. Reset the ingoing weights using the initialization distribution
        if use_lecun_init:
            _lecun_normal_reinit(layer, mask)
        else:
            _kaiming_uniform_reinit(layer, mask)

        # 2. Reset the outgoing weights to 0
        # NOTE: Don't reset the bias for the following layer or else you will create new dormant neurons
        if isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
            # Special case: Transition from conv to linear layer
            # Reset the outgoing weights to 0 with a mask created from the conv filters
            num_repeatition = next_layer.weight.data.shape[0] // mask.shape[0]
            linear_mask = torch.repeat_interleave(mask, num_repeatition)
            next_layer.weight.data[linear_mask, :] = 0.0
        else:
            # Standard case: layer and next_layer are both conv or both linear
            # Reset the outgoing weights to 0
            next_layer.weight.data[:, mask, ...] = 0.0

    return model


# FIXME: Check that this is correct
@torch.no_grad()
def _reset_adam_moments(optimizer: optim.Adam, reset_masks: dict[str, torch.Tensor]) -> optim.Adam:
    """Resets the moments of the Adam optimizer for the dormant neurons."""

    assert isinstance(optimizer, optim.Adam), "Moment resetting currently only supported for Adam optimizer"
    for i, mask in enumerate(reset_masks):
        # Reset the moments for the weights
        # NOTE: I don't think it's possible to just reset the step for moment that's being reset
        # NOTE: As far as I understand the code, they also don't reset the step count
        # optimizer.state_dict()["state"][i*2]['step'] = torch.tensor(0.0)
        optimizer.state_dict()["state"][i * 2]["exp_avg"][mask, ...] = 0.0
        optimizer.state_dict()["state"][i * 2]["exp_avg_sq"][mask, ...] = 0.0

        # Reset the moments for the bias
        # optimizer.state_dict()["state"][i*2 + 1]['step'] = torch.tensor(0.0)
        optimizer.state_dict()["state"][i * 2 + 1]["exp_avg"][mask] = 0.0
        optimizer.state_dict()["state"][i * 2 + 1]["exp_avg_sq"][mask] = 0.0

        # Reset the moments for the bias
    return optimizer


@torch.no_grad()
def run_redo(
    batch: ReplayBufferSamples,
    model: QNetwork,
    optimizer: optim.Adam,
    tau: float,
    re_initialize: bool,
    use_lecun_init: bool,
) -> tuple[nn.Module, optim.Adam, float, int]:
    """
    Checks the number of dormant neurons for a given model.
    If re_initialize is True, then the dormant neurons are re-initialized according to the scheme in
    https://arxiv.org/abs/2302.12902

    Returns the number of dormant neurons.
    """

    with torch.inference_mode():
        obs = batch.observations
        activations = {}
        activation_getter = partial(_get_activation, activations=activations)

        # Register hooks for all Conv2d and Linear layers to calculate activations
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                handles.append(module.register_forward_hook(activation_getter(name)))

        # Calculate activations
        _ = model(obs)

        # Masks for tau=0 logging
        zero_masks = _get_redo_masks(activations, 0.0)
        zero_count = sum([torch.sum(mask) for mask in zero_masks])
        zero_fraction = (zero_count / sum([torch.numel(mask) for mask in zero_masks])) * 100

        # Calculate the masks actually used for resetting
        masks = _get_redo_masks(activations, tau)
        dormant_count = sum([torch.sum(mask) for mask in masks])
        dormant_fraction = (dormant_count / sum([torch.numel(mask) for mask in masks])) * 100

        # Re-initialize the dormant neurons and reset the Adam moments
        if re_initialize:
            model = _reset_dormant_neurons(model, masks, use_lecun_init)
            optimizer = _reset_adam_moments(optimizer, masks)

        # Remove the hooks again
        for handle in handles:
            handle.remove()

        return {
            "model": model,
            "optimizer": optimizer,
            "zero_fraction": zero_fraction,
            "zero_count": zero_count,
            "dormant_fraction": dormant_fraction,
            "dormant_count": dormant_count,
        }
