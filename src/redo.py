import math
from functools import partial

import torch
import torch.nn as nn
from einops import repeat
from torch import optim

from .agent import QNetwork
from .buffer import ReplayBufferSamples


def _init_bias_kaiming_uniform(layer: nn.Linear | nn.Conv2d, mask: torch.Tensor) -> None:
    """Partially re-initializes the bias of a layer according to the Kaiming uniform scheme."""
    if isinstance(layer, nn.Conv2d):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(layer.bias[mask], -bound, bound)
    else:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(layer.bias[mask], -bound, bound)
    # TODO: Reset grads to None


@torch.inference_mode()
def _get_activation(name: str, activations: dict[str, torch.Tensor]):
    """Fetches and stores the activations of a network layer."""

    def hook(layer: nn.Linear | nn.Conv2d, input: tuple[torch.Tensor], output: torch.Tensor) -> None:
        if isinstance(layer, nn.Conv2d):
            activations[name] = output.abs().mean(dim=(0, 2, 3))
        else:
            assert isinstance(layer, nn.Linear), "Hook only supported for Conv2d and Linear layers"
            activations[name] = output.abs().mean(dim=0)

    return hook


@torch.inference_mode()
def _get_redo_masks(activations: dict[str, torch.Tensor], tau: float) -> torch.Tensor:
    """
    Computes the ReDo mask for a given set of activations.
    The returned mask has True where neurons are dormant and False where they are active.
    """
    masks = []
    for name, activation in list(activations.items())[:-1]:
        layer_mask = torch.ones_like(activation, dtype=torch.bool)
        # Divide by activation mean to make the threshold independent of the layer size
        # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
        # https://github.com/google/dopamine/issues/209
        normalized_activation = activation / (activation.mean() + 1e-9)
        if tau > 0.0:
            layer_mask[normalized_activation > tau] = 0
        else:
            layer_mask[torch.isclose(normalized_activation, torch.zeros_like(normalized_activation))] = 0
        masks.append(layer_mask)
    return masks


def _reset_dorman_neurons(model: QNetwork, redo_masks: torch.Tensor) -> QNetwork:
    """Re-initializes the dormant neurons of a model."""

    layers = list(model.named_modules())
    reset_layers = layers[1:-1]
    next_layers = layers[2:]
    prev_mask = None

    # NOTE: Neither continual backprop nor ReDo reset gradients I think
    for (name, layer), (next_name, next_layer), mask in zip(reset_layers, next_layers, redo_masks, strict=True):
        # TODO: Check that a neuron is not re-initialized twice
        if not torch.all(mask):
            # No dormant neurons in this layer
            continue
        elif "q" in next_name:
            # Last layer is reached, we're done checking and return
            return model
        elif isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight.data[mask, ...], a=math.sqrt(5))
            if layer.bias is not None:
                _init_bias_kaiming_uniform(layer, mask)

            # 2. Reset the outgoing weights to 0 with a mask created from the conv filters
            linear_mask = torch.flatten(repeat(mask, " c -> c h w ", h=7, w=7), start_dim=0)
            next_layer.weight.data[:, linear_mask, ...].data.fill_(0)
            if next_layer.bias is not None:
                next_layer.bias.data[linear_mask].data.fill_(0)
        else:
            # The initialization scheme is the same for conv2d and linear
            # 1. Reset the ingoing weights using the initialization distribution
            nn.init.kaiming_uniform_(layer.weight.data[mask, ...], a=math.sqrt(5))
            if layer.bias is not None:
                _init_bias_kaiming_uniform(layer, mask)

            # 2. Reset the outgoing weights to 0
            next_layer.weight.data[:, mask, ...].data.fill_(0)
            if next_layer.bias is not None:
                next_layer.bias.data[mask].data.fill_(0)

            prev_mask = mask

    return model


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


def run_redo(
    batch: ReplayBufferSamples,
    model: QNetwork,
    optimizer: optim.Adam,
    tau: float,
    re_initialize: bool,
) -> tuple[nn.Module, optim.Adam, float]:
    """
    Checks the number of dormant neurons for a given model.
    If re_initialize is True, then the dormant neurons are re-initialized according to the scheme in
    https://arxiv.org/abs/2302.12902

    Returns the number of dormant neurons.
    """

    # Just count dormant neurons when not resetting
    if not re_initialize:
        tau = 0.0

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
        # Remove the hooks again
        for handle in handles:
            handle.remove()

        # Calculate the ReDo masks and dormant neuron fraction
        masks = _get_redo_masks(activations, tau)
        dormant_count = sum([torch.sum(mask) for mask in masks])
        dormant_fraction = dormant_count / sum([torch.numel(mask) for mask in masks])

    # Return without re-initializing anything because we're just counting
    if not re_initialize:
        return model, optimizer, dormant_fraction

    # Re-initialize the dormant neurons and reset the Adam moments
    _reset_dorman_neurons(model, masks)
    _reset_adam_moments(optimizer, masks)

    return model, optimizer, dormant_fraction
