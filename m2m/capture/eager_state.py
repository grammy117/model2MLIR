"""Observe eager tensor state without changing the inputs used for capture.

Input deepcopy preserves shared storage between input views. Module buffers are
restored after observing the endpoint, including when the forward raises. Shared
input/buffer storage and metadata-changing buffers need a dedicated invocation
contract and are refused before execution rather than silently breaking aliases.
"""

import copy
from dataclasses import dataclass

import torch


@dataclass
class Observation:
    output: object
    inputs: tuple
    buffers: dict


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    return value


def observe(model, inputs, run=None):
    buffers = dict(model.named_buffers(remove_duplicate=False))
    roots = {value.untyped_storage()._cdata for value in buffers.values()}
    if any(isinstance(value, torch.Tensor) and value.untyped_storage()._cdata in roots for value in inputs):
        raise ValueError("eager capture requires explicit shared input/buffer alias semantics")
    # Detach first: the reference is evaluated without autograd and nonleaf
    # example inputs may otherwise reject deepcopy. The storage memo retains
    # input/input aliases even when distinct Tensor objects describe the views.
    arguments = copy.deepcopy(tuple(value.detach() if isinstance(value, torch.Tensor) else value for value in inputs))
    initial = {name: (value, _snapshot(value), tuple(value.shape), tuple(value.stride()))
               for name, value in buffers.items()}
    with torch.no_grad():
        try:
            output = model(*arguments) if run is None else run(arguments)
            endpoint = dict(model.named_buffers(remove_duplicate=False))
            if set(endpoint) != set(initial) or any(
                endpoint[name] is not original or tuple(original.shape) != shape or tuple(original.stride()) != stride
                for name, (original, _saved, shape, stride) in initial.items()
            ):
                raise ValueError("eager capture cannot restore changed buffer identities or metadata")
            return Observation(_snapshot(output), _snapshot(arguments), _snapshot(endpoint))
        finally:
            for original, saved, shape, stride in initial.values():
                if tuple(original.shape) == shape and tuple(original.stride()) == stride:
                    original.copy_(saved)


def functional_outputs(observation, exported, contract):
    """Order independently observed eager endpoints by the actual export ABI."""
    if contract is None:
        return observation.output
    from torch.export.graph_signature import InputKind

    users = (() if observation.output is None and not contract["user_outputs"] else
             observation.output if isinstance(observation.output, (tuple, list)) else (observation.output,))
    if len(users) != len(contract["user_outputs"]):
        raise ValueError("eager user results differ from the exported functional state contract")
    result = [None] * contract["output_count"]
    for index, value in zip(contract["user_outputs"], users, strict=True):
        result[index] = value
    user_specs = [spec for spec in exported.graph_signature.input_specs if spec.kind == InputKind.USER_INPUT]
    for state in contract["state_outputs"]:
        spec = exported.graph_signature.input_specs[state["input"]]
        value = (observation.inputs[user_specs.index(spec)] if spec.kind == InputKind.USER_INPUT
                 else observation.buffers[str(spec.target)])
        result[state["result"]] = value
    if any(not isinstance(value, torch.Tensor) for value in result):
        raise ValueError("functional state references require every original Tensor result")
    return tuple(result)
