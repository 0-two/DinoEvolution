"""Save and load trained policies as JSON the web app can read.

Four formats, because the algorithms produce four different kinds of policy:

``dino-genome/1``  fixed 9-8-3 network, 96 weights (GA, CMA-ES). The browser GA uses it too.
``dino-neat/1``    feed-forward graph as an ordered evaluation plan (NEAT).
``dino-mlp/1``     stack of dense layers with tanh between them (the PPO policy).
``dino-rule/1``    the two numbers of the rule-based baseline.

Every file may carry a ``meta`` object (algorithm, seed, scores). Readers ignore it.
``load_policy`` turns any of them back into ``act(observation) -> action``, which is what lets
the experiment runner test an exported genome exactly as the browser would play it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from sim.game import DINO_H, DINO_W, DINO_X, DUCK, DUCK_H, JUMP, NOOP, START_SPEED, DinoGame
from sim.network import GENOME_LENGTH, HIDDEN, INPUTS, OUTPUTS, FixedNetwork

Policy = Callable[[list[float]], int]
GameAct = Callable[[DinoGame], int]


# --- writers -------------------------------------------------------------------------------
def fixed_genome_payload(weights: np.ndarray, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "format": "dino-genome/1",
        "topology": {"inputs": INPUTS, "hidden": HIDDEN, "outputs": OUTPUTS, "activation": "tanh"},
        "weights": [float(w) for w in weights],
        "meta": meta or {},
    }


def neat_payload(genome: Any, config: Any, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Export a neat-python genome as the evaluation plan neat-python itself would run.

    The plan comes from ``FeedForwardNetwork.create``, so it is exact by construction and
    includes details that vary between neat-python versions: nodes that no connection feeds
    act as constants ``tanh(bias)``, and an output with no inputs is such a node too.

    Node ids follow neat-python: inputs are -1..-9, outputs are 0..2, hidden nodes are
    positive integers above that. ``nodes`` is in evaluation order. Each node computes
    ``tanh(bias + response * sum(w * value[source]))``.
    """
    import neat

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return {
        "format": "dino-neat/1",
        "inputs": list(net.input_nodes),
        "outputs": list(net.output_nodes),
        "activation": "tanh",
        "nodes": [
            {
                "id": node,
                "bias": float(bias),
                "response": float(response),
                "inputs": [[src, float(w)] for src, w in links],
            }
            for node, _act, _agg, bias, response, links in net.node_evals
        ],
        "meta": meta or {},
    }


def mlp_payload(
    layers: list[tuple[np.ndarray, np.ndarray]], meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Dense layers as (weight matrix [out, in], bias [out]). tanh between layers, none after."""
    return {
        "format": "dino-mlp/1",
        "activation": "tanh",
        "layers": [
            {"weights": np.asarray(w).tolist(), "bias": np.asarray(b).tolist()} for w, b in layers
        ],
        "meta": meta or {},
    }


def rule_payload(d: float, gain: float, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"format": "dino-rule/1", "d": d, "gain": gain, "meta": meta or {}}


def save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


# --- readers -------------------------------------------------------------------------------
def load_policy(payload: dict[str, Any]) -> Policy:
    """Build ``act(observation)`` from a payload. Not available for ``dino-rule/1``, which
    reads the game state directly; use :func:`load_rule` for that."""
    fmt = payload["format"]
    if fmt == "dino-genome/1":
        t = payload["topology"]
        if (t["inputs"], t["hidden"], t["outputs"]) != (INPUTS, HIDDEN, OUTPUTS):
            raise ValueError("genome does not match the 9-8-3 topology")
        if len(payload["weights"]) != GENOME_LENGTH:
            raise ValueError(f"expected {GENOME_LENGTH} weights")
        return FixedNetwork(payload["weights"]).act
    if fmt == "dino-neat/1":
        return _graph_policy(payload)
    if fmt == "dino-mlp/1":
        return _mlp_policy(payload)
    raise ValueError(f"no observation policy for format {fmt!r}")


def load_game_policy(payload: dict[str, Any]) -> GameAct:
    """Like :func:`load_policy` but returns ``act(game) -> action``, and also accepts rules."""
    if payload["format"] == "dino-rule/1":
        return load_rule(payload)
    act = load_policy(payload)
    return lambda game: act(game.observe())


def load_rule(payload: dict[str, Any]) -> GameAct:
    d, gain = float(payload["d"]), float(payload["gain"])

    def act(game: DinoGame) -> int:
        if game.airborne:
            return NOOP
        first, _ = game.upcoming()
        if first is None or first[0] - (DINO_X + DINO_W) >= d + gain * (game.speed - START_SPEED):
            return NOOP
        if first[1] >= DINO_H:
            return NOOP
        return DUCK if first[1] >= DUCK_H else JUMP

    return act


def _graph_policy(payload: dict[str, Any]) -> Policy:
    inputs: list[int] = payload["inputs"]
    outputs: list[int] = payload["outputs"]
    plan = [(n["id"], n["bias"], n["response"], n["inputs"]) for n in payload["nodes"]]

    def act(obs: list[float]) -> int:
        # Outputs start at 0.0, as in neat-python, in case a node never gets evaluated.
        value = dict.fromkeys(outputs, 0.0)
        value.update(zip(inputs, obs, strict=True))
        for node, bias, response, links in plan:
            total = sum(value[src] * w for src, w in links)
            value[node] = math.tanh(bias + response * total)
        outs = [value[o] for o in outputs]
        return outs.index(max(outs))

    return act


def _mlp_policy(payload: dict[str, Any]) -> Policy:
    layers = [
        (
            np.asarray(layer["weights"], dtype=np.float64),
            np.asarray(layer["bias"], dtype=np.float64),
        )
        for layer in payload["layers"]
    ]

    def act(obs: list[float]) -> int:
        x = np.asarray(obs, dtype=np.float64)
        for i, (w, b) in enumerate(layers):
            x = w @ x + b
            if i < len(layers) - 1:
                x = np.tanh(x)
        return int(np.argmax(x))

    return act
