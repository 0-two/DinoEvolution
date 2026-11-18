"""PPO baseline from Stable-Baselines3 on the Gymnasium wrapper of the same game.

It sees the same nine observations and picks among the same three actions as the evolved
networks. Compare by environment steps: one step is one game tick, summed over all parallel
environments. Evaluation episodes are not run during training, so every step counted is a
training step.

What "generation" means here: PPO has no generations, so the history logs one row per window of
about ``log_every`` environment steps. A row holds the best, mean and median score of the
episodes that finished inside that window. These are training-episode scores under the current
stochastic policy. For the evolutionary runs the logged best is the best of the population on its
K seeds, which is optimistic in a different way. Curves from the two families are therefore
comparable in shape, not in exact level. The generalisation test on unseen seeds, which plays
the final deterministic policy, is the like-for-like comparison.

Tuning: five settings of learning rate, entropy bonus, discount and environment count were
compared for 2M steps on one seed; the one used here had the highest training score at the end.
The first settings tried (learning rate 3e-4, discount 0.995) had not started to learn after 600k
steps. The scores are in results/notes/ppo_sweep.json and results/notes/dead_ends.json. That is
light tuning, and the report says so.

Parameters: ``n_envs``, ``n_steps`` (per env), ``batch_size``, ``n_epochs``, ``learning_rate``,
``gamma``, ``gae_lambda``, ``ent_coef``, ``hidden`` (width of the two hidden layers) and
``log_every``.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any

import numpy as np

from sim.evaluation import EvalConfig, Recorder, RunResult
from sim.genome_io import mlp_payload

DEFAULTS: dict[str, Any] = {
    "n_envs": 8,
    "n_steps": 256,
    "batch_size": 256,
    "n_epochs": 10,
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "hidden": 64,
    "log_every": 50_000,
}


def run(
    params: dict[str, Any],
    eval_cfg: EvalConfig,
    budget_steps: int,
    seed: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    # Imported here so that the other algorithms do not pay for loading PyTorch.
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from sim.env import DinoEnv

    p = {**DEFAULTS, **params}
    torch.set_num_threads(1)  # one process per run; extra threads only fight each other
    rec = Recorder("ppo", seed)
    window: list[float] = []
    last_logged = [0]

    class Log(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "episode_score" in info:
                    window.append(info["episode_score"])
            return True

        def _on_rollout_end(self) -> None:
            steps = self.num_timesteps
            if steps - last_logged[0] < p["log_every"] or not window:
                return
            rec.result.env_steps = steps
            row = rec.generation(
                window,
                episodes=len(window),
                best=max(window),
                mean=statistics.fmean(window),
                median=statistics.median(window),
            )
            rec.result.best_fitness = max(rec.result.best_fitness, row["mean"])
            window.clear()
            last_logged[0] = steps
            if progress:
                progress(row)

    env = DummyVecEnv(
        [lambda: DinoEnv(eval_cfg.max_ticks, eval_cfg.jump_penalty) for _ in range(p["n_envs"])]
    )
    env.seed(seed)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=p["n_steps"],
        batch_size=p["batch_size"],
        n_epochs=p["n_epochs"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        gae_lambda=p["gae_lambda"],
        ent_coef=p["ent_coef"],
        policy_kwargs={"net_arch": [p["hidden"], p["hidden"]]},
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=budget_steps, callback=Log())
    rec.result.env_steps = int(model.num_timesteps)

    net = model.policy
    layers = [
        (m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64))
        for m in (net.mlp_extractor.policy_net[0], net.mlp_extractor.policy_net[2], net.action_net)
    ]
    rec.result.best_genome = mlp_payload(layers, {"algorithm": "ppo", "seed": seed})
    return rec.finish()
