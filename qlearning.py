from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
import numpy as np

from plots import save_training_plots
from utils import ensure_dir, epsilon_greedy_action, evaluate_policy, describe_env


@dataclass
class QLConfig:
    env_id: str = "Taxi-v3"
    episodes: int = 10000
    alpha: float = 0.1
    gamma: float = 0.9
    epsilon: float = 0.1
    seed: int = 42
    max_steps: int = 500


def make_env_factory(env_id: str) -> Callable[[], gym.Env]:
    def _make() -> gym.Env:
        return gym.make(env_id)

    return _make


def train_q_learning(
    cfg: QLConfig,
    return_q: bool = False,
    save_q_path: str | None = None,
) -> Dict:
    """
    Train a tabular Q-Learning agent on the specified Gymnasium environment.

    Implements:
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    If save_q_path is provided, the learned Q-table is saved to disk using np.save.
    If return_q is True, the returned dict includes a "q_table" key.

    Returns a dict with training metrics, evaluation metrics, and per-episode histories.
    """
    make_env = make_env_factory(cfg.env_id)
    env = make_env()

    # Helpful environment sanity check (prof-style helper)
    # You already added Option A, but keeping it here is safe and useful.
    describe_env(env)

    n_states = int(env.observation_space.n)
    n_actions = int(env.action_space.n)
    q = np.zeros((n_states, n_actions), dtype=np.float64)

    rng = np.random.default_rng(cfg.seed)

    returns: List[float] = []
    steps_list: List[int] = []

    for _ep in range(cfg.episodes):
        obs, _info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        terminated = False
        truncated = False

        ep_return = 0.0
        ep_steps = 0

        while not (terminated or truncated) and ep_steps < cfg.max_steps:
            s = int(obs)
            a = epsilon_greedy_action(q[s], cfg.epsilon, rng)

            obs2, r, terminated, truncated, _info = env.step(a)
            s2 = int(obs2)

            td_target = float(r) + (0.0 if (terminated or truncated) else cfg.gamma * float(np.max(q[s2])))
            td_error = td_target - q[s, a]
            q[s, a] = q[s, a] + cfg.alpha * td_error

            obs = obs2
            ep_return += float(r)
            ep_steps += 1

        returns.append(ep_return)
        steps_list.append(ep_steps)

    env.close()

    def greedy_policy(state: int) -> int:
        return int(np.argmax(q[state]))

    eval_res = evaluate_policy(
        make_env=make_env,
        policy_fn=greedy_policy,
        n_episodes=200,
        seed=cfg.seed + 999,
        max_steps=cfg.max_steps,
    )

    # Optionally save learned Q-table for later simulation/inspection
    if save_q_path:
        out_dir = os.path.dirname(save_q_path)
        if out_dir:
            ensure_dir(out_dir)
        np.save(save_q_path, q)

    result: Dict = {
        "algo": "qlearning",
        "env_id": cfg.env_id,
        "episodes": cfg.episodes,
        "alpha": cfg.alpha,
        "gamma": cfg.gamma,
        "epsilon": cfg.epsilon,
        "seed": cfg.seed,
        "train_avg_return": float(np.mean(returns)),
        "train_avg_steps": float(np.mean(steps_list)),
        "eval_avg_return": eval_res.avg_return,
        "eval_avg_steps": eval_res.avg_steps,
        "eval_success_rate": eval_res.success_rate,
        "returns": returns,
        "steps": steps_list,
    }

    if return_q:
        result["q_table"] = q

    return result


def build_experiment_configs(env_id: str, episodes: int, seed: int, max_steps: int) -> List[QLConfig]:
    """
    Builds the experiment list aligned with the deliverables you cited.

    Baseline:
        alpha=0.1, gamma=0.9, epsilon=0.1

    Hyperparameter sweeps:
        alpha in [0.01, 0.001, 0.2] with epsilon=0.1, gamma=0.9
        epsilon in [0.2, 0.3] with alpha=0.1, gamma=0.9
    """
    configs: List[QLConfig] = []

    baseline = QLConfig(
        env_id=env_id,
        episodes=episodes,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1,
        seed=seed,
        max_steps=max_steps,
    )
    configs.append(baseline)

    for a in [0.01, 0.001, 0.2]:
        configs.append(
            QLConfig(
                env_id=env_id,
                episodes=episodes,
                alpha=a,
                gamma=0.9,
                epsilon=0.1,
                seed=seed,
                max_steps=max_steps,
            )
        )

    for e in [0.2, 0.3]:
        configs.append(
            QLConfig(
                env_id=env_id,
                episodes=episodes,
                alpha=0.1,
                gamma=0.9,
                epsilon=e,
                seed=seed,
                max_steps=max_steps,
            )
        )

    return configs


def run_qlearning_experiments(
    out_dir: str,
    env_id: str = "Taxi-v3",
    episodes: int = 10000,
    seed: int = 42,
    max_steps: int = 500,
) -> List[Dict]:
    """
    Runs baseline + alpha sweep + epsilon sweep, saving plots and a summary CSV.
    """
    ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)

    configs = build_experiment_configs(env_id=env_id, episodes=episodes, seed=seed, max_steps=max_steps)
    results: List[Dict] = []

    for cfg in configs:
        res = train_q_learning(cfg)
        results.append(res)

        tag = f"qlearning_a{cfg.alpha}_g{cfg.gamma}_e{cfg.epsilon}_seed{cfg.seed}"
        title = f"Q-Learning (alpha={cfg.alpha}, gamma={cfg.gamma}, epsilon={cfg.epsilon})"
        save_training_plots(
            out_path_prefix=os.path.join(plots_dir, tag),
            returns=res["returns"],
            steps=res["steps"],
            title_prefix=title,
        )

    write_summary_csv(os.path.join(out_dir, "qlearning_results.csv"), results)
    return results


def pick_best_alpha_epsilon(results: List[Dict]) -> Tuple[float, float]:
    """
    Choose best based on eval_avg_return (higher is better).
    Tie-breaker: fewer eval steps.
    """
    def key(r: Dict) -> Tuple[float, float]:
        return (float(r["eval_avg_return"]), -float(r["eval_avg_steps"]))

    best = max(results, key=key)
    return float(best["alpha"]), float(best["epsilon"])


def rerun_best_combo(
    out_dir: str,
    prior_results: List[Dict],
    env_id: str,
    episodes: int,
    seed: int,
    max_steps: int,
) -> Dict:
    """
    Picks the best (alpha, epsilon) from prior_results, reruns training,
    and saves the BEST Q-table and metadata for later simulation.

    Outputs:
        outputs/qtable_best.npy
        outputs/qtable_best_meta.json
    """
    plots_dir = os.path.join(out_dir, "plots")
    ensure_dir(plots_dir)

    best_alpha, best_epsilon = pick_best_alpha_epsilon(prior_results)

    cfg = QLConfig(
        env_id=env_id,
        episodes=episodes,
        alpha=best_alpha,
        gamma=0.9,
        epsilon=best_epsilon,
        seed=seed,
        max_steps=max_steps,
    )

    best_q_path = os.path.join(out_dir, "qtable_best.npy")
    res = train_q_learning(cfg, save_q_path=best_q_path)

    res["is_best_rerun"] = True

    # Save plots for the best rerun
    tag = f"qlearning_BEST_a{cfg.alpha}_g{cfg.gamma}_e{cfg.epsilon}_seed{cfg.seed}"
    title = f"Q-Learning BEST (alpha={cfg.alpha}, gamma={cfg.gamma}, epsilon={cfg.epsilon})"
    save_training_plots(
        out_path_prefix=os.path.join(plots_dir, tag),
        returns=res["returns"],
        steps=res["steps"],
        title_prefix=title,
    )

    # Save metadata about the best Q-table
    best_meta_path = os.path.join(out_dir, "qtable_best_meta.json")
    meta = {
        "env_id": cfg.env_id,
        "episodes": cfg.episodes,
        "alpha": cfg.alpha,
        "gamma": cfg.gamma,
        "epsilon": cfg.epsilon,
        "seed": cfg.seed,
        "max_steps": cfg.max_steps,
        "qtable_path": os.path.basename(best_q_path),
    }
    with open(best_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Rewrite CSV including best rerun row
    all_rows = [strip_large_lists(r) for r in prior_results] + [strip_large_lists(res)]
    write_summary_csv(os.path.join(out_dir, "qlearning_results.csv"), all_rows)

    return res


def strip_large_lists(r: Dict) -> Dict:
    rr = dict(r)
    rr.pop("returns", None)
    rr.pop("steps", None)
    rr.pop("q_table", None)
    return rr


def write_summary_csv(csv_path: str, results: List[Dict]) -> None:
    fieldnames = [
        "algo",
        "env_id",
        "episodes",
        "alpha",
        "gamma",
        "epsilon",
        "seed",
        "train_avg_return",
        "train_avg_steps",
        "eval_avg_return",
        "eval_avg_steps",
        "eval_success_rate",
        "is_best_rerun",
    ]

    rows = []
    for r in results:
        base = strip_large_lists(r)
        if "is_best_rerun" not in base:
            base["is_best_rerun"] = False
        for k in fieldnames:
            base.setdefault(k, "")
        rows.append({k: base.get(k, "") for k in fieldnames})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)