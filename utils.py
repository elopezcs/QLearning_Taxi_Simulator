from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Any, Dict, Optional

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def epsilon_greedy_action(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(low=0, high=q_values.shape[0]))
    return int(np.argmax(q_values))


def moving_average(x: List[float], window: int = 100) -> np.ndarray:
    if not x:
        return np.array([])
    w = max(1, int(window))
    x_arr = np.array(x, dtype=np.float64)
    if len(x_arr) < w:
        return np.full((len(x_arr),), float(np.mean(x_arr)), dtype=np.float64)
    kernel = np.ones(w, dtype=np.float64) / w
    ma = np.convolve(x_arr, kernel, mode="valid")
    pad = np.full((w - 1,), ma[0], dtype=np.float64)
    return np.concatenate([pad, ma])


# -----------------------------------------------------------------------------
# Taxi helper utilities (adapted from professor's starter helper)
# -----------------------------------------------------------------------------

def describe_env(env: Any) -> Tuple[int, int]:
    """
    Print a compact description of the environment and return (num_states, num_actions).

    Intended for sanity checks and report screenshots/snippets:
    - Observation space (Taxi-v3: Discrete(500))
    - Action space (Taxi-v3: Discrete(6))
    - Action meanings (if available)
    - Reward range (if available)
    """
    num_obs = int(env.observation_space.n)
    num_actions = int(env.action_space.n)

    print("Environment description")
    print(f"  Observation space: {env.observation_space} -> n={num_obs}")
    print(f"  Action space:      {env.action_space} -> n={num_actions}")

    # Action meanings are usually available via env.unwrapped.get_action_meanings()
    meanings = None
    try:
        meanings = env.unwrapped.get_action_meanings()
    except Exception:
        meanings = None

    if meanings:
        print(f"  Action meanings:   {meanings}")

    # Reward range may or may not exist depending on wrappers
    reward_range = getattr(env, "reward_range", None)
    if reward_range is not None:
        print(f"  Reward range:      {reward_range}")

    return num_obs, num_actions


def breakdown_obs(obs: int) -> Dict[str, int]:
    """
    Decode Taxi-v3 scalar observation into human-readable components.

    Taxi encoding (as in Gym Taxi docs):
        ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination

    Returns dict with:
        taxi_row, taxi_col, passenger_location, destination
    """
    o = int(obs)

    destination = o % 4
    o //= 4

    passenger_location = o % 5
    o //= 5

    taxi_col = o % 5
    o //= 5

    taxi_row = o

    return {
        "taxi_row": int(taxi_row),
        "taxi_col": int(taxi_col),
        "passenger_location": int(passenger_location),
        "destination": int(destination),
    }


def describe_obs(obs: int) -> str:
    """
    Return a human-readable string describing a Taxi-v3 observation.

    passenger_location mapping:
        0: Red
        1: Green
        2: Yellow
        3: Blue
        4: In taxi

    destination mapping (0..3):
        0: Red
        1: Green
        2: Yellow
        3: Blue
    """
    parts = breakdown_obs(obs)

    loc_names = ["Red", "Green", "Yellow", "Blue", "In taxi"]
    dest_names = ["Red", "Green", "Yellow", "Blue"]

    p_loc = parts["passenger_location"]
    dest = parts["destination"]

    p_str = loc_names[p_loc] if 0 <= p_loc < len(loc_names) else f"Unknown({p_loc})"
    d_str = dest_names[dest] if 0 <= dest < len(dest_names) else f"Unknown({dest})"

    return (
        f"obs={int(obs)} | taxi=(row={parts['taxi_row']}, col={parts['taxi_col']}), "
        f"passenger={p_str}, destination={d_str}"
    )


def simulate_greedy_policy(
    env_id: str,
    policy_fn: Callable[[int], int],
    episodes: int = 3,
    seed: int = 123,
    max_steps: int = 200,
    render_mode: str = "human",
    print_state_decode: bool = True,
) -> None:
    """
    Simulate episodes using a provided greedy policy function.

    This mirrors the professor's simulate_episodes idea but works with your
    function-based implementation (policy_fn(state)->action).

    Notes:
    - Uses Gymnasium API: step() -> (obs, reward, terminated, truncated, info)
    - If render_mode="human", a window will appear (when supported by your setup).

    Parameters
    ----------
    env_id : str
        Environment id, typically "Taxi-v3".
    policy_fn : Callable[[int], int]
        Function mapping integer state -> integer action.
    episodes : int
        Number of episodes to simulate.
    seed : int
        RNG seed for reproducibility.
    max_steps : int
        Safety cap to avoid infinite loops.
    render_mode : str
        "human" for rendering; you can set None or "ansi" if desired.
    print_state_decode : bool
        If True, prints decoded state info each step (can be verbose).
    """
    # Import locally to avoid circular imports if you restructure later
    import gymnasium as gym

    rng = np.random.default_rng(seed)

    env = gym.make(env_id, render_mode=render_mode)
    try:
        for ep in range(episodes):
            obs, _info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            print(f"\nSimulation episode {ep + 1}/{episodes}")
            if print_state_decode:
                print("  " + describe_obs(int(obs)))

            while not (terminated or truncated) and steps < max_steps:
                state = int(obs)
                action = int(policy_fn(state))

                obs, reward, terminated, truncated, _info = env.step(action)
                total_reward += float(reward)
                steps += 1

                if print_state_decode:
                    print(f"  step={steps:3d} action={action} reward={float(reward):6.1f} | {describe_obs(int(obs))}")
                else:
                    print(f"  step={steps:3d} action={action} reward={float(reward):6.1f}")

            print(f"Episode finished: steps={steps}, total_reward={total_reward:.1f}, terminated={terminated}, truncated={truncated}")

    finally:
        env.close()


# -----------------------------------------------------------------------------
# Evaluation (unchanged)
# -----------------------------------------------------------------------------

@dataclass
class EvalResult:
    avg_return: float
    avg_steps: float
    success_rate: float


def evaluate_policy(
    make_env: Callable[[], Any],
    policy_fn: Callable[[int], int],
    n_episodes: int = 200,
    seed: int = 123,
    max_steps: int = 500,
) -> EvalResult:
    rng = np.random.default_rng(seed)
    returns: List[float] = []
    steps_list: List[int] = []
    successes = 0

    for _ in range(n_episodes):
        env = make_env()
        obs, _info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        terminated = False
        truncated = False
        ep_return = 0.0
        ep_steps = 0

        while not (terminated or truncated) and ep_steps < max_steps:
            action = int(policy_fn(int(obs)))
            obs, reward, terminated, truncated, _info = env.step(action)
            ep_return += float(reward)
            ep_steps += 1

        # Simple success heuristic for Taxi-v3
        if terminated and ep_return > -50:
            successes += 1

        returns.append(ep_return)
        steps_list.append(ep_steps)
        env.close()

    return EvalResult(
        avg_return=float(np.mean(returns)),
        avg_steps=float(np.mean(steps_list)),
        success_rate=float(successes / n_episodes),
    )