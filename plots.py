from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt

from utils import ensure_dir, moving_average


def save_training_plots(
    out_path_prefix: str,
    returns: List[float],
    steps: List[int],
    title_prefix: str,
    ma_window: int = 100,
) -> None:
    out_dir = os.path.dirname(out_path_prefix)
    ensure_dir(out_dir)

    # Return plot
    plt.figure()
    plt.title(f"{title_prefix} | Return per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.plot(returns)
    ma = moving_average(returns, window=ma_window)
    if len(ma) == len(returns):
        plt.plot(ma)
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_returns.png", dpi=160)
    plt.close()

    # Steps plot
    plt.figure()
    plt.title(f"{title_prefix} | Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.plot(steps)
    ma_s = moving_average([float(s) for s in steps], window=ma_window)
    if len(ma_s) == len(steps):
        plt.plot(ma_s)
    plt.tight_layout()
    plt.savefig(out_path_prefix + "_steps.png", dpi=160)
    plt.close()