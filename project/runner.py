import numpy as np
import matplotlib.pyplot as plt
from env import Market, Actions


def run_strategy(env: Market, agent) -> dict:
    obs, _ = env.reset()

    pnl = []
    s0_prices = []
    s1_prices = []
    spreads = []
    zscores = []
    entries = []
    exits = []

    step = 0
    while True:
        action = agent.predict(obs)
        in_pos_before = obs["position"]

        s0_prices.append(env._pair_data.s[0][env._t])
        s1_prices.append(env._pair_data.s[1][env._t])
        spreads.append(float(obs["spread"][0]))
        zscores.append(float(obs["spread_z_score"][0]))

        obs, reward, terminated, truncated, _ = env.step(action)
        pnl.append(env._get_pnl())

        if action == Actions.Swap.value:
            if in_pos_before == 0:
                entries.append(step)
            else:
                exits.append(step)

        step += 1
        if terminated or truncated:
            break

    return {
        "pnl": np.array(pnl),
        "s0": np.array(s0_prices),
        "s1": np.array(s1_prices),
        "spreads": np.array(spreads),
        "zscores": np.array(zscores),
        "entries": entries,
        "exits": exits,
        "labels": env._pair_data.labels,
    }


def plot_strategy(results: dict):
    pnl = results["pnl"]
    s0 = results["s0"]
    s1 = results["s1"]
    spreads = results["spreads"]
    zscores = results["zscores"]
    entries = results["entries"]
    exits = results["exits"]
    labels = results["labels"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    axes[0].plot(pnl)
    axes[0].set_title("Portfolio Value")
    axes[0].set_ylabel("Value ($)")

    axes[1].plot(s0, label=labels[0])
    axes[1].plot(s1, label=labels[1])
    if entries:
        axes[1].scatter(entries, s0[entries], color="green", zorder=5, s=25, label="entry")
        axes[1].scatter(entries, s1[entries], color="green", zorder=5, s=25)
    if exits:
        axes[1].scatter(exits, s0[exits], color="red", zorder=5, s=25, label="exit")
        axes[1].scatter(exits, s1[exits], color="red", zorder=5, s=25)
    axes[1].set_title("Stock Prices")
    axes[1].set_ylabel("Price ($)")
    axes[1].legend()

    axes[2].plot(zscores)
    axes[2].axhline(0,    color="gray",   linewidth=0.8, linestyle="--")
    axes[2].axhline(2.0,  color="green",  linewidth=0.8, linestyle="--")
    axes[2].axhline(-2.0, color="green",  linewidth=0.8, linestyle="--")
    axes[2].axhline(0.5,  color="orange", linewidth=0.8, linestyle="--")
    axes[2].axhline(-0.5, color="orange", linewidth=0.8, linestyle="--")
    if entries:
        axes[2].scatter(entries, zscores[entries], color="green", zorder=5, s=25, label="entry")
    if exits:
        axes[2].scatter(exits, zscores[exits], color="red", zorder=5, s=25, label="exit")
    axes[2].set_title("Z-score (green=entry ±2, orange=exit ±0.5)")
    axes[2].set_ylabel("Z-score")
    axes[2].legend()

    axes[3].plot(spreads)
    axes[3].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    if entries:
        axes[3].scatter(entries, spreads[entries], color="green", zorder=5, s=25, label="entry")
    if exits:
        axes[3].scatter(exits, spreads[exits], color="red", zorder=5, s=25, label="exit")
    axes[3].set_title("Raw spread (beta-adjusted, rolling window)")
    axes[3].set_ylabel("Spread")
    axes[3].set_xlabel("Time step")
    axes[3].legend()

    plt.tight_layout()
    plt.show()

    return fig
