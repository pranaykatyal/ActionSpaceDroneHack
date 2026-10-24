"""Render training curves to PNG, so they can be read without TensorBoard.

    python scripts/plot_runs.py --es <es run dir> --out outputs/plots/es.png
    python scripts/plot_runs.py --ppo <skrl run dir> [--compare <skrl run dir>] --out outputs/plots/ppo.png

ES runs: the unperturbed policy's breach and interception rate per generation (fresh scenarios every
generation) and the population fitness. PPO runs: skrl's logged reward, value loss and policy std, plus
the official breach rate per checkpoint from <run>/eval_sweep.txt (scripts/skrl/eval_sweep.sh).
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BEST_FRESH_BREACH = 69.7  # FT2b agent_400, evaluate.py 100 envs x 300 episodes


def scalars(run: str) -> dict[str, list[tuple[float, float]]]:
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "events*"))):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            out.setdefault(tag, []).extend((e.step, e.value) for e in ea.Scalars(tag))
    return out


def sweep(run: str) -> list[tuple[int, float, float]]:
    """(iteration, interception %, breach %) per checkpoint of an eval_sweep.txt."""
    rows = []
    path = os.path.join(run, "eval_sweep.txt")
    if not os.path.exists(path):
        return rows
    for line in open(path):
        m = re.match(r"agent_(\d+)\s.*interception rate: ([\d.]+)%.*breach rate: ([\d.]+)%", line)
        if m:
            rows.append((int(m.group(1)) // 16, float(m.group(2)), float(m.group(3))))
    return rows


def moving(ys, k=5):
    return [sum(ys[max(0, i - k + 1) : i + 1]) / len(ys[max(0, i - k + 1) : i + 1]) for i in range(len(ys))]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_es(run, out):
    s = scalars(run)
    breach = [(x, 100 * y) for x, y in s.get("ES / center breach rate", [])]
    inter = [(x, 100 * y) for x, y in s.get("ES / center interception rate", [])]
    fit = s.get("ES / population fitness mean", [])
    pop, center = 240, 256
    args_txt = os.path.join(run, "params", "es_args.txt")
    if os.path.exists(args_txt):
        m = re.search(r"'pop': (\d+)", open(args_txt).read())
        pop = int(m.group(1)) if m else pop
        m = re.search(r"'center_envs': (\d+)", open(args_txt).read())
        center = int(m.group(1)) if m else center
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#fcfcfb")
    for ax, data, title, ylabel, color in [
        (axes[0], breach, f"Breach rate, unperturbed policy ({center} fresh episodes / gen)", "% of episodes", SERIES[0]),
        (axes[1], inter, "Interception rate, unperturbed policy", "% of attackers", SERIES[1]),
        (axes[2], fit, f"Population fitness (mean of {pop} copies)", "fitness", SERIES[2]),
    ]:
        ax.set_facecolor("#fcfcfb")
        if data:
            xs, ys = zip(*data)
            ax.plot(xs, ys, color=color, linewidth=1, alpha=0.35, label="per generation")
            ax.plot(xs, moving(list(ys)), color=color, linewidth=2, label="5-gen average")
        style(ax, title, "generation", ylabel)
    if breach and max(y for _, y in breach) > 40:  # the old best is only a useful reference near it
        axes[0].axhline(BEST_FRESH_BREACH, color=MUTED, linewidth=1, linestyle="--")
        axes[0].text(axes[0].get_xlim()[0], BEST_FRESH_BREACH, " start: FT2b fresh 69.7%", color=MUTED, fontsize=8, va="bottom")
    axes[0].axhline(20.0, color=MUTED, linewidth=1, linestyle=":")
    axes[0].text(axes[0].get_xlim()[0], 20.0, " goal 20%", color=MUTED, fontsize=8, va="bottom")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(os.path.basename(run.rstrip("/")), x=0.01, ha="left", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def plot_ppo(run, compare, out):
    runs = [r for r in (run, compare) if r]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#fcfcfb")
    panels = [
        (axes[0, 0], "Official breach rate per checkpoint (evaluate.py, 64 ep)", "% of episodes"),
        (axes[0, 1], "skrl reward, mean of last 100 finished episodes", "episode return"),
        (axes[1, 0], "Value loss (defender_0)", "loss"),
        (axes[1, 1], "Policy std (defender_0)", "std"),
    ]
    for ax, title, ylabel in panels:
        ax.set_facecolor("#fcfcfb")
        style(ax, title, "iteration (16 steps x 4096 envs)", ylabel)
    for color, r in zip(SERIES, runs):
        label = os.path.basename(r.rstrip("/"))
        rows = sweep(r)
        if rows:
            axes[0, 0].plot([a for a, _, _ in rows], [c for _, _, c in rows], color=color, linewidth=2, marker="o", markersize=4, label=label)
        s = scalars(r)
        for ax, tag in [(axes[0, 1], "Reward / Total reward (mean)"), (axes[1, 0], "Loss / Value loss (defender_0)"), (axes[1, 1], "Policy / Standard deviation (defender_0)")]:
            data = s.get(tag) or next((v for k, v in s.items() if k.startswith(tag.split(" (")[0])), [])
            if data:
                ax.plot([x / 16 for x, _ in data], [y for _, y in data], color=color, linewidth=2, label=label)
    axes[0, 0].axhline(BEST_FRESH_BREACH, color=MUTED, linewidth=1, linestyle="--")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--es")
    p.add_argument("--ppo")
    p.add_argument("--compare")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plot_es(a.es, a.out) if a.es else plot_ppo(a.ppo, a.compare, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
