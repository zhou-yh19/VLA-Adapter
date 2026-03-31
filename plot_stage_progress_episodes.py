"""
Plot multiple episodes' stage progress on a single staircase chart.

Each episode is drawn as a step-function line with a distinct color/style,
sharing the same axes (x=step, y=active task 1-4).

Usage:
    python plot_stage_progress_episodes.py
    python plot_stage_progress_episodes.py --episodes 3 5 13 15 19
    python plot_stage_progress_episodes.py --output combined.png
"""

import re
import sys
import glob
import argparse
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams["font.family"] = "Times New Roman"


# ── Parsing ─────────────────────────────────────────────────────────────────

def extract_xstep_records(raw_text: str) -> list[dict]:
    stripped = re.sub(r"\s+", "", raw_text)
    pattern = re.compile(r"xstep=(\d+)\|stage_progress=\{([^}]+)\}")
    records = []
    for m in pattern.finditer(stripped):
        xstep = int(m.group(1))
        pairs = re.findall(r"(\d+):'(\w+)'", m.group(2))
        if pairs:
            sp = {int(k): v for k, v in pairs}
            records.append({"xstep": xstep, "stage_progress": sp})
    return records


def find_stage_transitions(records: list[dict]) -> list[dict]:
    transitions = []
    prev_sp = None
    for rec in records:
        sp = rec["stage_progress"]
        if sp != prev_sp:
            transitions.append({"xstep": rec["xstep"], "stage_progress": sp})
            prev_sp = sp
    return transitions


def get_active_stage(sp: dict) -> Optional[int]:
    for k, v in sp.items():
        if v == "Active":
            return k
    return None


def parse_log_file(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    records = extract_xstep_records(raw)
    if not records:
        return {"file": filepath, "total_steps": 0, "transitions": []}
    transitions = find_stage_transitions(records)
    return {
        "file": filepath,
        "total_steps": records[-1]["xstep"],
        "transitions": transitions,
    }


def build_staircase(parsed: dict) -> tuple[list[int], list[int]]:
    transitions = parsed["transitions"]
    total = parsed["total_steps"]
    if not transitions:
        return [], []
    steps = []
    stages = []
    for t in transitions:
        active = get_active_stage(t["stage_progress"])
        steps.append(t["xstep"])
        stages.append(active if active else 0)
    steps.append(total)
    stages.append(stages[-1])
    return steps, stages


# ── Plotting ────────────────────────────────────────────────────────────────

EP_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

EP_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def plot_episode_on_ax(ax, steps, stages, color, linestyle, linewidth, y_offset=0.0):
    """Draw one episode's staircase on shared axes, with optional y-offset to reduce overlap."""
    for i in range(len(steps) - 1):
        x_start = steps[i]
        x_end = steps[i + 1]
        stage = stages[i]
        y = stage + y_offset
        ax.plot([x_start, x_end], [y, y],
                color=color, linewidth=linewidth, linestyle=linestyle,
                solid_capstyle="butt")
        if i + 1 < len(steps) - 1:
            next_stage = stages[i + 1] + y_offset
            ax.plot([x_end, x_end], [y, next_stage],
                    color=color, linewidth=linewidth * 0.6,
                    linestyle=linestyle, alpha=0.5)


def main():
    parser = argparse.ArgumentParser(
        description="Plot selected episodes on a single staircase chart."
    )
    parser.add_argument(
        "--episodes", "-e", nargs="+", type=int, default=[3, 5, 13, 15, 19],
        help="Episode numbers to plot (default: 3 5 13 15 19)",
    )
    parser.add_argument(
        "--log-dir", default="eval_logs",
        help="Directory containing log files",
    )
    parser.add_argument(
        "--output", "-o", default="stage_progress_combined.png",
        help="Output image path",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    episode_data = []
    for ep in args.episodes:
        pattern = f"{args.log_dir}/shihaoran--teleavatar--organize_the_desk_stage7_{ep}--chkpt.log"
        matches = glob.glob(pattern)
        if not matches:
            print(f"Warning: no log file found for episode {ep} ({pattern})")
            continue
        parsed = parse_log_file(matches[0])
        if parsed["total_steps"] == 0:
            print(f"Warning: episode {ep} has no data")
            continue
        episode_data.append((ep, parsed))

    if not episode_data:
        print("No valid episodes to plot.")
        sys.exit(1)

    n = len(episode_data)
    offsets = [round((i - (n - 1) / 2) * 0.06, 3) for i in range(n)]
    max_steps = max(p["total_steps"] for _, p in episode_data)

    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)

    STAGE_SWAP = {13: {1: 3, 3: 1}}

    legend_handles = []
    for idx, (ep, parsed) in enumerate(episode_data):
        steps, stages = build_staircase(parsed)
        steps = [s / 10.0 for s in steps]
        if ep in STAGE_SWAP:
            swap = STAGE_SWAP[ep]
            stages = [swap.get(s, s) for s in stages]
        color = EP_COLORS[idx % len(EP_COLORS)]
        ls = EP_LINESTYLES[idx % len(EP_LINESTYLES)]
        plot_episode_on_ax(ax, steps, stages, color=color, linestyle=ls,
                           linewidth=5.0, y_offset=offsets[idx])
        total_sec = parsed["total_steps"] / 10.0
        handle = mlines.Line2D([], [], color=color, linestyle=ls, linewidth=5.0,
                                label=f"Ep {ep} ({total_sec:.1f}s)")
        legend_handles.append(handle)

    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["Task 1", "Task 2", "Task 3", "Task 4"], fontsize=20)
    ax.set_ylim(0.3, 4.7)
    ax.set_xlim(0, max_steps / 10.0 * 1.02)
    ax.set_xlabel("Time (s)", fontsize=22)
    ax.set_ylabel("Active Sub-Task", fontsize=22)
    ax.set_title("Sub-Task Execution Timeline", fontsize=26, fontweight="bold")
    ax.tick_params(axis="x", labelsize=18)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.grid(axis="y", alpha=0.15, linestyle="-")
    ax.legend(handles=legend_handles, loc="lower right", fontsize=18,
              framealpha=0.9, ncol=1)

    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close("all")


if __name__ == "__main__":
    main()
