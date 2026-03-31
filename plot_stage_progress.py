"""
Plot stage progress as staircase diagrams.

For each episode, the active stage (Task) is plotted as a step function
over time (xstep). The y-axis represents which Task is currently active
(1-4), and the x-axis represents the step number.

Usage:
    python plot_stage_progress.py
    python plot_stage_progress.py "eval_logs/shihaoran--teleavatar--organize_the_desk_stage7_*--chkpt.log"
    python plot_stage_progress.py --output my_plot.png
"""

import re
import os
import sys
import glob
import argparse
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# ── Parsing (reused from parse_stage_progress.py) ──────────────────────────

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
    """Return the task number currently Active, or None."""
    for k, v in sp.items():
        if v == "Active":
            return k
    return None


def parse_log_file(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    records = extract_xstep_records(raw)
    if not records:
        return {"file": filepath, "total_steps": 0, "transitions": [], "records": records}
    transitions = find_stage_transitions(records)
    return {
        "file": filepath,
        "total_steps": records[-1]["xstep"],
        "transitions": transitions,
        "records": records,
    }


def build_staircase(parsed: dict) -> tuple[list[int], list[int]]:
    """
    Build (steps, stages) arrays for matplotlib step plot.
    stages[i] = active task number at step steps[i], 0 = no active task.
    """
    transitions = parsed["transitions"]
    total = parsed["total_steps"]
    if not transitions:
        return [], []

    steps = []
    stages = []
    for i, t in enumerate(transitions):
        active = get_active_stage(t["stage_progress"])
        stage_val = active if active else 0
        steps.append(t["xstep"])
        stages.append(stage_val)

    steps.append(total)
    stages.append(stages[-1])
    return steps, stages


# ── Plotting ────────────────────────────────────────────────────────────────

STAGE_COLORS = {
    1: "#4C72B0",
    2: "#DD8452",
    3: "#55A868",
    4: "#C44E52",
    0: "#CCCCCC",
}

STAGE_LABELS = {
    1: "Task 1",
    2: "Task 2",
    3: "Task 3",
    4: "Task 4",
    0: "Waiting",
}


def plot_staircase_single(ax, steps, stages, title=""):
    """Staircase step plot with lines only (no filled rectangles)."""
    if not steps:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return

    for i in range(len(steps) - 1):
        x_start = steps[i]
        x_end = steps[i + 1]
        stage = stages[i]
        color = STAGE_COLORS.get(stage, "#CCCCCC")
        ax.plot([x_start, x_end], [stage, stage], color=color, linewidth=2.2,
                solid_capstyle="butt")
        if i + 1 < len(steps) - 1:
            next_stage = stages[i + 1]
            ax.plot([x_end, x_end], [stage, next_stage],
                    color="#888888", linewidth=1, linestyle="--", alpha=0.6)

    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["Task 1", "Task 2", "Task 3", "Task 4"])
    ax.set_ylim(0.2, 4.8)
    ax.set_xlim(0, steps[-1] * 1.02)
    ax.set_title(title, fontsize=9, fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description="Plot stage progress staircase diagrams.")
    parser.add_argument(
        "log_pattern", nargs="?",
        default="eval_logs/shihaoran--teleavatar--organize_the_desk_stage7_*--chkpt.log",
        help="Glob pattern for log files",
    )
    parser.add_argument("--output", "-o", default="stage_progress_plot.png",
                        help="Output image path (default: stage_progress_plot.png)")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI (default: 200)")
    args = parser.parse_args()

    log_files = sorted(glob.glob(args.log_pattern))
    if not log_files:
        print(f"No log files found matching: {args.log_pattern}")
        sys.exit(1)

    def sort_key(f):
        m = re.search(r"stage7_(\d+)", f)
        return int(m.group(1)) if m else 0
    log_files.sort(key=sort_key)

    all_parsed = []
    for fpath in log_files:
        parsed = parse_log_file(fpath)
        all_parsed.append(parsed)

    n = len(all_parsed)

    # ── Figure 1: Individual staircase subplots (grid) ──────────────────
    cols = 4
    rows = (n + cols - 1) // cols
    fig1, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 2.2),
                              squeeze=False, constrained_layout=True)
    fig1.suptitle("Stage Progress per Episode (Staircase)", fontsize=14, fontweight="bold")

    for idx, parsed in enumerate(all_parsed):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        steps, stages = build_staircase(parsed)
        ep_num = idx + 1
        total = parsed["total_steps"]
        plot_staircase_single(ax, steps, stages,
                              title=f"Ep {ep_num} ({total} steps)")
        ax.set_xlabel("Step", fontsize=7)
        ax.tick_params(labelsize=7)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    legend_handles = [mlines.Line2D([], [], color=STAGE_COLORS[s], linewidth=2.2,
                                    label=STAGE_LABELS[s]) for s in [1, 2, 3, 4]]
    fig1.legend(handles=legend_handles, loc="upper right", fontsize=9,
                ncol=4, framealpha=0.9, bbox_to_anchor=(0.98, 0.99))

    out1 = args.output.replace(".png", "_grid.png")
    fig1.savefig(out1, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved grid plot: {out1}")
    plt.close("all")


if __name__ == "__main__":
    main()
