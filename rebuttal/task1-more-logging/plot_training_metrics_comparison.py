#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path("/data/zhengxj/projects/CoIRL-AD")
OUTPUT_DIR = BASE_DIR / "rebuttal/task1-more-logging"

EXP_WITH_COMP = {
    "label": "With competition",
    "files": [
        BASE_DIR / "work_dirs/coirl/rebuttal-task1-more-logging/20260326_154833.log.json",
    ],
}

EXP_NO_COMP = {
    "label": "Without competition",
    "files": [
        BASE_DIR / "work_dirs/coirl/rebuttal-task1-more-logging-no-comp/20260327_095313.log.json",
        BASE_DIR / "work_dirs/coirl/rebuttal-task1-more-logging-no-comp/20260328_230141.log.json",
    ],
}

METRICS = [
    "loss_rec",
    "debug_il_rl_mode_traj_l2",
    "debug_critic_cur_value_mean",
    "debug_critic_pred_fut_value_mean",
]


def parse_train_records(log_json_path: Path) -> List[dict]:
    records: List[dict] = []
    with log_json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("mode") != "train":
                continue
            if "epoch" not in item or "iter" not in item:
                continue
            records.append(item)
    return records


def infer_iters_per_epoch(records: List[dict], fallback: int = 3517) -> int:
    max_iter = 0
    for rec in records:
        it = rec.get("iter")
        if isinstance(it, int):
            max_iter = max(max_iter, it)
    if max_iter <= 0:
        return fallback
    # Iteration is logged every 20 steps, so max logged iter can be slightly below true per-epoch size.
    return max(max_iter, fallback)


def to_series(records: List[dict], metric: str, iters_per_epoch: int) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for rec in records:
        if metric not in rec:
            continue
        epoch = rec.get("epoch")
        it = rec.get("iter")
        value = rec.get(metric)
        if not isinstance(epoch, int) or not isinstance(it, int):
            continue
        if not isinstance(value, (int, float)):
            continue
        global_iter = (epoch - 1) * iters_per_epoch + it
        xs.append(global_iter)
        ys.append(float(value))
    return xs, ys


def build_no_comp_records() -> List[dict]:
    run1 = parse_train_records(EXP_NO_COMP["files"][0])
    run2 = parse_train_records(EXP_NO_COMP["files"][1])

    # Keep run1 only up to epoch 15 as requested; keep run2 from epoch 16 onward.
    run1 = [r for r in run1 if isinstance(r.get("epoch"), int) and r["epoch"] <= 15]
    run2 = [r for r in run2 if isinstance(r.get("epoch"), int) and r["epoch"] >= 16]
    return run1 + run2


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with_comp_records = parse_train_records(EXP_WITH_COMP["files"][0])
    no_comp_records = build_no_comp_records()

    all_records = with_comp_records + no_comp_records
    iters_per_epoch = infer_iters_per_epoch(all_records)

    for metric in METRICS:
        x_with, y_with = to_series(with_comp_records, metric, iters_per_epoch)
        x_no, y_no = to_series(no_comp_records, metric, iters_per_epoch)

        plt.figure(figsize=(10, 5), dpi=150)
        plt.plot(x_with, y_with, label=EXP_WITH_COMP["label"], linewidth=1.1)
        plt.plot(x_no, y_no, label=EXP_NO_COMP["label"], linewidth=1.1)
        plt.xlabel("Global iteration")
        plt.ylabel(metric)
        plt.title(f"Training Curve: {metric}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = OUTPUT_DIR / f"{metric}_comparison.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
