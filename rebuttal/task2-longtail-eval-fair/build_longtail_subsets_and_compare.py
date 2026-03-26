#!/usr/bin/env python3
# usage
# python rebuttal/task2-longtail-eval-fair/build_longtail_subsets_and_compare.py --coirl-results /data/zhengxj/projects/CoIRL-AD-models/ckpts/CoIRL-AD/results.pkl --law-results /data/zhengxj/projects/CoIRL-AD-models/ckpts/LAW/results.pkl --base-val-ann /data/zhengxj/projects/CoIRL-AD/data/nuscenes/vad_nuscenes_infos_temporal_val.pkl --out-dir rebuttal/task2-longtail-eval-fair/output --l2-th-1 0.3 --l2-th-2 0.6 --l2-th-3 1.0 --out-ann-l2 rebuttal/task2-longtail-eval-fair/output/vad_nuscenes_infos_temporal_val_longtail_l2_coirl_rebuttal.pkl --out-ann-col rebuttal/task2-longtail-eval-fair/output/vad_nuscenes_infos_temporal_val_longtail_colbox_coirl_rebuttal.pkl
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def mean_metrics(records, tokens, metric_keys):
    vals = {k: [] for k in metric_keys}
    for t in tokens:
        rec = records.get(t)
        if rec is None:
            continue
        for k in metric_keys:
            vals[k].append(float(rec[k]))

    out = {}
    for k in metric_keys:
        arr = np.array(vals[k], dtype=float)
        out[k] = float(arr.mean()) if arr.size > 0 else float("nan")
    out["num_samples"] = int(len(tokens))
    return out


def filter_tokens(records, l2_th_1, l2_th_2, l2_th_3):
    l2_tokens = []
    col_tokens = []

    for token, rec in records.items():
        valid = float(rec["fut_valid_flag"]) > 0.5
        if not valid:
            continue

        if (
            float(rec["plan_L2_1s"]) > l2_th_1
            and float(rec["plan_L2_2s"]) > l2_th_2
            and float(rec["plan_L2_3s"]) > l2_th_3
        ):
            l2_tokens.append(token)

        # Long-tail collision should use plan_obj_box_col keys per user requirement.
        if float(rec["plan_obj_box_col_3s"]) > 0.0:
            col_tokens.append(token)

    return l2_tokens, col_tokens


def build_filtered_ann(base_ann, token_set):
    infos = base_ann["infos"]
    new_infos = [x for x in infos if x["token"] in token_set]
    out = dict(base_ann)
    out["infos"] = new_infos
    return out


def write_tokens(path, tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tokens:
            f.write(t + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build CoIRL-based long-tail subsets and compare with LAW")
    parser.add_argument("--coirl-results", required=True)
    parser.add_argument("--law-results", required=True)
    parser.add_argument("--base-val-ann", required=True)
    parser.add_argument("--out-dir", default="rebuttal/task2-longtail-eval/output")
    parser.add_argument("--l2-th-1", type=float, default=0.3)
    parser.add_argument("--l2-th-2", type=float, default=0.6)
    parser.add_argument("--l2-th-3", type=float, default=1.0)
    parser.add_argument(
        "--out-ann-l2",
        default="data/nuscenes/vad_nuscenes_infos_temporal_val_longtail_l2_coirl_rebuttal.pkl",
    )
    parser.add_argument(
        "--out-ann-col",
        default="data/nuscenes/vad_nuscenes_infos_temporal_val_longtail_colbox_coirl_rebuttal.pkl",
    )
    args = parser.parse_args()

    coirl_results = load_pickle(Path(args.coirl_results))
    law_results = load_pickle(Path(args.law_results))
    base_ann = load_pickle(Path(args.base_val_ann))

    l2_tokens_coirl, col_tokens_coirl = filter_tokens(
        coirl_results,
        l2_th_1=args.l2_th_1,
        l2_th_2=args.l2_th_2,
        l2_th_3=args.l2_th_3,
    )

    l2_token_law, col_tokens_law = filter_tokens(
        law_results,
        l2_th_1=args.l2_th_1,
        l2_th_2=args.l2_th_2,
        l2_th_3=args.l2_th_3,
    )

    l2_set = set(l2_tokens_coirl + l2_token_law)
    col_set = set(col_tokens_coirl + col_tokens_law)

    # Keep deterministic ordering by base ann order.
    base_order_tokens = [x["token"] for x in base_ann["infos"]]
    l2_tokens_ordered = [t for t in base_order_tokens if t in l2_set]
    col_tokens_ordered = [t for t in base_order_tokens if t in col_set]

    metric_keys = [
        "plan_L2_1s",
        "plan_L2_2s",
        "plan_L2_3s",
        "plan_obj_box_col_1s",
        "plan_obj_box_col_2s",
        "plan_obj_box_col_3s",
        "fut_valid_flag",
    ]

    summary = {
        "thresholds": {
            "l2_1s_gt": args.l2_th_1,
            "l2_2s_gt": args.l2_th_2,
            "l2_3s_gt": args.l2_th_3,
            "col_key": "plan_obj_box_col_3s > 0",
            "valid_gate": "fut_valid_flag == True",
        },
        "subset_sizes": {
            "longtail_l2": len(l2_tokens_ordered),
            "longtail_col": len(col_tokens_ordered),
        },
        "metrics": {
            "longtail_l2": {
                "coirl": mean_metrics(coirl_results, l2_tokens_ordered, metric_keys),
                "law": mean_metrics(law_results, l2_tokens_ordered, metric_keys),
            },
            "longtail_col": {
                "coirl": mean_metrics(coirl_results, col_tokens_ordered, metric_keys),
                "law": mean_metrics(law_results, col_tokens_ordered, metric_keys),
            },
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_tokens(out_dir / "tokens_longtail_l2.txt", l2_tokens_ordered)
    write_tokens(out_dir / "tokens_longtail_colbox.txt", col_tokens_ordered)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Flat CSV for easy rebuttal table drafting.
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "model", *metric_keys, "num_samples"])
        for subset in ["longtail_l2", "longtail_col"]:
            for model in ["coirl", "law"]:
                row = summary["metrics"][subset][model]
                writer.writerow([subset, model, *[row[k] for k in metric_keys], row["num_samples"]])

    # Build filtered val annotation files for direct eval usage.
    l2_ann = build_filtered_ann(base_ann, set(l2_tokens_ordered))
    col_ann = build_filtered_ann(base_ann, set(col_tokens_ordered))
    save_pickle(Path(args.out_ann_l2), l2_ann)
    save_pickle(Path(args.out_ann_col), col_ann)

    print("Done.")
    print(f"longtail_l2_count: {len(l2_tokens_ordered)}")
    print(f"longtail_col_count: {len(col_tokens_ordered)}")
    print(f"summary_json: {out_dir / 'summary.json'}")
    print(f"summary_csv: {out_dir / 'summary.csv'}")
    print(f"out_ann_l2: {Path(args.out_ann_l2)}")
    print(f"out_ann_col: {Path(args.out_ann_col)}")


if __name__ == "__main__":
    main()
