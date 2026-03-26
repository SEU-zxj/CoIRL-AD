#!/usr/bin/env python3
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
    used = 0
    for token in tokens:
        rec = records.get(token)
        if rec is None:
            continue
        used += 1
        for k in metric_keys:
            vals[k].append(float(rec[k]))

    out = {"num_samples": int(used)}
    for k in metric_keys:
        arr = np.array(vals[k], dtype=float)
        out[k] = float(arr.mean()) if arr.size > 0 else float("nan")
    return out


def get_valid(rec):
    return float(rec["fut_valid_flag"]) > 0.5


def get_l2_tail(rec, th1, th2, th3):
    return (
        get_valid(rec)
        and float(rec["plan_L2_1s"]) > th1
        and float(rec["plan_L2_2s"]) > th2
        and float(rec["plan_L2_3s"]) > th3
    )


def get_col_tail(rec):
    # User requirement: always use plan_obj_box_col for long-tail collision.
    return get_valid(rec) and float(rec["plan_obj_box_col_3s"]) > 0.0


def merge_token_set(coirl_set, law_set, mode):
    if mode == "union":
        return coirl_set | law_set
    if mode == "intersection":
        return coirl_set & law_set
    raise ValueError(f"Unsupported mode: {mode}")


def classify_sources(final_set, coirl_set, law_set):
    coirl_only = len(final_set - law_set)
    law_only = len(final_set - coirl_set)
    both = len(final_set & coirl_set & law_set)
    return {
        "coirl_only": int(coirl_only),
        "law_only": int(law_only),
        "both": int(both),
    }


def build_filtered_ann(base_ann, token_set):
    infos = base_ann["infos"]
    new_infos = [x for x in infos if x["token"] in token_set]
    out = dict(base_ann)
    out["infos"] = new_infos
    return out


def write_tokens(path, tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(token + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build fair long-tail subsets from both CoIRL and LAW results, then compare metrics."
    )
    parser.add_argument("--coirl-results", required=True)
    parser.add_argument("--law-results", required=True)
    parser.add_argument("--base-val-ann", required=True)
    parser.add_argument("--out-dir", default="rebuttal/task2-longtail-eval-fair/output")

    parser.add_argument("--l2-th-1", type=float, default=0.3)
    parser.add_argument("--l2-th-2", type=float, default=0.6)
    parser.add_argument("--l2-th-3", type=float, default=1.0)

    parser.add_argument(
        "--merge-mode",
        choices=["union", "intersection"],
        default="union",
        help="How to merge bad-scenario sets from CoIRL and LAW.",
    )

    parser.add_argument(
        "--out-ann-l2",
        default="rebuttal/task2-longtail-eval-fair/output/vad_nuscenes_infos_temporal_val_longtail_l2_fair.pkl",
    )
    parser.add_argument(
        "--out-ann-col",
        default="rebuttal/task2-longtail-eval-fair/output/vad_nuscenes_infos_temporal_val_longtail_colbox_fair.pkl",
    )

    args = parser.parse_args()

    coirl_records = load_pickle(Path(args.coirl_results))
    law_records = load_pickle(Path(args.law_results))
    base_ann = load_pickle(Path(args.base_val_ann))

    common_tokens = set(coirl_records.keys()) & set(law_records.keys())

    coirl_l2 = set()
    law_l2 = set()
    coirl_col = set()
    law_col = set()

    for token in common_tokens:
        rec_c = coirl_records[token]
        rec_l = law_records[token]

        if get_l2_tail(rec_c, args.l2_th_1, args.l2_th_2, args.l2_th_3):
            coirl_l2.add(token)
        if get_l2_tail(rec_l, args.l2_th_1, args.l2_th_2, args.l2_th_3):
            law_l2.add(token)

        if get_col_tail(rec_c):
            coirl_col.add(token)
        if get_col_tail(rec_l):
            law_col.add(token)

    fair_l2 = merge_token_set(coirl_l2, law_l2, args.merge_mode)
    fair_col = merge_token_set(coirl_col, law_col, args.merge_mode)

    base_order = [x["token"] for x in base_ann["infos"]]
    fair_l2_ordered = [t for t in base_order if t in fair_l2]
    fair_col_ordered = [t for t in base_order if t in fair_col]

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
        "config": {
            "merge_mode": args.merge_mode,
            "l2_thresholds": {
                "plan_L2_1s_gt": args.l2_th_1,
                "plan_L2_2s_gt": args.l2_th_2,
                "plan_L2_3s_gt": args.l2_th_3,
            },
            "col_threshold": "plan_obj_box_col_3s > 0",
            "valid_gate": "fut_valid_flag == True",
            "common_tokens": len(common_tokens),
        },
        "subset_sizes": {
            "longtail_l2": len(fair_l2_ordered),
            "longtail_col": len(fair_col_ordered),
        },
        "source_breakdown": {
            "longtail_l2": classify_sources(fair_l2, coirl_l2, law_l2),
            "longtail_col": classify_sources(fair_col, coirl_col, law_col),
        },
        "metrics": {
            "longtail_l2": {
                "coirl": mean_metrics(coirl_records, fair_l2_ordered, metric_keys),
                "law": mean_metrics(law_records, fair_l2_ordered, metric_keys),
            },
            "longtail_col": {
                "coirl": mean_metrics(coirl_records, fair_col_ordered, metric_keys),
                "law": mean_metrics(law_records, fair_col_ordered, metric_keys),
            },
        },
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_tokens(out_dir / "tokens_longtail_l2_fair.txt", fair_l2_ordered)
    write_tokens(out_dir / "tokens_longtail_colbox_fair.txt", fair_col_ordered)

    with open(out_dir / "summary_fair.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "summary_fair.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "model", *metric_keys, "num_samples"])
        for subset in ["longtail_l2", "longtail_col"]:
            for model in ["coirl", "law"]:
                row = summary["metrics"][subset][model]
                writer.writerow([subset, model, *[row[k] for k in metric_keys], row["num_samples"]])

    l2_ann = build_filtered_ann(base_ann, set(fair_l2_ordered))
    col_ann = build_filtered_ann(base_ann, set(fair_col_ordered))
    save_pickle(Path(args.out_ann_l2), l2_ann)
    save_pickle(Path(args.out_ann_col), col_ann)

    print("Done fair filtering.")
    print(f"merge_mode: {args.merge_mode}")
    print(f"longtail_l2_count: {len(fair_l2_ordered)}")
    print(f"longtail_col_count: {len(fair_col_ordered)}")
    print(f"summary_json: {out_dir / 'summary_fair.json'}")
    print(f"summary_csv: {out_dir / 'summary_fair.csv'}")
    print(f"out_ann_l2: {Path(args.out_ann_l2)}")
    print(f"out_ann_col: {Path(args.out_ann_col)}")


if __name__ == "__main__":
    main()
