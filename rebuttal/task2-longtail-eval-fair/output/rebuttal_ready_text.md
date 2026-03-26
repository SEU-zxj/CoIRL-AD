# Fair Long-Tail Evaluation (Union of Hard Cases from Both Models)

To avoid selection bias in long-tail analysis, we constructed fair subsets by identifying hard scenarios from both methods (CoIRL and LAW) under the same criteria, then taking the union of those scenarios and evaluating both methods on exactly the same subset. Specifically, for long-tail L2 we used `fut_valid_flag=True` and `L2@1s>0.3`, `L2@2s>0.6`, `L2@3s>1.0`; for long-tail collision we used `fut_valid_flag=True` and `plan_obj_box_col_3s>0`. This yields 2247 scenarios for long-tail L2 and 124 scenarios for long-tail collision. On both fair subsets, CoIRL outperforms LAW on all L2 horizons and on box-collision metrics.

## Subset Construction Summary

- Merge mode: `union` (hard cases from CoIRL or LAW)
- Long-tail L2 subset size: `2247`
- Long-tail collision subset size: `124`
- L2 source breakdown: CoIRL-only `614`, LAW-only `807`, overlap `826`
- Collision source breakdown: CoIRL-only `33`, LAW-only `46`, overlap `45`

## Fair Comparison Table

| Subset | Model | L2@1s | L2@2s | L2@3s | Box-Col@1s | Box-Col@2s | Box-Col@3s | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Long-tail L2 | CoIRL | 0.4533 | 0.9235 | 1.5273 | 0.0007 | 0.0014 | 0.0053 | 2247 |
| Long-tail L2 | LAW | 0.5160 | 1.0054 | 1.6001 | 0.0011 | 0.0018 | 0.0074 | 2247 |
| Long-tail Collision | CoIRL | 0.3718 | 0.8408 | 1.5074 | 0.0242 | 0.0423 | 0.1519 | 124 |
| Long-tail Collision | LAW | 0.4030 | 0.9428 | 1.6729 | 0.0363 | 0.0504 | 0.1882 | 124 |

## Reproducibility

- Script: `rebuttal/task2-longtail-eval-fair/build_longtail_subsets_and_compare_fair.py`
- Main output json: `rebuttal/task2-longtail-eval-fair/output/summary_fair.json`
- Main output csv: `rebuttal/task2-longtail-eval-fair/output/summary_fair.csv`
- Filtered L2 pkl: `rebuttal/task2-longtail-eval-fair/output/vad_nuscenes_infos_temporal_val_longtail_l2_fair.pkl`
- Filtered Collision pkl: `rebuttal/task2-longtail-eval-fair/output/vad_nuscenes_infos_temporal_val_longtail_colbox_fair.pkl`
