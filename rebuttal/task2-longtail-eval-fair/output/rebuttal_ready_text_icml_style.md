# Fair Long-Tail Evaluation Under Symmetric Subset Construction

To address potential selection bias in long-tail analysis, we reconstructed long-tail subsets using a symmetric protocol: we first identify hard scenarios from both CoIRL and LAW under the same criteria, then evaluate both methods on the union of those scenarios. This avoids favoring either method in subset definition.

For long-tail L2, we use `fut_valid_flag=True` and `L2@1s>0.3`, `L2@2s>0.6`, `L2@3s>1.0`. For long-tail collision, we use `fut_valid_flag=True` and `plan_obj_box_col_3s>0`. The resulting fair subsets contain 2247 scenes (L2) and 124 scenes (collision).

## Fair Subset Summary

- Merge policy: union of hard cases from CoIRL and LAW
- Long-tail L2: 2247 scenes
- Long-tail collision: 124 scenes
- L2 source breakdown: CoIRL-only 614, LAW-only 807, overlap 826
- Collision source breakdown: CoIRL-only 33, LAW-only 46, overlap 45

## Results on Fair Subsets

| Subset | Model | L2@1s | L2@2s | L2@3s | Box-Col@1s | Box-Col@2s | Box-Col@3s | N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Long-tail L2 | CoIRL | 0.4533 | 0.9235 | 1.5273 | 0.0007 | 0.0014 | 0.0053 | 2247 |
| Long-tail L2 | LAW | 0.5160 | 1.0054 | 1.6001 | 0.0011 | 0.0018 | 0.0074 | 2247 |
| Long-tail Collision | CoIRL | 0.3718 | 0.8408 | 1.5074 | 0.0242 | 0.0423 | 0.1519 | 124 |
| Long-tail Collision | LAW | 0.4030 | 0.9428 | 1.6729 | 0.0363 | 0.0504 | 0.1882 | 124 |

Under this symmetric subset construction, CoIRL achieves lower planning L2 and lower box-collision rates than LAW on both long-tail subsets.

## Reproducibility

- Subset builder script: `rebuttal/task2-longtail-eval-fair/build_longtail_subsets_and_compare_fair.py`
- Summary JSON: `rebuttal/task2-longtail-eval-fair/output/summary_fair.json`
- Summary CSV: `rebuttal/task2-longtail-eval-fair/output/summary_fair.csv`
- Fair long-tail L2 annotations: `rebuttal/task2-longtail-eval-fair/output/vad_nuscenes_infos_temporal_val_longtail_l2_fair.pkl`
- Fair long-tail collision annotations: `rebuttal/task2-longtail-eval-fair/output/vad_nuscenes_infos_temporal_val_longtail_colbox_fair.pkl`
