# ICML Rebuttal Execution Plan (CoIRL-AD)

Date: 2026-03-26
Branch policy: all edits and experiments must stay on `icml_rebuttal`.

## Conversation Context Snapshot
- User is preparing ICML rebuttal experiments for CoIRL-AD.
- Workspace root: `/data/zhengxj/projects/CoIRL-AD`.
- Confirmed branch in repo: `icml_rebuttal`.
- Confirmed checkpoints exist:
  - CoIRL: `/data/zhengxj/projects/CoIRL-AD-models/ckpts/CoIRL-AD/epoch_21.pth`
  - LAW: `/data/zhengxj/projects/CoIRL-AD-models/ckpts/LAW/epoch_15.pth`
- Environment detail from user:
  - Conda root: `/data/zhengxj/softwares/anaconda3`
  - Env name: `coirl`
- Env check succeeded in this session:
  - python executable in env: `/data/zhengxj/softwares/anaconda3/envs/coirl/bin/python`
  - `mmcv` import works (`1.4.0`).
- CoIRL eval smoke test succeeded (checkpoint loaded and evaluation progressed with expected metrics, matching paper trend).
- LAW eval had a launch conflict at least once due to distributed port collision (`Address already in use`) while another eval was active.

## Rebuttal Tasks (Agreed)
1. Add richer training logs (TensorBoard):
   - critic output
   - L2 error between IL actor output and RL actor mode trajectory
   Then retrain and visualize these curves.
2. Redo long-tail evaluation with unbiased split construction:
   - Previously provided long-tail pkl files were based on baseline-driven filtering.
   - New requirement: re-filter eval set based on CoIRL performance, then evaluate both CoIRL and LAW on those new splits.
3. Offline RL comparison:
   - First priority: replace GRPO with CQL and compare.
   - Optional (time permitting): PPO variant (no replay buffer) and compare expected ordering: GRPO > PPO > CQL.

## Planned Execution Order
1. Task 1 first: implement logging and verify in TensorBoard.
2. Task 2 second: regenerate CoIRL-filtered long-tail splits and run CoIRL vs LAW evaluation on same splits.
3. Task 3 third: implement CQL variant and run comparison; optional PPO after core results.

## Technical Plan Details

### A) Task 1: Logging Implementation
- Primary edit target:
  - `projects/mmdet3d_plugin/CoIRL/CoIRL.py`
- In `forward_pts_train` (where IL outputs, RL policy outputs, and critic values are available), add scalar logs to `losses/log_vars` so MMCV `TensorboardLoggerHook` records them.
- Suggested tags:
  - `debug/critic_value_mean`
  - `debug/il_rl_mode_traj_l2`
- Metric definitions:
  - critic mean: mean over current critic value tensor
  - IL-RL trajectory L2: mean of `||traj_il - traj_rl_mode||_2` over horizon and batch
- Verify by short training run and checking TensorBoard event tags.

### B) Task 2: CoIRL-based Long-tail Re-filter + Eval
- Build a deterministic script under `tools/` to:
  - read base val annotations (`vad_nuscenes_infos_temporal_val.pkl`)
  - read CoIRL result file/metrics
  - apply `fut_valid_flag=True` gating
  - create two splits:
    - L2 long-tail (threshold-driven on 1s/2s/3s)
    - collision long-tail (non-zero 3s collision)
  - write new pkl files with new names (do not overwrite old files)
  - output scene counts and summary stats
- Evaluate both models on each new split with same `ann_file`:
  - CoIRL config and LAW config adjusted via `ann_file` or config variants
- Save outputs under rebuttal directories for direct comparison tables.

### C) Task 3: GRPO to CQL (and optional PPO)
- Identify RL update block in CoIRL training path and make algorithm selectable via config.
- Implement CQL objective variant with minimal disruption to existing GRPO path.
- Add new config variant for CQL experiments.
- Run train/eval and compare with GRPO.
- Optional: add PPO variant without replay buffer if schedule allows.

## Command/Run Notes
- Always activate env before train/test:
  - `source /data/zhengxj/softwares/anaconda3/etc/profile.d/conda.sh && conda activate coirl`
- Use unique distributed port for concurrent tests to avoid collisions.
- Keep outputs under rebuttal-specific directories (for traceability and easy table generation).

## Risks and Controls
- Risk: distributed port collision during test.
  - Control: set different master ports or run tests sequentially.
- Risk: logging tensors not scalar-safe.
  - Control: explicitly reduce to scalar with `.mean()` and check for NaN.
- Risk: split generation mismatch between results and annotation order.
  - Control: key by unique sample token/scene token and assert alignment counts.

## Next Immediate Actions
1. Implement Task 1 logging edits in CoIRL model file.
2. Run short sanity training and verify TensorBoard tags.
3. Implement Task 2 split-regeneration script and run first pass on CoIRL outputs.

---
This file is intentionally written as a context handoff artifact for a new agent/session if needed.
