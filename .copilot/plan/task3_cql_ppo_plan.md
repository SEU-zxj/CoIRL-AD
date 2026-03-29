# Task 3 Plan: Offline RL Comparison (CQL First, PPO Second)

Date: 2026-03-27
Branch: icml_rebuttal only
Owner intent: keep previous GRPO/Task1/Task2 behavior unchanged.

## Goal
Implement a new RL method switch (`rl_method`) so we can compare:
- existing method: GRPO (default, unchanged)
- new baseline: CQL (first implementation target)
- optional follow-up: PPO (after CQL sanity pass)

## Confirmed Design Decisions
1. Implement CQL first, PPO later.
2. CQL first version is single-Q baseline (not twin-Q yet).
3. Keep actor (`WaypointHead_RL`) and actor update under CQL.
4. Use observed next-frame latent transitions (offline-style) rather than world-model-predicted next state for CQL targets.
5. Preserve full backward compatibility: GRPO path and no-competition ablations must remain intact.
6. For CQL baseline, use strict constant reward (`r = 1`) instead of rule-based rewards.

## High-Level Strategy
1. Add `rl_method` config plumbing with default `GRPO`.
2. Build transition extraction for offline tuples `(s_t, a_t, r_t, s_{t+1}, mask)` from temporal queue.
3. Add CQL branch (single-Q) without touching GRPO equations.
4. Add CQL-specific logs and config.
5. Run short sanity, then training/eval.
6. Only after CQL is stable, add PPO branch.

## Phase A: Non-Regressive Plumbing
### A1. Add `rl_method` switch
- Primary file: `projects/mmdet3d_plugin/CoIRL/CoIRL.py`
- Add model arg: `rl_method='GRPO'`.
- Route RL logic in `forward_pts_train` by method:
  - `GRPO`: existing code path untouched.
  - `CQL`: new branch.
  - `PPO`: placeholder/error until Phase C.

### A2. Config variants
- Keep existing configs unchanged.
- Add new config files under `projects/configs/coirl/`:
  - `coirl_cql.py`
  - `coirl_ppo.py` (placeholder for later activation)

## Phase B: CQL Implementation (Single-Q Baseline)
### B1. Transition extraction (offline tuples)
- Source function context: `forward_train`, `obtain_history_feat`, `forward_pts_train` in `projects/mmdet3d_plugin/CoIRL/CoIRL.py`.
- Build aligned tuples using observed temporal sequence:
  - `s_t`: previous frame RL latent
  - `a_t`: dataset trajectory label (`ego_fut_trajs` with masks)
  - `s_{t+1}`: current frame RL latent
  - `mask`: `fut_valid_flag` and trajectory mask
  - `r_t`: strict constant reward (`r = 1`), mask-aware
- Add strong assertions for temporal alignment and shapes.

### B2. Q critic module
- Add new file (do not modify existing value critic):
  - `projects/mmdet3d_plugin/CoIRL/utils/cql_critic.py`
- Keep existing `projects/mmdet3d_plugin/CoIRL/utils/critic.py` for GRPO.
- Single-Q baseline:
  - `Q(s,a)` network consumes RL latent state tokens + encoded trajectory action.
  - target network / EMA copy optional for first pass (recommended if stable).

### B3. CQL losses
- Critic loss = TD loss + conservative penalty.
- TD target:
  - `r + gamma * E_{a'~pi(.|s')} [Q_target(s', a')]` (sample approximation).
  - with this plan, `r` is fixed to 1 for valid transitions.
- Conservative penalty:
  - increase `Q` on dataset actions relative to policy/random sampled actions.
  - practical single-Q form with log-sum-exp term and weight `cql_alpha`.
- Actor loss:
  - maximize `Q(s, a_pi)` with stabilization term if needed.

### B4. Logging for diagnosis
- Add CQL logs to `losses` for tensorboard:
  - `cql_td_loss`
  - `cql_penalty`
  - `cql_q_data_mean`
  - `cql_q_policy_mean`
  - `cql_gap`
  - `cql_actor_loss`

## Phase C: PPO (Deferred)
### C1. Activation condition
- Start only if CQL short-run is stable and produces valid comparisons.

### C2. PPO branch
- Add PPO-specific actor objective using policy ratio/clipping.
- Use no replay buffer, as requested.
- Keep separated under `rl_method == 'PPO'`.

## Files Expected to Change
1. `projects/mmdet3d_plugin/CoIRL/CoIRL.py`
2. `projects/mmdet3d_plugin/CoIRL/dense_heads/waypoint_query_decoder.py`
3. `projects/mmdet3d_plugin/CoIRL/utils/cql_critic.py` (new)
4. `projects/configs/coirl/coirl_cql.py` (new)
5. `projects/configs/coirl/coirl_ppo.py` (new placeholder now, implementation later)

## Validation Checklist
1. GRPO regression test:
- Existing `coirl.py` run behavior unchanged.

2. CQL smoke test:
- No NaN in losses.
- CQL logs appear in tensorboard.
- Shape assertions all pass.

3. CQL short training:
- Stable loss trend over first epochs.
- Comparable output metrics produced by existing test scripts.

4. Reproducibility:
- Save commands/configs under rebuttal notes.

## Risks and Mitigations
1. Temporal alignment bugs for observed `s_{t+1}`:
- Mitigation: explicit frame-index assertions and mask checks.

2. Single-Q overestimation/instability:
- Mitigation: conservative weight sweep, optional target network EMA.

5. Constant reward may weaken task semantics and reduce absolute performance:
- Mitigation: explicitly frame this as an offline-RL ablation baseline intended to test whether GRPO with richer feedback is advantageous.

3. Batch-size assumptions (`bs==1`) in existing RL helpers:
- Mitigation: preserve assumptions first; avoid broad refactor in initial CQL pass.

4. Side effects on existing code:
- Mitigation: strict `rl_method` dispatch, default stays GRPO.

## Execution Order
1. Add `rl_method` plumbing + config scaffold.
2. Add transition extraction helpers.
3. Add CQL critic and CQL branch losses.
4. Add logs and run smoke test.
5. If stable, run full CQL experiment.
6. Start PPO phase afterward.
