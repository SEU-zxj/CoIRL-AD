# Task3 CQL Shape-Safe Plan

## Goal
Refine CQL into a mathematically consistent single-Q trajectory-action baseline for rebuttal experiments.

## Locked Decisions
1. Action granularity: trajectory-level action `a_traj`.
2. Reward: constant `1.0` baseline.
3. Validity: use only `ego_fut_masks`; sample is valid only if all horizon steps are valid.
4. Remove `fut_valid_t` dependency from CQL path.
5. Keep EMA target critic for stabilization (single-Q, not double-Q).
6. Actor objective: keep `Q * logpi` form.
7. Actor update uses `s_t` only.
8. Control variables aligned with CoIRL:
- `rl_actor_use_bc=True`
- `disable_competition=False`

## Implementation Scope
1. In `CoIRL.py`, make all CQL values trajectory-level:
- `q_data: [B]`
- `q_next: [B]`
- `q_policy/q_random: [B, K]`
- `cql_lse: [B]`
2. Use trajectory-level mask `valid_traj: [B]` built from `ego_fut_masks` only.
3. Ensure `td_target` is `[B]` and all masked reductions operate on `[B]`.
4. Keep IL path unchanged.
5. Add concise shape comments and assertions for debug readability.

## Verification
1. `py_compile` on:
- `projects/mmdet3d_plugin/CoIRL/CoIRL.py`
- `projects/mmdet3d_plugin/CoIRL/utils/cql_critic.py`
2. CQL config sanity:
- `projects/configs/coirl/coirl_cql.py`
3. Runtime smoke train:
- confirm no shape errors
- confirm finite CQL logs
- check `cql_valid_ratio` under strict all-valid gating
