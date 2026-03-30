## Plan: PPO-Lite via World Model

Implement a lightweight on-policy PPO-style branch tailored to this codebase: no replay buffer, no importance ratio/clip/KL, single full-trajectory sampling (no GRPO group sampling and no step-aware construction), TD(1)-style advantage using world-model predicted next state, and value critic reuse from current GRPO path. Each trajectory [T, 2] is treated as one action, so advantage is a scalar per sample. This is intentionally a pragmatic actor-critic baseline for rebuttal, not textbook PPO.

**Steps**
1. Define PPO-lite scope and branch dispatch in model (*blocks all later steps*).
2. Add PPO-lite actor/critic loss path in RL head using world-model one-step transition (*depends on 1*).
3. Integrate PPO-lite call path in training loop with preserved GRPO/CQL behavior (*depends on 2*).
4. Add PPO config variant and stability defaults (*parallel with 3 once interfaces are fixed*).
5. Verify numerics and non-regression (*depends on 3 and 4*).

**Phase 1: Scope Lock and Interfaces**
1. Add/confirm rl_method dispatch value PPO in [projects/mmdet3d_plugin/CoIRL/CoIRL.py](projects/mmdet3d_plugin/CoIRL/CoIRL.py).
2. Explicitly define PPO-lite semantics:
- On-policy, no replay buffer.
- No old-policy ratio, no clipping, no KL penalty.
- Single full-trajectory sampling from policy distribution.
- Treat one trajectory [T, 2] as one action.
- TD(1)-style scalar advantage: A = r + gamma * V(s_hat_next) - V(s).
- s_hat_next predicted by world model from s and sampled action.
3. Keep bs==1 assumption for first pass.

**Phase 2: RL Head PPO-lite Losses**
1. In [projects/mmdet3d_plugin/CoIRL/dense_heads/waypoint_query_decoder.py](projects/mmdet3d_plugin/CoIRL/dense_heads/waypoint_query_decoder.py), add a new PPO-lite loss method (separate from GRPO functions).
2. Sampling:
- Draw one full trajectory from current policy: shape [B, T, 2].
- Do not use group sampling.
- Do not use step-aware group construction used by GRPO.
3. World-model transition:
- Use existing world-model rollout utility to get predicted next latent state for the sampled trajectory.
4. Critic/value:
- Reuse existing value critic V(s) and refer/target critic usage pattern already present in RL head.
5. Reward:
- Reuse current GRPO reward components (collision, drivable area, imitation).
6. Advantage:
- Compute scalar A using TD(1): r + gamma * V(s_hat_next) - V(s).
7. Actor loss:
- Use policy-gradient style loss with trajectory log-prob and detached scalar advantage.
8. Critic loss:
- Reuse value regression to scalar TD target without mask-based reduction.

**Phase 3: Model Integration**
1. In [projects/mmdet3d_plugin/CoIRL/CoIRL.py](projects/mmdet3d_plugin/CoIRL/CoIRL.py), add PPO branch in forward_pts_train alongside existing GRPO and CQL branches.
2. Route PPO branch to the new RL-head PPO-lite loss method.
3. Preserve existing IL/world-model losses and competition logic behavior unless explicitly disabled by config.
4. Ensure GRPO and CQL branches remain untouched for backward compatibility.

**Phase 4: Config and Stability Controls**
1. Add new config [projects/configs/coirl/coirl_ppo.py](projects/configs/coirl/coirl_ppo.py) extending base coirl config.
2. Set defaults per your decisions:
- rl_method = PPO.
- rl_actor_use_bc = True.
- keep competition setting consistent with baseline comparison protocol.
3. Add PPO-specific knobs in config (with conservative defaults):
- ppo_adv_gamma.
- optional reduced max std clamp in policy head path.
- optional actor/critic loss weights if separate tuning needed.

**Phase 5: Verification**
1. Static checks:
- compile [projects/mmdet3d_plugin/CoIRL/CoIRL.py](projects/mmdet3d_plugin/CoIRL/CoIRL.py)
- compile [projects/mmdet3d_plugin/CoIRL/dense_heads/waypoint_query_decoder.py](projects/mmdet3d_plugin/CoIRL/dense_heads/waypoint_query_decoder.py)
- compile [projects/configs/coirl/coirl_ppo.py](projects/configs/coirl/coirl_ppo.py)
2. Runtime smoke run:
- confirm no DDP unused-parameter errors.
- confirm finite losses and gradients.
- confirm std does not saturate immediately after applying lower max-std setting.
3. Logging sanity:
- add/read PPO diagnostics: ppo_reward_mean, ppo_adv_mean, ppo_actor_loss, ppo_critic_loss.
4. Non-regression:
- run one quick GRPO sanity to ensure unchanged behavior.

**Relevant files**
- [projects/mmdet3d_plugin/CoIRL/CoIRL.py](projects/mmdet3d_plugin/CoIRL/CoIRL.py) — rl_method dispatch and PPO branch wiring.
- [projects/mmdet3d_plugin/CoIRL/dense_heads/waypoint_query_decoder.py](projects/mmdet3d_plugin/CoIRL/dense_heads/waypoint_query_decoder.py) — PPO-lite sampling, reward, advantage, actor/critic losses.
- [projects/configs/coirl/coirl_ppo.py](projects/configs/coirl/coirl_ppo.py) — PPO experiment config.
- [projects/mmdet3d_plugin/CoIRL/utils/reward_function.py](projects/mmdet3d_plugin/CoIRL/utils/reward_function.py) — reused reward components.

**Decisions**
- Included:
- PPO-lite (on-policy actor-critic) with world-model next-state prediction.
- No replay buffer.
- No importance ratio/clip/KL.
- No GRPO group sampling.
- No GRPO step-aware group construction.
- Reuse value critic and reward components from current GRPO stack.
- bs==1 assumption retained for first pass.

- Excluded:
- Textbook PPO ratio clipping and old-policy snapshots.
- GAE(lambda) for this first PPO pass.

**Further Considerations**
1. This branch should be described as PPO-lite actor-critic in internal notes to avoid algorithm naming ambiguity.
2. If instability appears, first fallback is lowering std max and actor loss weight before adding ratio/clip complexity.
3. If reviewer rigor requires textbook PPO, phase-2 extension can add old-policy ratio+clip after PPO-lite baseline results.
