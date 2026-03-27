# Task 1 Ablation Plan: Disable Competitive Sharing (Keep Critic EMA)

Date: 2026-03-27
Branch policy: keep all edits and runs on `icml_rebuttal`.

## Goal
Run only one new ablation experiment for Task 1:
- Disable IL<->RL knowledge/parameter sharing from CompetitiveLearningMachine.
- Keep critic training and reference-critic EMA update behavior unchanged.

This isolates the effect of competitive sharing on critic value dynamics for rebuttal analysis.

## Scope Decision (Locked)
- Include: disable competition-triggered actor swapping only.
- Keep: `set_refer_critic()` update path active each iteration when `use_critic=True`.
- Exclude: baseline training (handled by another agent), Task 2/Task 3 changes.

## Why Not Just Comment One Line
`CompetitiveLearningMachine` currently handles two responsibilities:
1. actor sharing/competition (`competition()` / `swap_knowledge()`)
2. reference critic EMA (`set_refer_critic()`)

Commenting the wrong call can also remove refer-critic updates and change RL target dynamics, which would confound the ablation.

## Implementation Plan

### 1) Add a config-controlled ablation switch
- Add a model-level flag (recommended name: `disable_competitive_swap=False`) in:
  - `projects/mmdet3d_plugin/CoIRL/CoIRL.py` (`CoIRL.__init__`)
- Pass this flag into CLM construction.

### 2) Gate only the sharing path inside CLM
- Edit:
  - `projects/mmdet3d_plugin/CoIRL/utils/competitive_learning_machine.py`
- Keep this path active:
  - `if self.use_critic: self.set_refer_critic()`
- Gate this path by flag:
  - periodic `self.competition()` call (the one that triggers swap behavior)
- Return logging dict keys consistently so existing logger parsing is not broken.

### 3) Add ablation config file
- Create:
  - `projects/configs/coirl/coirl_no_competition.py`
- Inherit from base CoIRL config and only override the new flag:
  - `model = dict(disable_competitive_swap=True)`

### 4) Run only ablation training
- Use same hyperparameters / seed / GPU setup convention as baseline run from the other agent.
- Use a distinct work_dir name, for example:
  - `work_dirs/coirl_rebuttal/no_competition_ablation`

## Runtime Notes
- Activate conda env before run.
- Avoid distributed port collision by explicitly setting a unique port when needed.

## Minimal Validation Checklist
1. Sanity-run starts without config/build errors.
2. TensorBoard still logs critic tags:
   - `debug_critic_cur_value_mean`
   - `debug_critic_pred_fut_value_mean`
3. No periodic swap side effects in ablation run.
4. Refer-critic path remains active when `use_critic=True`.

## Comparison Output (with baseline from other agent)
After ablation finishes, compare in one figure per metric:
- baseline curve (sharing enabled)
- ablation curve (sharing disabled)

Priority metrics:
- `debug_critic_cur_value_mean`
- `debug_critic_pred_fut_value_mean`
Optional support metric:
- `debug_il_rl_mode_traj_l2`

## Deliverables
1. Code change: config-gated no-sharing behavior.
2. New config: `coirl_no_competition.py`.
3. One ablation training run logs.
4. Combined critic-curve figure with baseline logs from parallel run.
