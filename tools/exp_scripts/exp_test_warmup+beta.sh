sleep 5
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/20_coirl_v2+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+warmup 8 coirl_v2_rebuttal_exp/20_no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+warmup

sleep 5
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/21_coirl_v2+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+warmup+beta_1e-6 8 coirl_v2_rebuttal_exp/21_no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+warmup+beta_1e-6

sleep 5
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/22_coirl_v2+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+warmup+beta_1e-5 8 coirl_v2_rebuttal_exp/22_no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+warmup+beta_1e-5

sleep 5
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/23_coirl_v2+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+warmup+beta_1e-4 8 coirl_v2_rebuttal_exp/23_no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+warmup+beta_1e-4

sleep 5
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/24_coirl_v2+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+warmup+beta_1e-3 8 coirl_v2_rebuttal_exp/24_no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+warmup+beta_1e-3