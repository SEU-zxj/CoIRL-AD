sleep 5
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/16_coirl_v2+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.3 8 coirl_v2_rebuttal_exp/16_no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.3

sleep 5
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/17_coirl_v2+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1 8 coirl_v2_rebuttal_exp/17_no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1

sleep 5
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/18_coirl_v2+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.7 8 coirl_v2_rebuttal_exp/18_no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.7