# sleep 5
# cd ~/projects/CoIRL-AD
# ./tools/nusc_my_train.sh coirl/exp/rebuttal_final_1+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1 8 coirl_v2_rebuttal_exp/rebuttal_final_1+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1

# sleep 5
# cd ~/projects/CoIRL-AD
# ./tools/nusc_my_train.sh coirl/exp/rebuttal_final_2+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+perception_loss 8 coirl_v2_rebuttal_exp/rebuttal_final_2+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+perception_loss

sleep 5
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/rebuttal_final_3+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+beta_1e-4+perception_loss 8 coirl_v2_rebuttal_exp/rebuttal_final_3+no_uncertainty+wm_action_mean+cmd_usage_before_planning+critic_gamma_0.1+beta_1e-4+perception_loss