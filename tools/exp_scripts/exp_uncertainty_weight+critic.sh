sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/7_coirl_v2+model_uncertainty_weight_1e-2+wm_action_mean 8 coirl_v2_rebuttal_exp/7_model_uncertainty_weight_1e-2+wm_action_mean
sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/8_coirl_v2+model_uncertainty_weight_1e-4+wm_action_mean 8 coirl_v2_rebuttal_exp/8_model_uncertainty_weight_1e-4+wm_action_mean
sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/9_coirl_v2+model_uncertainty_weight_1e-6+wm_action_mean 8 coirl_v2_rebuttal_exp/9_model_uncertainty_weight_1e-6+wm_action_mean
sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/10_coirl_v2+no_uncertainty+wm_action_gt+no_critic 8 coirl_v2_rebuttal_exp/10_no_uncertainty+wm_action_gt+no_critic
sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/11_coirl_v2+no_uncertainty+wm_action_gt+critic_gamma_0.1 8 coirl_v2_rebuttal_exp/10_no_uncertainty+wm_action_gt+critic_gamma_0.1
sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/12_coirl_v2+no_uncertainty+wm_action_mean+no_critic 8 coirl_v2_rebuttal_exp/12_no_uncertainty+wm_action_mean+no_critic