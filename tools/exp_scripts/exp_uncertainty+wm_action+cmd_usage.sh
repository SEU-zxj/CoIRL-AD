sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/2_coirl_v2+model_uncertainty+wm_action_mean 8 coirl_v2_rebuttal_exp/2_model_uncertainty+wm_action_mean
sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/3_coirl_v2+model_uncertainty+wm_action_sampling 8 coirl_v2_rebuttal_exp/3_model_uncertainty+wm_action_sampling
sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/4_coirl_v2+model_uncertainty+wm_action_gt 8 coirl_v2_rebuttal_exp/4_model_uncertainty+wm_action_gt
sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/5_coirl_v2+no_uncertainty+wm_action_gt 8 coirl_v2_rebuttal_exp/5_no_uncertainty+wm_action_gt
sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/6_coirl_v2+no_uncertainty+wm_action_mean+cmd_usage_before_planning 8 coirl_v2_rebuttal_exp/6_no_uncertainty+wm_action_mean+cmd_usage_before_planning