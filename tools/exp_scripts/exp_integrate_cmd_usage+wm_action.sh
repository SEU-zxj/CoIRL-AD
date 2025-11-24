sleep 31800
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/13_coirl_v2+no_uncertainty+wm_action_gt+cmd_usage_before_planning 8 coirl_v2_rebuttal_exp/13_no_uncertainty+wm_action_gt+cmd_usage_before_planning
sleep 10
cd ~/projects/CoIRL-AD
./tools/nusc_my_train.sh coirl/exp/14_coirl_v2+model_uncertainty_weight_1e-4+wm_action_gt+cmd_usage_before_planning 8 coirl_v2_rebuttal_exp/14_model_uncertainty_weight_1e-4+wm_action_gt+cmd_usage_before_planning