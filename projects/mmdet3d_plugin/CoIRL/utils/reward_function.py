import torch
import torch.nn as nn
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric

class DrivableAreaConstrain(nn.Module):
    """EndPoint constraint to push ego vehicle go to the end point."""

    def __init__(self, patch_h=60.0, patch_w=30.0):
        super().__init__()
        self.max_x = patch_w / 2
        self.max_y = patch_h / 2

    def forward(self, traj_group):
        """
        ego_fut_preds (Tensor): [B, T, 2]
        """
        # Ensure T >= 4 or 6 (based on max_y), otherwise slice to available timesteps
        T = traj_group.shape[1]
        slice_T = min(4, T)
        invalid_mask_x_axis = ((traj_group[:, :slice_T, 0] > self.max_x) | (traj_group[:, :slice_T, 0] < -self.max_x)).any(dim=-1)  # [B]
        invalid_mask_y_axis = ((traj_group[:, :slice_T, 1] > self.max_y) | (traj_group[:, :slice_T, 1] < -self.max_y)).any(dim=-1)  # [B]
        invalid_mask = invalid_mask_x_axis | invalid_mask_y_axis
        valid_mask = ~invalid_mask
        dac_score = valid_mask.int()
        return dac_score
    
class ImitationConstrain(nn.Module):
    """Imitation constraint to push ego vehicle mimic the expert drivers' trajectory."""

    def __init__(
        self,
    ):
        super().__init__()
    
    def reward_function(self, dist):
        '''
        dist (Tensor): (B, T)
        '''
        return torch.exp(-dist)

    def forward(
            self,
            ego_fut_preds,
            ego_fut_gt,
            ego_fut_masks
        ):
        '''
        ego_fut_preds (Tensor): [B, T, 2]
        ego_fut_gt (Tensor): [B, T, 2]
        ego_fut_masks (Tensor): [B, T]

        and here, the `ego_fut_preds` and `ego_fut_gt` are waypoints (after cumsum), instead of displacement of waypoints
        '''

        dist = torch.linalg.norm(ego_fut_preds - ego_fut_gt, dim=-1) # (B, T)
        reward = self.reward_function(dist) # (B, T)
        reward = reward * ego_fut_masks # (B, T)
        # valid_count = ego_fut_masks.sum(dim=-1).clamp(min=1)
        # reward = reward.sum(dim=-1) / valid_count #(B,)
        return reward # (B, T)
    
class CollsionConstrain(nn.Module):
    """calculate the collision score of given trajectory
    no collision: 1
    collision: 0
    """
    def __init__(self):
        super().__init__()
    def forward(self, ego_fut_preds, ego_fut_gt, gt_bboxes_3d, gt_attr_labels, fut_valid_flag):
        '''
        ego_fut_preds (Tensor): [B, T, 2]
        ego_fut_gt (Tensor): [B, T, 2]
        gt_bboxes_3d List[Tensor]: [n_agent, position_info]
        gt_attr_labels (Tensor): [n_agent, agent_feat]
        fut_valid_flag (Tensor): [B,]
        '''
        bz, t, _ = ego_fut_preds.shape
        gt_bbox = gt_bboxes_3d[0]
        gt_attr_label = gt_attr_labels[0].to('cpu')
        fut_valid_flag = bool(fut_valid_flag[0])        
        n_agent = gt_bbox.tensor.size(0)

        if n_agent == 0:
            # if there are no agents, impossibly to collision with others
            return torch.ones((bz, t), device=ego_fut_preds.device)

        self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_bbox, gt_attr_label.unsqueeze(0))
        occupancy = torch.logical_or(segmentation, pedestrian)

        if fut_valid_flag:
            _, obj_box_coll = self.planning_metric.evaluate_coll(
                ego_fut_preds.detach(),
                ego_fut_gt,
                occupancy)
            box_col = obj_box_coll.to(ego_fut_gt.device) # [B, T]
        else:
            box_col = torch.zeros((bz, t), device=ego_fut_gt.device) # [self.group_size,]

        no_colision_score = (box_col <= 0).float() # [self.group_size, T]

        return no_colision_score
    
class CollsionConstrain_RL(nn.Module):
    """calculate the collision score of given trajectory
    no collision: 1
    collision: 0
    """
    def __init__(self):
        super().__init__()
    def forward(self, ego_fut_preds, ego_fut_gt, gt_bboxes_3d, gt_attr_labels, fut_valid_flag):
        '''
        ego_fut_preds (Tensor): [B*group_size, T, 2]
        ego_fut_gt (Tensor): [B*group_size, T, 2]
        gt_bboxes_3d List[Tensor]: [n_agent, position_info]
        gt_attr_labels (Tensor): [n_agent, agent_feat]
        fut_valid_flag (Tensor): [B,]
        '''
        bs = fut_valid_flag.size(0)
        bs_group_size, traj_len, _ = ego_fut_preds.shape
        group_size = bs_group_size // bs
        gt_bbox = gt_bboxes_3d[0]
        gt_attr_label = gt_attr_labels[0].to('cpu')
        fut_valid_flag = bool(fut_valid_flag[0])        
        n_agent = gt_bbox.tensor.size(0)

        if n_agent == 0:
            # if there are no agents, impossibly to collision with others
            return torch.ones((bs, group_size, traj_len), device=ego_fut_preds.device)

        self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(
            gt_bbox, gt_attr_label.unsqueeze(0))
        occupancy = torch.logical_or(segmentation, pedestrian)

        bev_h, bev_w = occupancy.size(2), occupancy.size(3)
        occupancy = occupancy.expand(bs*group_size, traj_len, bev_h, bev_w)

        if fut_valid_flag:
            _, obj_box_coll = self.planning_metric.evaluate_coll_batch(
                ego_fut_preds.detach(),
                ego_fut_gt,
                occupancy)
            box_col = obj_box_coll.to(ego_fut_gt.device) # [bs*group_size, T]
        else:
            box_col = torch.zeros((bs*group_size, traj_len), device=ego_fut_gt.device) # [bs*self.group_size, T]

        no_colision_score = (box_col <= 0).float().reshape(bs, group_size, traj_len) # [bs, self.group_size, T]

        return no_colision_score