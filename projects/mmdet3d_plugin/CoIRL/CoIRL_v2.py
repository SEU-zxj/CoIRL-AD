import time
import copy
import numpy as np
import torch, cv2, os, random
import torch.nn as nn
from mmdet.models import DETECTORS
import timm
from mmdet3d.core import bbox3d2result
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from projects.mmdet3d_plugin import VAD
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric
from mmdet3d.models import builder

from projects.mmdet3d_plugin.CoIRL.utils import CompetitiveLearningMachine
import matplotlib.pyplot as plt
from ipdb import set_trace
import pickle

@DETECTORS.register_module()
class CoIRL_v2(VAD):
    def __init__(self,
                use_video=False,
                use_swin=False,
                only_front_view=False,
                use_multi_view=True,
                swin_input_channel=768,
                hidden_channel=256,
                use_semantic=False,
                semantic_img_backbone=None,
                flow_only=False,
                all_zero=False,
                semantic_only=False,
                use_2d_waypoint=False,
                wm_loss_weight=0.2,
                rl_loss_weight=1.0,
                rl_traj_gauss_nll_weight=0.1,
                compete_min_threshold=1.0,
                eval_method='il',
                save_results_flag=False,
                results_path=None,
                bev_encoder=None,
                actor_il_head=None,
                actor_rl_head=None,
                rl_actor_use_bc=None,
                competition_warmup_flag=False,
                competition_warmup_threshold=5000,
                **kwargs,
                 ):
        super().__init__( **kwargs)
        self.perception_mode = self.pts_bbox_head.perception_mode
        self.depth_eps = 2.75
        self.ref_pts_cam_list = []
        self.use_video = use_video
        self.use_swin = use_swin
        self.use_semantic = use_semantic

        self.flow_only = flow_only
        self.semantic_only = semantic_only
        self.all_zero = all_zero

        self.only_front_view = only_front_view
        self.use_2d_waypoint = use_2d_waypoint

        if semantic_img_backbone is not None:
            self.semantic_img_backbone = builder.build_backbone(semantic_img_backbone)

        if (not self.with_img_neck) and self.use_swin:
            self.swin_img_mlp = nn.Linear(swin_input_channel, hidden_channel)
        
        self.metrics_history = []
        self.call_count = 0
        self.wm_loss_weight = wm_loss_weight
        self.rl_loss_weight = rl_loss_weight
        self.rl_traj_gauss_nll_weight = rl_traj_gauss_nll_weight

        self.bev_encoder = builder.build_backbone(bev_encoder)
        self.actor_il_head = builder.build_backbone(actor_il_head)
        self.actor_rl_head = builder.build_backbone(actor_rl_head)
        self.group_size = self.actor_rl_head.group_size
        self.debug_std = self.actor_rl_head.debug_std
        self.use_critic = self.actor_rl_head.use_critic
        self.rl_actor_use_bc = rl_actor_use_bc
        self.eval_method = eval_method
        
        self.save_results_flag = save_results_flag
        self.results_path = results_path
        if save_results_flag:
            self.results_return = {} # key is sample_idx, value is a dict with keys: ['scene_token', eval_metrics, 'ego_pred_traj']
        self.CLM = CompetitiveLearningMachine(il_actor=self.actor_il_head, rl_actor=self.actor_rl_head, use_critic=self.use_critic, min_threshold=compete_min_threshold, competition_warmup_flag=competition_warmup_flag, competition_warmup_threshold=competition_warmup_threshold)
        

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            # My modification
            if self.use_video:
                img = img.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
                img_feats = self.img_backbone(img)
                img_feats = img_feats.mean(dim=2, keepdim=True)
            else:
                if self.use_swin:
                    img = img.unsqueeze(2)
                img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        if not isinstance(img_feats, tuple):
            img_feats = [img_feats.squeeze(2)]

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        elif self.use_swin: #swin without fpn
            img_feats = [self.swin_img_mlp(img_feats[0].permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()]

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    
    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    # def obtain_history_feat(self, imgs_queue, img_metas_list, is_test=False):
    #     """Obtain history BEV features iteratively.
    #     """
    #     bs, len_queue, num_cams, C, H, W = imgs_queue.shape
    #     imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
    #     img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
    #     losses = {}
    #     for i in range(len_queue):
    #         img_metas = [each[i] for each in img_metas_list]
    #         img_feats = [each_scale[:, i] for each_scale in img_feats_list][0]
    #         pred_ego_fut_trajs, cur_img_feat, pred_img_feat = self.pts_bbox_head(img_feats, img_metas)

    #         # compute loss
    #         if not is_test:
    #             # loss waypoint
    #             gt_ego_fut_trajs = img_metas[0]['ego_fut_trajs'].to(img_feats.device)
    #             gt_ego_fut_masks = img_metas[0]['ego_fut_masks'].squeeze(0).unsqueeze(-1).to(img_feats.device)
    #             loss_waypoint = self.pts_bbox_head.loss_3d(pred_ego_fut_trajs,
    #                                                         gt_ego_fut_trajs,
    #                                                         gt_ego_fut_masks,
    #                                                         )
    #             losses.update({
    #                 f'prev_frame_loss_waypoint_{i}': loss_waypoint
    #             })
    #             #  RL loss
    #             if self.rl_actor_use_bc:
    #                 if self.debug_std:
    #                     preds_ego_future_policy, _, _ = self.pts_bbox_head_rl(img_feats, img_metas)
    #                 else:
    #                     preds_ego_future_policy, _ = self.pts_bbox_head_rl(img_feats, img_metas)
    #                 loss_rl_bc = self.pts_bbox_head_rl.loss_rl_bc(preds_ego_future_policy, gt_ego_fut_trajs.unsqueeze(0), gt_ego_fut_masks.unsqueeze(0).unsqueeze(0).squeeze(-1))
    #                 losses.update({
    #                     f'prev_frame_loss_rl_bc_{i}': loss_rl_bc * self.rl_traj_gauss_nll_weight
    #                 })
    #     return losses, pred_img_feat

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [img_metas_list[i]]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            self.actor_rl_head.refer_critic.eval()
            return prev_bev
        
    def obtain_current_bev_and_pred_fut_state(self, img, img_metas, prev_bev):
        """Use world model to predict future state, and to fully utilize the data, also use imitation loss for IL Actor and behavior cloning loss for RL Actor here
        """
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        assert len(img_feats) == 1, "there ate multi timestep img input in `obtain_current_bev_and_pred_fut_state`"
        # first obtain current bev, shape (B, 200*200, 256)
        cur_bev_embed = self.pts_bbox_head(
                img_feats, img_metas, prev_bev, only_bev=True)
        cur_state = self.bev_encoder(cur_bev_embed)
        # then calculate loss
        losses = {}
        # ===IL Actor=== #
        preds_ego_future_traj, pred_fut_state = self.actor_il_head(cur_state, img_metas)
        gt_ego_fut_trajs = img_metas[0]['ego_fut_trajs'].to(img_feats[0].device)
        gt_ego_fut_masks = img_metas[0]['ego_fut_masks'].squeeze(0).unsqueeze(-1).to(img_feats[0].device)
        loss_waypoint = self.actor_il_head.loss_3d(preds_ego_future_traj, gt_ego_fut_trajs, gt_ego_fut_masks)
        losses.update({
            f'prev_frame_loss_waypoint_0': loss_waypoint
        })
        # ===RL Actor=== #
        if self.rl_actor_use_bc:
            if self.debug_std:
                preds_ego_future_policy, _ = self.actor_rl_head(cur_state, img_metas)
            else:
                preds_ego_future_policy = self.actor_rl_head(cur_state, img_metas)
            loss_rl_bc = self.actor_rl_head.loss_rl_bc(preds_ego_future_policy, gt_ego_fut_trajs.unsqueeze(0), gt_ego_fut_masks.unsqueeze(0).unsqueeze(0).squeeze(-1))
            losses.update({
                f'prev_frame_loss_rl_bc_0': loss_rl_bc * self.rl_traj_gauss_nll_weight
            })        

        return cur_state, pred_fut_state, losses

    @force_fp32(apply_to=('img','points','prev_bev'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      map_gt_bboxes_3d=None,
                      map_gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      semantic_img=None,
                      flow_img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      map_gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ego_his_trajs=None,
                      ego_fut_trajs=None,
                      ego_fut_masks=None,
                      ego_fut_cmd=None,
                      ego_lcf_feat=None,
                      gt_attr_labels=None,
                      fut_valid_flag=None
                      ):
        """
        agent lcf feat (x, y, yaw, vx, vy, width, length, height, type)
        """
        if self.only_front_view:
            img = img[:, :, 0:1, ...]
        # in v2, len_queue should be 7 in training process
        len_queue = img.size(1)
        assert len_queue == 7, 'len_queue is not 7'
        # here we process the 7 (id: 0, 1, 2, 3, 4, 5, 6) images via steps below:
        # 1) prev_wm_img: for id 0, 1, 2, get prev_bev, temporal augmented
        # 2) wm_img: for id 3, get current bev (use prev_bev), and predict future bev via world model and then calculate loss for IL Actor and RL Actor (as we want to fully use the data)
        # 3) prev_img: (forget prev_bev, only use world model pred future_bev) for id 3, 4, 5, get prev_bev, temporal augmented
        # 4) cur_img: go training steps at image id 6 with labels

        # here `img_metas` is a list of dict: [{0:{}, 1:{}, ..., 6:{}}]
        img_metas_copy = copy.deepcopy(img_metas)

        prev_wm_img = img[:, 0:3, ...]
        prev_wm_img_metas = [img_metas_copy[0][0], img_metas_copy[0][1], img_metas_copy[0][2]]

        wm_img = img[:, 3, ...]
        wm_img_metas = [img_metas_copy[0][3]]

        prev_img = img[:, 3:6, ...]
        prev_img_metas = [img_metas_copy[0][3], img_metas_copy[0][4], img_metas_copy[0][5]]

        cur_img = img[:, 6, ...]
        cur_img_metas = [img_metas_copy[0][6]]
        
        # 1)
        prev_wm_bev = self.obtain_history_bev(imgs_queue=prev_wm_img, img_metas_list=prev_wm_img_metas)
        # 2)
        _, wm_pred_fut_state, prev_frame_losses = self.obtain_current_bev_and_pred_fut_state(img=wm_img, img_metas=wm_img_metas, prev_bev=prev_wm_bev)
        # 3)
        prev_bev = self.obtain_history_bev(imgs_queue=prev_img, img_metas_list=prev_img_metas)
        # 4)
        cur_img_feats = self.extract_feat(img=cur_img, img_metas=cur_img_metas)

        losses = self.forward_pts_train(cur_img_feats, 
                                        cur_img_metas,
                                        prev_bev=prev_bev,
                                        gt_bboxes_3d=gt_bboxes_3d,
                                        gt_labels_3d=gt_labels_3d,
                                        map_gt_bboxes_3d=map_gt_bboxes_3d,
                                        map_gt_labels_3d=map_gt_labels_3d,
                                        gt_labels=gt_labels,
                                        gt_bboxes=gt_bboxes,
                                        wm_pred_fut_state=wm_pred_fut_state,
                                        ego_his_trajs=ego_his_trajs, ego_fut_trajs=ego_fut_trajs,
                                        ego_fut_masks=ego_fut_masks, ego_fut_cmd=ego_fut_cmd,
                                        ego_lcf_feat=ego_lcf_feat, gt_attr_labels=gt_attr_labels,
                                        fut_valid_flag=fut_valid_flag
                                    )
        losses.update(prev_frame_losses)
        return losses

    def forward_pts_train(self,
                          img_feats,
                          img_metas,
                          prev_bev,
                          gt_bboxes_3d=None,
                          gt_labels_3d=None,
                          map_gt_bboxes_3d=None,
                          map_gt_labels_3d=None,
                          gt_labels=None,
                          gt_bboxes=None,
                          wm_pred_fut_state=None,
                          ego_his_trajs=None,
                          ego_fut_trajs=None,
                          ego_fut_masks=None,
                          ego_fut_cmd=None,
                          ego_lcf_feat=None,
                          gt_attr_labels=None,
                          fut_valid_flag=None
                        ):
        """Forward function
        Args:
            ego_fut_cmd: [turn_left, turn_right, go_straight]
            ego_lcf_feat: (vx, vy, ax, ay, w, length, width, vel, steer), w: yaw角速度

        """
        #get the ego info   
        losses = {}
        B = ego_his_trajs.size(0)
        ego_his_trajs = ego_his_trajs.reshape(B, -1)
        ego_lcf_feat = ego_lcf_feat.reshape(B, -1)
        ego_fut_cmd = ego_fut_cmd.reshape(B, -1)
        ego_info = torch.cat([ego_his_trajs, ego_lcf_feat, ego_fut_cmd], dim=1)
        
        prev_pred_state = wm_pred_fut_state
        # ===== Perception Module Pipeline ====== #
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev,
                                ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        if self.perception_mode == 'perception-based':
            loss_inputs = [
                gt_bboxes_3d, gt_labels_3d, map_gt_bboxes_3d, map_gt_labels_3d,
                outs, ego_fut_trajs, ego_fut_masks, ego_fut_cmd, gt_attr_labels
            ]
            perception_losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

            losses.update(perception_losses)

        cur_bev_embed = outs['bev_embed']
        cur_state = self.bev_encoder(cur_bev_embed)
        # ===== IL Actor Training Pipeline ====== #
        preds_ego_future_traj, pred_fut_state = self.actor_il_head(cur_state, img_metas, ego_info)
        
        # world model loss
        loss_rec = self.actor_il_head.loss_reconstruction(prev_pred_state, cur_state.detach())
        losses['loss_rec'] = loss_rec * self.wm_loss_weight

        # waypoint loss
        loss_waypoint = self.actor_il_head.loss_3d(preds_ego_future_traj,
                                            ego_fut_trajs.squeeze(1),
                                            ego_fut_masks.squeeze(0).squeeze(0).unsqueeze(-1),
                                            )
        losses.update({'loss_waypoint': loss_waypoint})

        # ===== RL Actor Training Pipeline ====== #
        if self.debug_std:
            preds_ego_future_policy, std_info = self.actor_rl_head(cur_state, img_metas, ego_info)
            losses.update(std_info)
        else:
            preds_ego_future_policy = self.actor_rl_head(cur_state, img_metas, ego_info)

        if self.use_critic:
            # sample trajs and construct step aware trajectories
            sample_traj_group = preds_ego_future_policy.rsample([self.group_size]) # [G, B, T, 2]
            sample_traj_group = sample_traj_group.permute(1,0,2,3) # [B, G, T, 2]

            mode_traj = preds_ego_future_policy.mean # [B, T, 2]
            traj_len = mode_traj.size(1)

            traj_group = mode_traj.unsqueeze(1).unsqueeze(2).repeat(1, traj_len, self.group_size, 1, 1) # [B, T, G, T, 2]

            for i in range(traj_len):
                traj_group[:,i,:,i,:] = sample_traj_group[:,:,i,:] # [B, T, G, T, 2]
            
            # compute value
            cur_value = self.actor_rl_head.critic(cur_state.detach()) # [B,]
            cur_value = cur_value.unsqueeze(1).unsqueeze(2).expand(B, traj_len, self.group_size) # [B, T, G]

            # pred fut_state
            traj_group = traj_group.reshape(B, traj_len*self.group_size, traj_len, 2)
            with torch.no_grad():
                pred_fut_state_rl = self.actor_il_head.wm_group_prediction(cur_state.detach(), traj_group.detach()) # [B, T*G, n_token, H]
            with torch.no_grad():
                pred_fut_value = self.actor_rl_head.refer_critic(pred_fut_state_rl.detach()) # [B, T*G]
                pred_fut_value = pred_fut_value.reshape(B, traj_len, self.group_size) # [B, T, G]
            
            # compute actor_loss and reward
            traj_group = traj_group.reshape(B, traj_len, self.group_size, traj_len, 2)
            # reward [B, T, G, T]
            loss_actor, reward = self.actor_rl_head.compute_actor_loss(traj_group, preds_ego_future_policy, ego_fut_trajs, gt_bboxes_3d, gt_attr_labels, fut_valid_flag, ego_fut_masks, cur_value, pred_fut_value)

            # compute critic_loss
            loss_critic = self.actor_rl_head.compute_critic_loss(reward, cur_value, pred_fut_value)

            losses.update({
                'loss_rl': loss_actor * self.rl_loss_weight,
                'loss_critic': loss_critic * self.rl_loss_weight
            })
        else:
            loss_rl = self.actor_rl_head.loss_rl_group_sampling(preds_ego_future_policy, ego_fut_trajs, gt_bboxes_3d, gt_attr_labels, fut_valid_flag, ego_fut_masks)

            losses.update({
                'loss_rl': loss_rl * self.rl_loss_weight
            })

        if self.rl_actor_use_bc:
            loss_rl_bc = self.actor_rl_head.loss_rl_bc(preds_ego_future_policy, ego_fut_trajs, ego_fut_masks)
            losses.update({
                'loss_rl_bc': loss_rl_bc * self.rl_traj_gauss_nll_weight
            })        

        competition_info = self.CLM.competitive_learning(preds_ego_future_traj, preds_ego_future_policy.mean, ego_fut_trajs.squeeze(1), gt_bboxes_3d, gt_attr_labels, fut_valid_flag, ego_fut_masks.squeeze(0).squeeze(0))

        losses.update(competition_info)

        return losses

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        bbox_results = self.simple_test(
            img_metas=img_metas,
            img=img,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            **kwargs
        )

        return bbox_results
    
    def simple_test(
        self,
        img_metas,
        img,
        gt_bboxes_3d,
        gt_labels_3d,
        fut_valid_flag=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs,
    ):
        len_queue = img.size(1)
        # in v2, len_queue is 4 in testing mode, if it is not 4, we can also test!
        # assert len_queue == 4, "only support len_queue == 4 in test mode of v2"
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        self.pts_bbox_head.prev_view_feat = None
        if len_queue > 1:
            prev_img_metas = [prev_img_metas[0][idx] for idx in range(len_queue-1)]
            prev_bev = self.obtain_history_bev(imgs_queue=prev_img, img_metas_list=prev_img_metas)
        else:
            prev_bev = None

        cur_img = img[:, -1, ...]
        cur_img_metas = [each[len_queue-1] for each in img_metas]

        cur_img_feats = self.extract_feat(img=cur_img, img_metas=cur_img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        metric_dict = self.simple_test_pts(
            cur_img_feats,
            cur_img_metas,
            prev_bev,
            gt_bboxes_3d,
            gt_labels_3d,
            fut_valid_flag=fut_valid_flag,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )
        
        for result_dict in bbox_list:
            result_dict['metric_results'] = metric_dict

        return bbox_list
    
    def simple_test_pts(
        self,
        img_feats,
        img_metas,
        prev_bev,
        gt_bboxes_3d,
        gt_labels_3d,
        fut_valid_flag=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """Test function"""
        B = ego_his_trajs.size(0)
        ego_his_trajs_ = ego_his_trajs.reshape(B, -1)
        ego_lcf_feat_ = ego_lcf_feat.reshape(B, -1)
        ego_fut_cmd_ = ego_fut_cmd.reshape(B, -1)
        ego_info = torch.cat([ego_his_trajs_, ego_lcf_feat_, ego_fut_cmd_], dim=1)

        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev,
                                        ego_his_trajs=ego_his_trajs, ego_lcf_feat=ego_lcf_feat)
        cur_bev_embed = outs['bev_embed']
        cur_state = self.bev_encoder(cur_bev_embed)

        preds_ego_future_traj, _ = self.actor_il_head(cur_state, img_metas)
        if self.eval_method == 'decouple':
            preds_ego_future_policy = self.actor_rl_head(cur_state, img_metas, is_test=True)
            preds_ego_future_traj_rl = preds_ego_future_policy.mean
            
        with torch.no_grad():
            # pre-process
            gt_bbox = gt_bboxes_3d[0][0]
            gt_label = gt_labels_3d[0][0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0][0])

            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = preds_ego_future_traj[0]
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            
            ego_fut_preds = ego_fut_preds.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_preds[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                fut_valid_flag = fut_valid_flag
            )

            if self.eval_method == 'decouple':
                ego_fut_preds_rl = preds_ego_future_traj_rl[0]
                ego_fut_preds_rl = ego_fut_preds_rl.cumsum(dim=-2)

                metric_dict_planner_stp3_rl = self.compute_planner_metric_stp3(
                    pred_ego_fut_trajs = ego_fut_preds_rl[None],
                    gt_ego_fut_trajs = ego_fut_trajs[None],
                    gt_agent_boxes = gt_bbox,
                    gt_agent_feats = gt_attr_label.unsqueeze(0),
                    fut_valid_flag = fut_valid_flag
                )

                metric_dict_planner_stp3_rl = {f"{key}_rl_actor": metric_dict_planner_stp3_rl[key] for key in metric_dict_planner_stp3_rl.keys()}

            # mid print
            # update metrics
            if self.eval_method == "decouple":
                metric_dict_planner_stp3.update(metric_dict_planner_stp3_rl)
                self.metrics_history.append(metric_dict_planner_stp3)
            else:
                self.metrics_history.append(metric_dict_planner_stp3)
            self.call_count += 1

            # print results
            if self.call_count % 500 == 0 and 'vis' not in self.eval_method:
                self.compute_and_print_metrics_average()

            if self.save_results_flag:
                self.insert_sample_results(
                    sample_token=img_metas[0]['sample_idx'],
                    scene_token=img_metas[0]['scene_token'],
                    metric_dict=metric_dict_planner_stp3,
                    ego_fut_traj=ego_fut_preds
                )

        return metric_dict_planner_stp3
    
    def compute_and_print_metrics_average(self):
        # compute avg
        avg_metrics = {key: sum(m[key] for m in self.metrics_history) / len(self.metrics_history) for key in self.metrics_history[0]}
        # print avg
        print(f"\n Average metrics after {len(self.metrics_history)} calls: {avg_metrics}")

    def insert_sample_results(self, sample_token, scene_token, metric_dict, ego_fut_traj):
        sample_result = {'scene_token': scene_token}
        sample_result.update(metric_dict)
        sample_result.update({'ego_fut_traj': ego_fut_traj.detach().cpu().numpy()})

        self.results_return[sample_token] = sample_result

    def save_results(self):
        with open(self.results_path, 'wb') as f:
            pickle.dump(self.results_return, f)

@DETECTORS.register_module()
class BEVEncoder(BaseModule):
    def __init__(self, bev_h, bev_w, hidden_channel, backbone_out_chans, backbone):
        super().__init__()
        self.bev_h=bev_h
        self.bev_w=bev_w
        self.hidden_channel = hidden_channel
        self.backbone_out_chans = backbone_out_chans
        # 1. Normalization (Channel-wise LayerNorm over embedding dim = 256)
        self.norm = nn.LayerNorm(hidden_channel)

        # 2. BEV backbone: Swin Transformer
        self.backbone = timm.create_model(**backbone)

        # 3. linear projection: 768 → 256 with 2-layer MLP
        self.proj = nn.Sequential(
            nn.Linear(backbone_out_chans, hidden_channel*2),
            nn.GELU(),
            nn.Linear(hidden_channel*2, hidden_channel)
        )

    def forward(self, bev_embed):
        """
        bev_embed: (B, 40000, 256)
        40000 = 200 x 200
        """
        B, N, C = bev_embed.shape

        # 1. Normalize
        bev_embed = self.norm(bev_embed)

        # 2. Reshape for Swin: (B, C, H, W)
        x = bev_embed.transpose(1, 2).reshape(B, C, self.bev_h, self.bev_w)

        # 3. Forward Swin
        feats = self.backbone(x)      # tuple of features
        x = feats[0]                  # (B, 7, 7, 768)

        # 4. Flatten spatial dims → tokens
        x = x.flatten(1, 2)  # (B, 49, 768)

        # 5. Project to 256
        x = self.proj(x)   # (B, 49, 256)

        return x