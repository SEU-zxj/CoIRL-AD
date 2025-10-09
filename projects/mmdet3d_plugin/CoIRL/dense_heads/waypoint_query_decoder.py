import torch, random
import torch.nn as nn
from mmcv.runner import BaseModule, force_fp32
from torch.nn import functional as F
import math

from mmdet3d.models import builder
from mmdet3d.models.builder import build_loss
from mmdet3d.core import AssignResult, PseudoSampler
from mmdet.core import build_bbox_coder, build_assigner, multi_apply, reduce_mean
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid

from projects.mmdet3d_plugin.CoIRL.utils import DrivableAreaConstrain, ImitationConstrain, CollsionConstrain_RL

from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from PIL import Image
import time, json
from torch.distributions.multivariate_normal import MultivariateNormal

from projects.mmdet3d_plugin.CoIRL.dense_heads.utils import get_locations
from projects.mmdet3d_plugin.VAD.planner.metric_stp3 import PlanningMetric

@HEADS.register_module()
class WaypointHead_IL(BaseModule):
    def __init__(self,
                num_proposals=6,
                #MHA
                hidden_channel=256,
                dim_feedforward=1024,
                num_heads=8,
                dropout=0.0,
                #pos embedding
                depth_step=0.8,
                depth_num=64,
                depth_start = 0,
                position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                stride=32,
                num_views=6,
                #others
                train_cfg=None,
                test_cfg=None,
                use_wm=True,
                use_causal=True, # CoIRL-AD use inv causal while LAW do not use causal
                num_spatial_token=36,
                num_tf_layers=2,
                num_traj_modal=1,
                **kwargs,
                ):
        """
        use to predict the waypoints
        """
        super().__init__(**kwargs)
        self.use_wm = use_wm

        # query feature
        self.num_views = num_views
        self.num_proposals = num_proposals
        self.view_query_feat = nn.Parameter(torch.randn(1, self.num_views, hidden_channel, self.num_proposals))
        self.waypoint_query_feat = nn.Parameter(torch.randn(1, self.num_proposals, hidden_channel))

        # spatial attn
        spatial_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        ) 
        
        self._spatial_decoder = nn.ModuleList( [
            nn.TransformerDecoder(spatial_decoder_layer, 1) 
            for _ in range(self.num_views)])

        self.use_causal = use_causal
        if use_causal:
            # inv AR
            # example, no means masked
            # first output last point, then backward planning
            # yes yes yes
            #  no yes yes
            #  no  no yes
            self.causal_mask = torch.tril(torch.ones(self.num_proposals, self.num_proposals), diagonal=-1).bool()
            # self.causal_mask = torch.triu(torch.ones(self.num_proposals, self.num_proposals), diagonal=1).bool()
            self.auto_regression_attention = nn.MultiheadAttention(embed_dim=hidden_channel, num_heads=8, batch_first=True)

        # wp_attn
        wp_decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_channel,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        self.wp_attn = nn.TransformerDecoder(wp_decoder_layer, 1) # input: Bz, num_token, d_model

        # world model
        wm_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self._wm_decoder = nn.TransformerDecoder(wm_decoder_layer, num_tf_layers) 

        self.action_aware_encoder = nn.Sequential(
            nn.Linear(hidden_channel + 6*2, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
        )
        
        # loss
        self.loss_plan_reg = build_loss(dict(type='L1Loss', loss_weight=1.0))
        self.loss_plan_rec = nn.MSELoss()

        # head
        self.num_traj_modal = num_traj_modal
        self.waypoint_head = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, self.num_traj_modal* 2)
            )

        # position embedding
        ##img pos embed
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.depth_start = depth_start
        self.stride = stride

        self.position_encoder = nn.Sequential(
                nn.Linear(self.position_dim, hidden_channel*4),
                nn.ReLU(),
                nn.Linear(hidden_channel*4, hidden_channel),
            )

        self.pc_range = nn.Parameter(torch.tensor(point_cloud_range), requires_grad=False)
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        # LID depth
        index = torch.arange(start=0, end=self.depth_num, step=1).float()
        index_1 = index + 1
        bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
        coords_d = self.depth_start + bin_size * index * index_1
        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

    def prepare_location(self, img_metas, img_feats):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = img_feats.shape[:2]
        x = img_feats.flatten(0, 1)
        location = get_locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location
    
    def img_position_embeding(self, img_feats, img_metas):
        """
        from streampetr
        """
        eps = 1e-5
        B, num_views, C, H, W = img_feats.shape
        assert num_views == self.num_views, 'num_views should be equal to self.num_views'
        BN = B * num_views
        num_sample_tokens = num_views * H * W
        LEN = num_sample_tokens
        img_pixel_locations = self.prepare_location(img_metas, img_feats)

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        img_pixel_locations[..., 0] = img_pixel_locations[..., 0] * pad_w
        img_pixel_locations[..., 1] = img_pixel_locations[..., 1] * pad_h

        # Depth
        D = self.coords_d.shape[0]
        pixel_centers = img_pixel_locations.detach().view(B, LEN, 1, 2).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([pixel_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        lidar2img = torch.from_numpy(np.stack(img_metas[0]['lidar2img'])).to(img_feats.device).float()
        lidar2img = lidar2img[:num_views]
        img2lidars = lidar2img.inverse()
        img2lidars = img2lidars.view(num_views, 1, 1, 4, 4).repeat(B, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3]) #normalize
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d) #(B, num_views*H*W, 3*64)
        coords_position_embeding = self.position_encoder(pos_embed)
        return coords_position_embeding
    
    def forward(self, img_feat, img_metas, ego_info=None, is_test=False):
        # init
        losses = {}
        Bz, num_views, num_channels, height, width = img_feat.shape
        init_view_query_feat = self.view_query_feat.clone().repeat(Bz, 1, 1, 1).permute(0, 1, 3, 2)
        init_waypoint_query_feat = self.waypoint_query_feat.clone().repeat(Bz, 1, 1)

        # img pos emb
        img_pos = self.img_position_embeding(img_feat, img_metas)
        img_pos = img_pos.reshape(Bz, num_views, height, width, num_channels)
        img_pos = img_pos.permute(0, 1, 4, 2, 3)
        img_feat_emb = img_feat + img_pos   

        # spatial view feat
        img_feat_emb = img_feat_emb.reshape(Bz, num_views, num_channels, height*width).permute(0, 1, 3, 2)
        spatial_view_feat = torch.zeros_like(init_view_query_feat)
        for i in range(self.num_views):
            spatial_view_feat[:, i] = self._spatial_decoder[i](init_view_query_feat[:, i], img_feat_emb[:, i])

        batch_size, num_view, num_tokens, num_channel = spatial_view_feat.shape
        spatial_view_feat = spatial_view_feat.reshape(batch_size, -1, num_channel)

        # predict wp
        updated_waypoint_query_feat = self.wp_attn(init_waypoint_query_feat, spatial_view_feat) #final_view_feat.shape torch.Size([1, 1440, 256])

        if self.use_causal:
            # inv AR
            if self.causal_mask.device == torch.device('cpu'):
                self.causal_mask = self.causal_mask.cuda()

            updated_waypoint_query_feat, _ = self.auto_regression_attention(
                query=updated_waypoint_query_feat,
                key=updated_waypoint_query_feat,
                value=updated_waypoint_query_feat,
                attn_mask=self.causal_mask,
                need_weights=False
            )

        cur_waypoint = self.waypoint_head(updated_waypoint_query_feat)

        if self.num_traj_modal > 1:
            assert self.num_traj_modal == 3
            bz, traj_len, _ = cur_waypoint.shape
            cur_waypoint = cur_waypoint.reshape(bz, traj_len, self.num_traj_modal, 2)
            ego_cmd = img_metas[0]['ego_fut_cmd'].to(img_feat.device)[0, 0]
            cur_waypoint = cur_waypoint[: ,: ,ego_cmd == 1].squeeze(2)

        # world model prediction
        wm_next_latent = self.wm_prediction(spatial_view_feat, cur_waypoint)

        return cur_waypoint, spatial_view_feat, wm_next_latent
    
    def loss_reconstruction(self, 
            reconstructed_view_query_feat,
            observed_view_query_feat,
            ):
        loss_rec = self.loss_plan_rec(reconstructed_view_query_feat, observed_view_query_feat)
        return loss_rec
    
    def wm_prediction(self, view_query_feat, cur_waypoint):
        batch_size, num_tokens, num_channel = view_query_feat.shape
        cur_waypoint = cur_waypoint.reshape(batch_size, 1, -1).repeat(1, num_tokens, 1)
        cur_view_query_feat_with_ego = torch.cat([view_query_feat, cur_waypoint], dim=-1) 
        action_aware_latent = self.action_aware_encoder(cur_view_query_feat_with_ego)

        wm_next_latent = self._wm_decoder(action_aware_latent, action_aware_latent)
        return wm_next_latent

    def wm_group_prediction(self, view_query_feat, cur_waypoint_group):
        '''
        Designed for provide current state and trajectory sampled from a policy, predict the future state for each trajs in the group
        view_query_feat (Tensor): [B, num_of_token, hidden_dim]
        cur_waypoint_group (Tensor): [B, group_size, 6, 2]
        '''
        batch_size, num_tokens, hidden_dim = view_query_feat.shape
        group_size = cur_waypoint_group.size(1)

        view_query_feat_expand = view_query_feat.unsqueeze(1).expand(-1, group_size, -1, -1) # [B, group_size, num_of_token, hidden_dim]
        view_query_feat_flat = view_query_feat_expand.reshape(batch_size*group_size, num_tokens, hidden_dim) # [B*group_size, num_of_token, hidden_dim]
        cur_waypoint_group = cur_waypoint_group.reshape(batch_size*group_size, 1, -1).expand(-1, num_tokens, -1) # [B*group_size, num_of_token, 12]

        cur_view_query_feat_with_ego = torch.cat([view_query_feat_flat, cur_waypoint_group], dim=-1) 
        action_aware_latent = self.action_aware_encoder(cur_view_query_feat_with_ego) # [B*group_size, num_of_token, hidden_dim]

        wm_next_latent = self._wm_decoder(action_aware_latent, action_aware_latent) # [B*group_size, num_of_token, hidden_dim]
        wm_next_latent = wm_next_latent.reshape(batch_size, group_size, num_tokens, hidden_dim) # [B, group_size, num_of_token, hidden_dim]
        return wm_next_latent

    def loss_3d(self, 
            preds_ego_future_traj,
            gt_ego_future_traj,
            gt_ego_future_traj_mask,
            ego_info=None,
            ):
        loss_waypoint = self.loss_plan_reg(preds_ego_future_traj, gt_ego_future_traj, gt_ego_future_traj_mask)
        return loss_waypoint

@HEADS.register_module()
class WaypointHead_RL(BaseModule):
    def __init__(self,
                num_proposals=6,
                #MHA
                hidden_channel=256,
                dim_feedforward=1024,
                num_heads=8,
                dropout=0.0,
                #pos embedding
                depth_step=0.8,
                depth_num=64,
                depth_start = 0,
                position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                stride=32,
                num_views=6,
                #others
                train_cfg=None,
                test_cfg=None,
                use_wm=False,
                num_spatial_token=36,
                num_tf_layers=2,
                num_traj_modal=1,
                group_size=32,
                simple_gaussian=True,
                min_std_list=None,
                max_std_list=None,
                max_abs_rho=None,
                debug_std=False,
                use_critic=False,
                critic=None,
                **kwargs,
                ):
        """
        use to predict the waypoints
        """
        super().__init__(**kwargs)
        self.use_wm = use_wm

        # query feature
        self.num_views = num_views
        self.num_proposals = num_proposals
        self.view_query_feat = nn.Parameter(torch.randn(1, self.num_views, hidden_channel, self.num_proposals))
        self.waypoint_query_feat = nn.Parameter(torch.randn(1, self.num_proposals, hidden_channel))

        # spatial attn
        spatial_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_channel,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        ) 
        
        self._spatial_decoder = nn.ModuleList( [
            nn.TransformerDecoder(spatial_decoder_layer, 1) 
            for _ in range(self.num_views)])

        # inv AR
        # example, no means masked
        # first output last point, then backward planning
        # yes yes yes
        #  no yes yes
        #  no  no yes
        self.causal_mask = torch.tril(torch.ones(self.num_proposals, self.num_proposals), diagonal=-1).bool()
        self.auto_regression_attention = nn.MultiheadAttention(embed_dim=hidden_channel, num_heads=8, batch_first=True)

        # wp_attn
        wp_decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_channel,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        self.wp_attn = nn.TransformerDecoder(wp_decoder_layer, 1) # input: Bz, num_token, d_model

        # world model
        if self.use_wm:
            self.action_aware_encoder = nn.Sequential(
            nn.Linear(hidden_channel + 6*2, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
            )

            wm_decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_channel,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            
            self._wm_decoder = nn.TransformerDecoder(wm_decoder_layer, num_tf_layers)
            self.loss_plan_rec = nn.MSELoss()

        # head
        self.num_traj_modal = num_traj_modal
        self.waypoint_head = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, hidden_channel),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channel, self.num_traj_modal* 2)
            )

        self.simple_gaussian = simple_gaussian
        if simple_gaussian:
            # only model std
            self.waypoint_cov_head = nn.Sequential(
                    nn.Linear(hidden_channel, hidden_channel),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_channel, hidden_channel),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_channel, self.num_traj_modal)
                )
        else:
            # model std_x, std_y, and rho
            self.waypoint_cov_head = nn.Sequential(
                    nn.Linear(hidden_channel, hidden_channel),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_channel, hidden_channel),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_channel, self.num_traj_modal*3)
                )
        
        self.min_std_list = min_std_list
        self.max_std_list = max_std_list
        self.max_abs_rho = max_abs_rho
        self.debug_std = debug_std

        self.group_size = group_size

        # position embedding
        ##img pos embed
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.depth_start = depth_start
        self.stride = stride

        self.position_encoder = nn.Sequential(
                nn.Linear(self.position_dim, hidden_channel*4),
                nn.ReLU(),
                nn.Linear(hidden_channel*4, hidden_channel),
            )

        self.pc_range = nn.Parameter(torch.tensor(point_cloud_range), requires_grad=False)
        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        # LID depth
        index = torch.arange(start=0, end=self.depth_num, step=1).float()
        index_1 = index + 1
        bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
        coords_d = self.depth_start + bin_size * index * index_1
        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        # reward function
        self.drivable_area_compliance_op = DrivableAreaConstrain()
        self.imitation_scorer = ImitationConstrain()
        self.collision_scorer = CollsionConstrain_RL()

        self.use_critic = use_critic
        if use_critic:
            self.critic = builder.build_backbone(critic)
            self.refer_critic = copy.deepcopy(self.critic)
            for p in self.refer_critic.parameters(): 
                p.requires_grad = False
            self.refer_critic.eval()
            self.gamma = self.critic.gamma

    def prepare_location(self, img_metas, img_feats):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = img_feats.shape[:2]
        x = img_feats.flatten(0, 1)
        location = get_locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location
    
    def img_position_embeding(self, img_feats, img_metas):
        """
        from streampetr
        """
        eps = 1e-5
        B, num_views, C, H, W = img_feats.shape
        assert num_views == self.num_views, 'num_views should be equal to self.num_views'
        BN = B * num_views
        num_sample_tokens = num_views * H * W
        LEN = num_sample_tokens
        img_pixel_locations = self.prepare_location(img_metas, img_feats)

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        img_pixel_locations[..., 0] = img_pixel_locations[..., 0] * pad_w
        img_pixel_locations[..., 1] = img_pixel_locations[..., 1] * pad_h

        # Depth
        D = self.coords_d.shape[0]
        pixel_centers = img_pixel_locations.detach().view(B, LEN, 1, 2).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([pixel_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        lidar2img = torch.from_numpy(np.stack(img_metas[0]['lidar2img'])).to(img_feats.device).float()
        lidar2img = lidar2img[:num_views]
        img2lidars = lidar2img.inverse()
        img2lidars = img2lidars.view(num_views, 1, 1, 4, 4).repeat(B, H*W, D, 1, 1).view(B, LEN, D, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3]) #normalize
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d) #(B, num_views*H*W, 3*64)
        coords_position_embeding = self.position_encoder(pos_embed)
        return coords_position_embeding
    
    def forward(self, img_feat, img_metas, ego_info=None, is_test=False):
        # init
        losses = {}
        Bz, num_views, num_channels, height, width = img_feat.shape
        init_view_query_feat = self.view_query_feat.clone().repeat(Bz, 1, 1, 1).permute(0, 1, 3, 2)
        init_waypoint_query_feat = self.waypoint_query_feat.clone().repeat(Bz, 1, 1)

        # img pos emb
        img_pos = self.img_position_embeding(img_feat, img_metas)
        img_pos = img_pos.reshape(Bz, num_views, height, width, num_channels)
        img_pos = img_pos.permute(0, 1, 4, 2, 3)
        img_feat_emb = img_feat + img_pos   

        # spatial view feat
        img_feat_emb = img_feat_emb.reshape(Bz, num_views, num_channels, height*width).permute(0, 1, 3, 2)
        spatial_view_feat = torch.zeros_like(init_view_query_feat)
        for i in range(self.num_views):
            spatial_view_feat[:, i] = self._spatial_decoder[i](init_view_query_feat[:, i], img_feat_emb[:, i])

        batch_size, num_view, num_tokens, num_channel = spatial_view_feat.shape
        spatial_view_feat = spatial_view_feat.reshape(batch_size, -1, num_channel)

        # predict wp
        updated_waypoint_query_feat = self.wp_attn(init_waypoint_query_feat, spatial_view_feat) #final_view_feat.shape torch.Size([1, 1440, 256])

        # inv AR
        if self.causal_mask.device == torch.device('cpu'):
            self.causal_mask = self.causal_mask.cuda()

        updated_waypoint_query_feat, _ = self.auto_regression_attention(
            query=updated_waypoint_query_feat,
            key=updated_waypoint_query_feat,
            value=updated_waypoint_query_feat,
            attn_mask=self.causal_mask,
            need_weights=False
        )

        cur_waypoint = self.waypoint_head(updated_waypoint_query_feat)
        cur_waypoint_cov = self.waypoint_cov_head(updated_waypoint_query_feat)

        if self.num_traj_modal > 1:
            assert self.num_traj_modal == 3
            bz, traj_len, _ = cur_waypoint.shape
            cur_waypoint = cur_waypoint.reshape(bz, traj_len, self.num_traj_modal, 2)
            if self.simple_gaussian:
                cur_waypoint_cov = cur_waypoint_cov.reshape(bz, traj_len, self.num_traj_modal)
            else:
                cur_waypoint_cov = cur_waypoint_cov.reshape(bz, traj_len, self.num_traj_modal, 3)
            ego_cmd = img_metas[0]['ego_fut_cmd'].to(img_feat.device)[0, 0]
            cur_waypoint = cur_waypoint[: ,: ,ego_cmd == 1].squeeze(2)
            cur_waypoint_cov = cur_waypoint_cov[:, :, ego_cmd == 1].squeeze(2)

        if self.simple_gaussian:
            # [B, T]
            cur_waypoint_std = F.softplus(cur_waypoint_cov)
            # make sure min/max are tensors on the same device
            min_std = torch.tensor(self.min_std_list, device=cur_waypoint_std.device).view(1, -1)
            max_std = torch.tensor(self.max_std_list, device=cur_waypoint_std.device).view(1, -1)

            cur_waypoint_std = torch.clamp(cur_waypoint_std, min=min_std, max=max_std)
            Sigma = torch.eye(2, device=cur_waypoint_std.device).unsqueeze(0).unsqueeze(0).repeat(bz, traj_len, 1, 1) # [B, T, 2, 2]
            Sigma = Sigma * (cur_waypoint_std**2).unsqueeze(-1).unsqueeze(-1)

            # get the std log
            if self.debug_std:
                std_log = {}
                for i in range(traj_len):
                    std_log[f'debug_std_frame_{i}'] = cur_waypoint_std[:,i].mean()
        else:
            # [B, T, 3]
            cur_waypoint_std_x = F.softplus(cur_waypoint_cov[:,:,0]) # [B, T]
            cur_waypoint_std_y = F.softplus(cur_waypoint_cov[:,:,1]) # [B, T]
            cur_waypoint_rho = torch.tanh(cur_waypoint_cov[:,:,2])*self.max_abs_rho  # [B, T]

            min_std = torch.tensor(self.min_std_list, device=cur_waypoint_std_x.device).view(1, -1)
            max_std = torch.tensor(self.max_std_list, device=cur_waypoint_std_x.device).view(1, -1)

            cur_waypoint_std_x = torch.clamp(cur_waypoint_std_x, min=min_std, max=max_std)
            cur_waypoint_std_y = torch.clamp(cur_waypoint_std_y, min=min_std, max=max_std)

            Sigma = torch.zeros((bz, traj_len, 2, 2), device=cur_waypoint_std_x.device)
            Sigma[:,:,0,0] = cur_waypoint_std_x**2
            Sigma[:,:,1,1] = cur_waypoint_std_y**2
            Sigma[:,:,0,1] = cur_waypoint_std_x * cur_waypoint_std_y * cur_waypoint_rho
            Sigma[:,:,1,0] = cur_waypoint_std_x * cur_waypoint_std_y * cur_waypoint_rho

            jitter = 1e-4 * torch.eye(2, device=Sigma.device).unsqueeze(0).unsqueeze(0)
            Sigma = Sigma + jitter

            # get the std log
            if self.debug_std:
                std_log = {}
                for i in range(traj_len):
                    std_log[f'debug_std_x_frame_{i}'] = cur_waypoint_std_x[:,i].mean()
                    std_log[f'debug_std_y_frame_{i}'] = cur_waypoint_std_y[:,i].mean()
                    std_log[f'debug_rho_frame_{i}'] = cur_waypoint_rho[:,i].mean()

        # world model prediction
        if self.use_wm:
            wm_next_latent = self.wm_prediction(spatial_view_feat, cur_waypoint)

        policy = MultivariateNormal(cur_waypoint, Sigma)

        if not self.debug_std or is_test:
            if self.use_wm:
                return policy, spatial_view_feat, wm_next_latent
            else:
                return policy, spatial_view_feat
        else:
            if self.use_wm:
                return policy, spatial_view_feat, wm_next_latent, std_log
            else:
                return policy, spatial_view_feat, std_log
    
    def loss_reconstruction(self, 
            reconstructed_view_query_feat,
            observed_view_query_feat,
            ):
        loss_rec = self.loss_plan_rec(reconstructed_view_query_feat, observed_view_query_feat)
        return loss_rec
    
    def wm_prediction(self, view_query_feat, cur_waypoint):
        batch_size, num_tokens, num_channel = view_query_feat.shape
        cur_waypoint = cur_waypoint.reshape(batch_size, 1, -1).repeat(1, num_tokens, 1)
        cur_view_query_feat_with_ego = torch.cat([view_query_feat, cur_waypoint], dim=-1) 
        action_aware_latent = self.action_aware_encoder(cur_view_query_feat_with_ego)

        wm_next_latent = self._wm_decoder(action_aware_latent, action_aware_latent)
        return wm_next_latent
    
    def loss_3d(self, 
            preds_ego_future_traj,
            gt_ego_future_traj,
            gt_ego_future_traj_mask,
            ego_info=None,
            ):
        loss_waypoint = self.loss_plan_reg(preds_ego_future_traj, gt_ego_future_traj, gt_ego_future_traj_mask)
        return loss_waypoint
    
    def loss_rl_bc(self, policy, gt_ego_fut_trajs, ego_fut_masks):
        gt_ego_fut_trajs = gt_ego_fut_trajs.squeeze(0) # [B, T, 2]
        ego_fut_masks = ego_fut_masks.squeeze(0).squeeze(0) # [B, T]

        log_prob = policy.log_prob(gt_ego_fut_trajs) # [B, T]
        log_prob = log_prob * ego_fut_masks # [B, T]

        loss_rl_bc = -log_prob.sum(dim=-1) # [B,]

        return loss_rl_bc.mean()

    def loss_rl_group_sampling(self, policy, gt_ego_fut_trajs, gt_bboxes_3d, gt_attr_labels, fut_valid_flag, ego_fut_masks):
        # step aware policy gradient with group sampling
        # parallel the step aware operation (training faster)
        bs, traj_len, _ = policy.mean.shape

        ego_fut_masks = ego_fut_masks.squeeze(0).squeeze(0) # [B, T]
        gt_ego_fut_trajs = gt_ego_fut_trajs.squeeze(0) # [B, T, 2]

        ego_fut_masks = ego_fut_masks.unsqueeze(1).unsqueeze(2).expand(bs, traj_len, self.group_size, traj_len).flatten(0, 2) # [bs*traj_len*group_size, T]
        gt_ego_fut_trajs = gt_ego_fut_trajs.unsqueeze(1).unsqueeze(2).expand(bs, traj_len, self.group_size, traj_len, 2).flatten(0, 2) # [bs*traj_len*group_size, 6, 2]
        
        assert bs == 1, "now only support batch_size = 1"
        # T gaussian distribution on x-y plane
        sample_traj_group = policy.rsample([self.group_size]) # [self.group_size, bs, T, 2]
        sample_traj_group = sample_traj_group.permute(1,0,2,3) # [bs, self.group_size, T, 2]
        
        mode_traj = policy.mean # [B, T, 2]

        traj_group = mode_traj.unsqueeze(1).unsqueeze(2).repeat(1, traj_len, self.group_size, 1, 1) # [bs, T, self.group_size, T, 2]

        for i in range(traj_len):
            traj_group[:,i,:,i,:] = sample_traj_group[:,:,i,:] # [B, T, self.group_size, T, 2]
        
        log_prob_group = policy.log_prob(traj_group) # [B, T, G, T]

        # ego planning metric
        traj_group = traj_group.flatten(0, 2) # [B*T*group_size, 6, 2]

        no_colision_score = self.collision_scorer(traj_group.cumsum(dim=-2), gt_ego_fut_trajs.cumsum(dim=-2).detach(), gt_bboxes_3d, gt_attr_labels, fut_valid_flag).reshape(bs, traj_len, self.group_size, traj_len) # [bs, T, self.group_size, T]
        dac_score = self.drivable_area_compliance_op(traj_group.cumsum(dim=-2).detach()).reshape(bs, traj_len, self.group_size) # [bs, T, self.group_size]
        imitation_score = self.imitation_scorer(traj_group.detach(), gt_ego_fut_trajs, ego_fut_masks).reshape(bs, traj_len, self.group_size, traj_len) # [bs, T, self.group_size, T]

        reward = no_colision_score * dac_score.unsqueeze(-1) * imitation_score # [B, T, self.group_size, T]

        loss_rl = 0

        for i in range(traj_len):
            # for each frame, actually we only care about traj_group[:,i,:,:i+1,:]
            step_aware_reward = reward[:,i,:,:i+1].sum(dim=-1) # [B, self.group_size, i+1] => [bs, self.group_size]
            step_aware_advantage = (step_aware_reward - step_aware_reward.mean(dim=-1, keepdim=True)) / (step_aware_reward.std(dim=-1, keepdim=True) + 1e-6)
            step_aware_log_prob_group = log_prob_group[:,i,:,:i+1].sum(dim=-1) # [bs, self.group_size, i+1] => [bs, self.group_size]
        
            step_aware_loss_rl = - step_aware_advantage.detach() * step_aware_log_prob_group # [bs, self.group_size]
            loss_rl += step_aware_loss_rl.mean()

        return loss_rl

    def compute_actor_loss(self, traj_group, policy, gt_ego_fut_trajs, gt_bboxes_3d, gt_attr_labels, fut_valid_flag, ego_fut_masks, cur_value, pred_fut_value):
        # used for the version that use world model and critic
        # new! in world model + critic version, we use cumsum for imitation_score for both loss_actor and competition
        # new! in world model + critic version, we care about all points of a traj when we conduct step aware policy gradient
        # parallel the step aware operation (training faster)
        bs, traj_len, _ = policy.mean.shape

        ego_fut_masks = ego_fut_masks.squeeze(0).squeeze(0) # [B, T]
        gt_ego_fut_trajs = gt_ego_fut_trajs.squeeze(0) # [B, T, 2]

        ego_fut_masks = ego_fut_masks.unsqueeze(1).unsqueeze(2).expand(bs, traj_len, self.group_size, traj_len).flatten(0, 2) # [B*T*G, T]
        gt_ego_fut_trajs = gt_ego_fut_trajs.unsqueeze(1).unsqueeze(2).expand(bs, traj_len, self.group_size, traj_len, 2).flatten(0, 2) # [B*T*G, 6, 2]
        
        assert bs == 1, "now only support batch_size = 1"
        # T gaussian distribution on x-y plane
        
        log_prob_group = policy.log_prob(traj_group) # [B, T, G, T]

        # ego planning metric
        traj_group = traj_group.flatten(0, 2) # [B*T*G, 6, 2]

        no_colision_score = self.collision_scorer(traj_group.cumsum(dim=-2).detach(), gt_ego_fut_trajs.cumsum(dim=-2).detach(), gt_bboxes_3d, gt_attr_labels, fut_valid_flag).reshape(bs, traj_len, self.group_size, traj_len) # [B, T, G, T]
        dac_score = self.drivable_area_compliance_op(traj_group.cumsum(dim=-2).detach()).reshape(bs, traj_len, self.group_size) # [B, T, G]
        imitation_score = self.imitation_scorer(traj_group.cumsum(dim=-2).detach(), gt_ego_fut_trajs.cumsum(dim=-2).detach(), ego_fut_masks).reshape(bs, traj_len, self.group_size, traj_len) # [B, T, G, T]

        reward = no_colision_score * dac_score.unsqueeze(-1) * imitation_score # [B, T, G, T]

        loss_actor = 0

        # for each frame, actually we care about all traj_group! This is different from the version use_critic=False
        step_aware_reward = reward.sum(dim=-1) # [B, T, G]
        step_aware_critic_reward = (step_aware_reward + self.gamma * pred_fut_value) - cur_value # [B, T, G]
        step_aware_critic_advantage = (step_aware_critic_reward - step_aware_critic_reward.mean(dim=-1, keepdim=True)) / (step_aware_critic_reward.std(dim=-1, keepdim=True) + 1e-6)
        step_aware_log_prob_group = log_prob_group.sum(dim=-1) # [B, T, G]
    
        step_aware_loss_rl = - step_aware_critic_advantage.detach() * step_aware_log_prob_group # [B, T, G]
        loss_actor += step_aware_loss_rl.mean()

        return loss_actor, reward

    def compute_critic_loss(self, reward, cur_value, pred_fut_value):
        '''
        reward (Tensor): [B, T, G, T]
        cur_value (Tensor): [B, T, G]
        pred_fut_value (Tensor): [B, T, G]
        '''
        reward_sum = reward.sum(dim=-1) # [B, T, G]
        loss_critic = self.critic.compute_critic_loss(reward_sum.detach(), cur_value, pred_fut_value.detach())
        return loss_critic
