import torch
import torch.nn as nn
import torch.distributed as dist
from projects.mmdet3d_plugin.CoIRL.utils import CollsionConstrain, ImitationConstrain

class CompetitiveLearningMachine_v3:
    def __init__(self, il_actor, rl_actor, use_critic, max_threshold=10.0, min_threshold=1.0, competition_batch_size=100, swap_percentage=0.5, competition_warmup_flag=False, competition_warmup_threshold=5000):
        '''
        we will sum the score of il_actor and rl_actor for `competition_batch_size` data.
        if abs(il_score - rl_score) >= max_threshold, directly cover the params of actor perform worse with the better one
        if abs(il_score - rl_score) <= min_threshold, use soft interpolation with swap_percentage: worse_actor.param = swap_percentage * better_actor.param + (1 - swap_percentage) * worse_actor.param
        else, do nothing
        '''
        super().__init__()
        self.il_actor = il_actor
        self.rl_actor = rl_actor
        self.use_critic = use_critic

        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.competition_batch_size = competition_batch_size
        self.swap_percentage = swap_percentage

        self.competition_warmup_flag = competition_warmup_flag
        self.competition_warmup_threshold = competition_warmup_threshold

        self.collision_scorer = CollsionConstrain()
        self.imitation_scorer = ImitationConstrain()

        self.iter = 0
        self.il_score_sum = torch.zeros(1)
        self.rl_score_sum = torch.zeros(1)
        self.n_il_win = torch.zeros(1)
        self.n_rl_win = torch.zeros(1)

        # these are just for logging
        self.il_score_sum_last_comp = torch.zeros(1)
        self.rl_score_sum_last_comp = torch.zeros(1)
        self.score_diff_last_comp = torch.zeros(1)

        self.learning_parameter_list = ['waypoint_query_feat', 'waypoint_pos_feat', 'navi_embedding']
        self.learning_layer_list = ['navi_se', 'tokenlearner', 'latent_decoder', 'auto_regression_attention', 'wp_attn', 'waypoint_head']

    def set_refer_critic(self):
        refer_critic = self.rl_actor.refer_critic
        critic = self.rl_actor.critic

        # refer_critic.set_weight(critic)
        refer_critic.set_weight_ema(critic)

    def cache_comp_stats(self, il_score_sum, rl_score_sum, score_diff):
        device = next(self.il_actor.parameters()).device
        # convert scalar or tensor to single-element tensor on correct device
        self.il_score_sum_last_comp = torch.tensor([il_score_sum], device=device)
        self.rl_score_sum_last_comp = torch.tensor([rl_score_sum], device=device)
        self.score_diff_last_comp = torch.tensor([score_diff], device=device)


    def init_record(self):
        device = next(self.il_actor.parameters()).device
        self.il_score_sum = torch.zeros(1, device=device)
        self.rl_score_sum = torch.zeros(1, device=device)

    def give_score(self, pred_traj, gt_traj, gt_bboxes_3d, gt_attr_labels, fut_valid_flag, ego_fut_masks):
        '''
        pred_traj (Tensor): [B, T, 2]
        gt_traj (Tensor): [B, T, 2]
        '''
        collision_score = self.collision_scorer(pred_traj.cumsum(dim=-2), gt_traj.cumsum(dim=-2), gt_bboxes_3d, gt_attr_labels, fut_valid_flag) # [B, T]
        imitation_score = self.imitation_scorer(pred_traj, gt_traj, ego_fut_masks) # [B, T]

        collision_score = collision_score.all(dim=-1)
        imitation_score = imitation_score.mean(dim=-1)

        return collision_score * imitation_score # [B,]
    
    def swap_knowledge(self, worse_actor, better_actor, ratio):
        with torch.no_grad():
            for learning_param_name in self.learning_parameter_list:
                worse_actor_param = getattr(worse_actor, learning_param_name)
                better_actor_param = getattr(better_actor, learning_param_name)

                assert worse_actor_param.shape == better_actor_param.shape
                worse_actor_param.data.copy_(ratio * better_actor_param.data + (1 - ratio) * worse_actor_param.data)

            for learning_layer_name in self.learning_layer_list:
                worse_actor_layer = getattr(worse_actor, learning_layer_name)
                better_actor_layer = getattr(better_actor, learning_layer_name)

                for (p_worse_actor_layer, p_better_actor_layer) in zip(worse_actor_layer.parameters(), better_actor_layer.parameters()):
                    assert p_worse_actor_layer.shape == p_better_actor_layer.shape
                    p_worse_actor_layer.data.copy_(ratio * p_better_actor_layer.data + (1 - ratio) * p_worse_actor_layer.data)

    def all_reduce_tensor(self, tensor):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= dist.get_world_size()
        return tensor

    def competition(self):
        il_score = self.all_reduce_tensor(self.il_score_sum.clone())
        rl_score = self.all_reduce_tensor(self.rl_score_sum.clone())        

        score_diff = abs(il_score - rl_score)
        if score_diff.item() >= self.max_threshold:
            # hard param cover
            if rl_score.item() < il_score.item():
                self.swap_knowledge(worse_actor=self.rl_actor, better_actor=self.il_actor, ratio=1.0)
                self.n_il_win += 1
            else:
                self.swap_knowledge(worse_actor=self.il_actor, better_actor=self.rl_actor, ratio=1.0)
                self.n_rl_win += 1
        elif score_diff.item() <= self.min_threshold:
            pass
        else:
            # soft interpolation
            if rl_score.item() < il_score.item():
                self.swap_knowledge(worse_actor=self.rl_actor, better_actor=self.il_actor, ratio=self.swap_percentage)
                self.n_il_win += 1
            else:
                self.swap_knowledge(worse_actor=self.il_actor, better_actor=self.rl_actor, ratio=self.swap_percentage)
                self.n_rl_win += 1
        
        ret_dict = {
            'il_score_sum': il_score,
            'rl_score_sum': rl_score,
            'score_diff': il_score - rl_score,
            'il_win': self.n_il_win.cuda(),
            'rl_win': self.n_rl_win.cuda()
        }

        self.cache_comp_stats(il_score, rl_score, il_score - rl_score)

        self.init_record()

        if dist.is_initialized():
            for r in range(dist.get_world_size()):
                if dist.get_rank() == r:
                    print(f"[Rank {r}] il_score_sum={il_score.item()} rl_score_sum={rl_score.item()} score_diff={score_diff.item()} il_win={self.n_il_win.item()}, rl_win={self.n_rl_win.item()}")
                dist.barrier()  # wait for this rank to finish printing
        else:
            print(f"[Rank 0] il_win={self.n_il_win.item()}, rl_win={self.n_rl_win.item()}")

        return ret_dict

    def competitive_learning(self, il_actor_pred_traj, rl_actor_pred_traj, gt_traj, gt_bboxes_3d, gt_attr_labels, fut_valid_flag, ego_fut_masks):
        il_score = self.give_score(il_actor_pred_traj, gt_traj, gt_bboxes_3d, gt_attr_labels, fut_valid_flag, ego_fut_masks).mean()
        rl_score = self.give_score(rl_actor_pred_traj, gt_traj, gt_bboxes_3d, gt_attr_labels, fut_valid_flag, ego_fut_masks).mean()

        if self.il_score_sum.device == torch.device("cpu"):
            self.il_score_sum = self.il_score_sum.cuda()
        if self.rl_score_sum.device == torch.device("cpu"):
            self.rl_score_sum = self.rl_score_sum.cuda()

        self.il_score_sum += il_score
        self.rl_score_sum += rl_score

        self.iter += 1

        if self.use_critic:
            self.set_refer_critic()

        warmup_end_flag = (not self.competition_warmup_flag) or (self.competition_warmup_flag and self.iter > self.competition_warmup_threshold)

        if (self.iter % self.competition_batch_size == 0) and warmup_end_flag:
            ret_dict = self.competition()
            ret_dict.update({
                'il_score': il_score,
                'rl_score': rl_score,
            })
        else:
            ret_dict = {
                'il_score': il_score,
                'rl_score': rl_score,
                'il_score_sum': self.il_score_sum_last_comp.cuda(),
                'rl_score_sum': self.rl_score_sum_last_comp.cuda(),
                'score_diff': self.score_diff_last_comp.cuda(),
                'il_win': self.n_il_win.cuda(),
                'rl_win': self.n_rl_win.cuda()
            }

        return ret_dict