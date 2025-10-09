import torch
from torch import nn
from mmdet.models import HEADS

@HEADS.register_module()
class Critic(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, n_layer, gamma, ema_tau):
        super().__init__()
        self.value_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        # world model
        value_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True,
        )
        self.value_decoder = nn.TransformerDecoder(value_decoder_layer, n_layer)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.mse_loss = nn.MSELoss()

        self.learning_parameter_list = ['value_query']
        self.learning_layer_list = ['value_decoder', 'value_head']

        self.gamma = gamma
        self.ema_tau = ema_tau


    def forward(self, state):
        '''
        state (Tensor): [B, n_token, hidden] (cur_state), [B, T*G, n_token, hidden] (fut_state, after sampling a group of action)
        '''
        if state.dim() == 3:
            bs, n_token, hidden = state.shape
            value_query = self.value_query.expand(bs, 1, hidden)
            value_feat = self.value_decoder(tgt=value_query, memory=state) # [B, 1, H]
            value = self.value_head(value_feat) # [B, 1, 1]
            value = value.reshape(bs) # [B,]
        elif state.dim() == 4:
            bs, TG, n_token, hidden = state.shape
            state = state.reshape(bs*TG, n_token, hidden) # [B*T*G, n_token, H]
            value_query = self.value_query.expand(bs*TG, 1, hidden) # [B*T*G, 1, H]
            value_feat = self.value_decoder(tgt=value_query, memory=state) # [B*T*G, 1, H]
            value = self.value_head(value_feat) # [B*T*G, 1, 1]
            value = value.reshape(bs, TG) # [B, T*G]
        else:
            raise NotImplementedError()

        return value
    
    def compute_critic_loss(self, reward, cur_value, pred_value):
        '''
        reward (Tensor): [B, T, G] (sum of all six point)
        cur_value (Tensor): [B, T, G]
        pred_value (Tensor): [B, T, G]
        '''
        target = reward + self.gamma * pred_value
        loss_critic = self.mse_loss(cur_value, target.detach())
        return loss_critic
    
    def set_weight(self, critic):
        '''
        self (Tensor): The reference critic (freeze weight to make sure the supervise loss is consistent)
        critic (Tensor): The training critic (provide weight)
        '''
        with torch.no_grad():
            for learning_param_name in self.learning_parameter_list:
                self_param = getattr(self, learning_param_name)
                critic_param = getattr(critic, learning_param_name)

                assert self_param.shape == critic_param.shape
                self_param.data.copy_(critic_param)

            for learning_layer_name in self.learning_layer_list:
                self_layer = getattr(self, learning_layer_name)
                critic_layer = getattr(critic, learning_layer_name)

                for (p_self_layer, p_critic_layer) in zip(self_layer.parameters(), critic_layer.parameters()):
                    assert p_self_layer.shape == p_critic_layer.shape
                    p_self_layer.data.copy_(p_critic_layer)

    def set_weight_ema(self, critic):
        '''
        self (Tensor): The reference critic (freeze weight to make sure the supervise loss is consistent)
        critic (Tensor): The training critic (provide weight)
        '''
        with torch.no_grad():
            for learning_param_name in self.learning_parameter_list:
                self_param = getattr(self, learning_param_name)
                critic_param = getattr(critic, learning_param_name)

                assert self_param.shape == critic_param.shape
                self_param.data.copy_(self.ema_tau * self_param + (1 - self.ema_tau) * critic_param)

            for learning_layer_name in self.learning_layer_list:
                self_layer = getattr(self, learning_layer_name)
                critic_layer = getattr(critic, learning_layer_name)

                for (p_self_layer, p_critic_layer) in zip(self_layer.parameters(), critic_layer.parameters()):
                    assert p_self_layer.shape == p_critic_layer.shape
                    p_self_layer.data.copy_(self.ema_tau * p_self_layer + (1 - self.ema_tau) * p_critic_layer)