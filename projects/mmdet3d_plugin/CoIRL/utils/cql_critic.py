import torch
from torch import nn
from mmdet.models import HEADS


@HEADS.register_module()
class CQLSingleQCritic(nn.Module):
    """Single-Q critic for continuous trajectory actions.

    Inputs:
      - state: [B, N_token, H]
      - action: [B, T, 2] or [B, K, T, 2]
    Output:
      - q: [B] or [B, K]
    """

    def __init__(
        self,
        hidden_dim,
        traj_len,
        num_heads,
        dropout,
        n_layer,
        gamma,
        ema_tau,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.traj_len = traj_len

        self.action_encoder = nn.Sequential(
            nn.Linear(traj_len * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        q_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.q_decoder = nn.TransformerDecoder(q_decoder_layer, n_layer)

        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.gamma = gamma
        self.ema_tau = ema_tau

        self.learning_layer_list = ["action_encoder", "q_decoder", "q_head"]

    def _forward_single(self, state, action):
        if state.dim() != 3:
            raise ValueError(f'Expected state shape [B, N, H], got {tuple(state.shape)}')
        if action.dim() != 3:
            raise ValueError(f'Expected action shape [B, T, 2], got {tuple(action.shape)}')

        bs, _, hidden = state.shape
        assert hidden == self.hidden_dim, "state hidden dim mismatch"
        assert action.size(1) == self.traj_len and action.size(2) == 2, "action shape mismatch"

        action = action.reshape(bs, self.traj_len * 2)
        action_query = self.action_encoder(action).unsqueeze(1)  # [B, 1, H]
        q_feat = self.q_decoder(tgt=action_query, memory=state)
        q_val = self.q_head(q_feat).reshape(bs)  # [B], one Q for one trajectory action
        return q_val

    def forward(self, state, action):
        if action.dim() == 3:
            # action: [B, T, 2] -> q: [B]
            return self._forward_single(state, action)

        if action.dim() == 4:
            bs, k, t, c = action.shape
            assert t == self.traj_len and c == 2
            state_expand = state.unsqueeze(1).expand(-1, k, -1, -1).reshape(bs * k, state.size(1), state.size(2))
            action_flat = action.reshape(bs * k, t, c)
            q_flat = self._forward_single(state_expand, action_flat)
            # action: [B, K, T, 2] -> q: [B, K]
            return q_flat.reshape(bs, k)

        raise NotImplementedError("action dim must be 3 or 4")

    def set_weight(self, critic):
        with torch.no_grad():
            for learning_layer_name in self.learning_layer_list:
                self_layer = getattr(self, learning_layer_name)
                critic_layer = getattr(critic, learning_layer_name)
                for p_self, p_critic in zip(self_layer.parameters(), critic_layer.parameters()):
                    p_self.data.copy_(p_critic)

    def set_weight_ema(self, critic):
        with torch.no_grad():
            for learning_layer_name in self.learning_layer_list:
                self_layer = getattr(self, learning_layer_name)
                critic_layer = getattr(critic, learning_layer_name)
                for p_self, p_critic in zip(self_layer.parameters(), critic_layer.parameters()):
                    p_self.data.copy_(self.ema_tau * p_self + (1 - self.ema_tau) * p_critic)
