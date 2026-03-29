_base_ = ['./coirl.py']

model = dict(
    # Keep GRPO path intact in base config; this file activates CQL branch only.
    rl_method='CQL',
    rl_actor_use_bc=True,
    # rl_actor_use_bc=False,
    # Recommended for cleaner CQL baseline behavior.
    disable_competition=False,
    # disable_competition=True,
    # CQL branch uses a dedicated Q critic, so legacy V critic is disabled here.
    pts_bbox_head_rl=dict(
        use_critic=False,
    ),
    cql_critic=dict(
        type='CQLSingleQCritic',
        hidden_dim=256,
        traj_len=6,
        num_heads=8,
        dropout=0.1,
        n_layer=2,
        gamma=0.99,
        ema_tau=0.9,
    ),
    cql_alpha=1.0,
    cql_num_action_samples=8,
    cql_random_action_scale=2.0,
)
