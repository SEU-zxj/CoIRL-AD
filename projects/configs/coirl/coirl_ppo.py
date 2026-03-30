_base_ = ['./coirl.py']

model = dict(
    rl_method='PPO',
    rl_actor_use_bc=True,
    pts_bbox_head_rl=dict(
        use_critic=True,
        critic=dict(
            gamma=0.2,
        ),
    ),
    ppo_gamma=0.2,
    ppo_adv_clip=5.0,
)
