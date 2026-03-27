_base_ = ['./coirl.py']

model = dict(
    # Disable only CLM actor competition/swap.
    # Critic and reference-critic EMA update remain active.
    disable_competition=True,
)
