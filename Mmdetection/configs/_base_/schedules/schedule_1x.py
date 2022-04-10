# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001, 
    betas=(0.9,0.999), 
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    ))
optimizer_config = dict(
    grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=5)

'''
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=488,
    warmup_ratio=0.001,
    periods=[1,4,8,12],
    restart_weights=[1,1,0.6,0.3],
    min_lr=0
)
'''

