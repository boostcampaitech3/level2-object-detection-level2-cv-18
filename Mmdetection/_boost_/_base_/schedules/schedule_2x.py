# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(
    type='AdamW',
    lr=1e-5,
    betas=(0.9,0.999),
    weight_decay=0.05)

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineRestart', # policy
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    periods = [1, 12, 12, 12],
    restart_weights = [1, 1, 0.5, 0.5],
    min_lr=1e-6)

    # step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=36)
