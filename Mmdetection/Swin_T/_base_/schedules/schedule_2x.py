optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9,0.999),
    weight_decay=0.05)

optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineRestart', 
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    periods = [1, 12, 12, 12],
    restart_weights = [1, 1, 0.5, 0.5],
    min_lr=1e-5)

runner = dict(type='EpochBasedRunner', max_epochs=24)
