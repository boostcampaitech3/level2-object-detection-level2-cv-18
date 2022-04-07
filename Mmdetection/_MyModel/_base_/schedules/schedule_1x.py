# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=488,
    warmup_ratio=0.001,
    periods=[1, 10, 10, 10, 10, 10],
    restart_weights=[1,1,0.8,0.6,0.5,0.3],
    min_lr=0.000001
)
runner = dict(type='EpochBasedRunner', max_epochs=50)
