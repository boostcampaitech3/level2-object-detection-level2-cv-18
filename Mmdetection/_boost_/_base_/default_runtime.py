checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook',interval=50),
        dict(type='WandbLoggerHook',interval=1000,
            init_kwargs=dict(
            project = 'cascade_rcnn_swin',
            entity = 'cv18',
            name = 'swins + swint',
            ),
            )
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None

# workflow = [('train', 1), ('val', 1)]
workflow = [('train', 1)]

evaluation = dict(interval=12, classwise=True,metric='bbox',save_best='bbox_mAP')