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
            name = 'psedo_cascade_swin_t',
            ),
            )
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = '/opt/ml/ysw/level2-object-detection-level2-cv-18/Mmdetection/_boost_/work_dirs/cascade_swin_fpn_1x/latest.pth'
resume_from = None

# workflow = [('train', 1), ('val', 1)]
workflow = [('train', 1)]
