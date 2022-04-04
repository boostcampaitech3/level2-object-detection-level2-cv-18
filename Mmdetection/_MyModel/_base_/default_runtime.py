checkpoint_config = dict(interval=20)
# yapf:disable
# evaluation = dict(interval=1, classwise=True,metric='bbox',save_best='bbox_mAP')

# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook', interval=250),
        ###################################
        dict(type='WandbLoggerHook',interval=1000,
            init_kwargs=dict(
                project='JSH',
                entity = 'cv18',
                name = 'aug_set2_fpncarafe'
            ),
            )
    ])