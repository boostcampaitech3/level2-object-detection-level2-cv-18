checkpoint_config = dict(interval=9)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),  
        dict(type='WandbLoggerHook',interval=1000,
            init_kwargs=dict(
                project='Cascade_swin_t',
                entity = 'cv18',
                name = 'train_pseudo_fold'
                ),
            ),
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(save_best='bbox_mAP', classwise=True, metric=['bbox'])
runner = dict(type='EpochBasedRunner', max_epochs=24)
work_dir = './work_dirs/swin_b'
gpu_ids = range(0, 1)