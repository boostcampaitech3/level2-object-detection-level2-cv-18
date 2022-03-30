# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
    type='OneOf', # 이 중 하나 선택해서 실행
    transforms=[
        dict(type='Flip',p=1.0), # 가로, 세로 또는 둘다 뒤집는 것 중 하나
        dict(type='RandomRotate90',p=1.0) # 무작위로 사진을 90도 회전
    ],
    p=0.5), # Oneof 라는 dict 자체가 실행될 확률
    dict(type='RandomResizedCrop',height=1024, width=1024, scale=(0.5, 1.0), p=0.5), # 랜덤하게 잘라서 resize(1024,1024), 전체 이미지의 50% ~ 100% 범위
    # dict(type='RandomBrightnessContrast',brightness_limit=0.1, contrast_limit=0.15, p=0.5), # 랜덤하게 임의의 밝기 조정
    # dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5), # 색조, 채도, 값 변경
    # dict(type='GaussNoise', p=0.3), # 가우스 노이즈 추가
    # dict(type="GaussianBlur",p=0.3), # 가우스 필터를 사용한 blur처리
    # dict(
    # type='OneOf',
    # transforms=[
    #     dict(type='Blur', p=1.0),
    #     dict(type='GaussianBlur', p=1.0),
    #     dict(type='MedianBlur', blur_limit=5, p=1.0),
    #     dict(type='MotionBlur', p=1.0)
    # ],
    # p=0.1)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='AutoAugment',
    #     policies=[[dict(type='Resize',
    #                     img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                                (736, 1333), (768, 1333), (800, 1333)],
    #                     multiscale_mode='value',
    #                     keep_ratio=True)],
    #               [dict(type='Resize',
    #                     img_scale=[(400, 1333), (500, 1333), (600, 1333)],
    #                     multiscale_mode='value',
    #                     keep_ratio=True),
    #                dict(type='RandomCrop',
    #                     rop_type='absolute_range',
    #                     crop_size=(384, 600),
    #                     allow_negative_crop=True),
    #                dict(type='Resize',
    #                     img_scale=[(480, 1333), (512, 1333), (544, 1333),
    #                                (576, 1333), (608, 1333), (640, 1333),
    #                                (672, 1333), (704, 1333), (736, 1333),
    #                                (768, 1333), (800, 1333)],
    #                     multiscale_mode='value',
    #                     override=True,
    #                     keep_ratio=True)]]
    #     ),
    
    # dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Albu',transforms=albu_train_transforms,
         bbox_params=dict(type='BboxParams',
                          format='pascal_voc',
                          label_fields=['gt_labels'],
                          min_visibility=0.0,
                          filter_lost_elements=True),
         keymap={'img': 'image',
                 'gt_bboxes': 'bboxes'},
         update_pad_shape=False,
         skip_img_without_anno=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# val_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

# evaluation = dict(interval=12, classwise=True,metric='bbox',save_best='bbox_mAP')
