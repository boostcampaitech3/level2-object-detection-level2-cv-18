# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/level2-object-detection-level2-cv-18/dataset/'
classes = ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
    type='OneOf',
    transforms=[
        dict(type='ChannelShuffle',p=1.0),
        dict(type='InvertImg',p=1.0),
    ], p=0.5
    ),
    dict(
    type='OneOf',
    transforms=[
        dict(type='Blur', blur_limit=33,p=1.0),
        dict(type='GaussianBlur', blur_limit=33,p=1.0),
        dict(type='MedianBlur', blur_limit=33, p=1.0),
        dict(type='MotionBlur', blur_limit=33,p=1.0)
    ], p=0.3
    ),
    dict(type='Cutout',num_holes=30, max_h_size=25, max_w_size=25, fill_value=255, p=0.5),
    dict(type='RandomRotate90',p=0.5),
    dict(type='RandomResizedCrop', height=1024, width=1024, scale=(0.3, 0.6), p=0.5),
    dict(type='HueSaturationValue', hue_shift_limit=50, sat_shift_limit=40, val_shift_limit=20, p=0.5), 
    
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms, ########## `albu_train_transforms`
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.1,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True
        ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024,1024),
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes = classes,
        ann_file=data_root + 'train_Kfold0.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = classes,
        ann_file=data_root + 'val_Kfold0.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes = classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
