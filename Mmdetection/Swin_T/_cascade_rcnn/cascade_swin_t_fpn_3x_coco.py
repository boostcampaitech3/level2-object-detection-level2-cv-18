_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection2.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(in_channels=[96, 192, 384, 768]))
    # neck=[
    #     dict(
    #         type='FPN',
            # in_channels=[96, 192, 384, 768]),
            # in_channels=[256, 512, 1024, 2048],
            # in_channels=[192, 384, 768, 1536],
        #     out_channels=256,
        #     start_level=1,
        #     add_extra_convs='on_output',
        #     num_outs=5),
        # dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=6)])


