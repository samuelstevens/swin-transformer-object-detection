_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/datasets/fishnet.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

pretrained = "/mnt/10tb/models/hierarchical-vision/pretrained/fuzzy-fig-192-latest.pth"

model = dict(
    type="MaskRCNN",
    backbone=dict(
        _delete_=True,
        type="SwinTransformerV2",
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        pretrained_window_sizes=[12, 12, 12, 6],
        pretrained=pretrained,
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        type="StandardRoIHead",
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            num_classes=34,
        ),
    ),
)

img_norm_cfg = dict(
    mean=[96.614, 101.776, 106.960], std=[56.928, 60.422, 66.307], to_rgb=True
)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="Resize",
                    img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="Resize",
                    img_scale=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                    ],
                    multiscale_mode="value",
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
)

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=12)

auto_scale_lr = dict(enable=True, base_batch_size=16)
