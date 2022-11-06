_base_ = "./faster_rcnn_swinv2_fishnet.py"

# OVERRIDE: machine-specific location
pretrained = "/local/scratch/stevens.994/hierarchical-vision/pretrained-checkpoints/swin_base_patch4_window7_224.pth"

# OVERRIDE: machine-specific location
work_dir = f"/local/scratch/stevens.994/hierarchical-vision/object-detection/{{ fileBasenameNoExtension }}"

# OVERRIDE: using swin v1 rather than v2
model = dict(
    type="FasterRCNN",
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
)

data = dict(
    # OVERRIDE: machine-specific (GPU memory)
    samples_per_gpu=2,
    workers_per_gpu=2,
)
