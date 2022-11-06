_base_ = "./faster_rcnn_swinv2_fishnet.py"

# OVERRIDE: machine-specific location
pretrained = "/mnt/10tb/models/hierarchical-vision/pretrained-checkpoints/swinv2_base_patch4_window8_256.pth"

# OVERRIDE: machine-specific location
work_dir = f"/mnt/10tb/models/hierarchical-vision/object-detection/{{ fileBasenameNoExtension }}"

model = dict(
    type="FasterRCNN",
    backbone=dict(
        type="SwinTransformerV2",
        pretrained_window_sizes=[8, 8, 8, 8],
        pretrained=pretrained,
    ),
)


data = dict(
    # OVERRIDE: machine-specific (GPU memory)
    samples_per_gpu=1,
    workers_per_gpu=2,
)
