_base_ = "./faster_rcnn_swinv2_fishnet.py"

# OVERRIDE: machine-specific location
pretrained = "/mnt/10tb/models/hierarchical-vision/pretrained-checkpoints/groovy-grape-192-latest.pth"

# OVERRIDE: machine-specific location
work_dir = f"/mnt/10tb/models/hierarchical-vision/object-detection/{{ fileBasenameNoExtension }}"

# OVERRIDE: include pretrained key
model = dict(
    type="FasterRCNN",
    backbone=dict(
        type="SwinTransformerV2",
        pretrained=pretrained,
    ),
)

data = dict(
    # OVERRIDE: machine-specific (GPU memory)
    samples_per_gpu=1,
    workers_per_gpu=2,
)
