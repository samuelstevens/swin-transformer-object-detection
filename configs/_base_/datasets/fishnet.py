_base_ = ["./coco_detection.py"]

dataset_type = "CocoDataset"
data_root = "/mnt/10tb/data/fishnet/coco/"
img_norm_cfg = dict(
    mean=[96.61452949576494, 101.77696375818135, 106.9601640305532],
    std=[56.92885560916941, 60.42261287077057, 66.30769925921233],
    to_rgb=True,
)


classes = (
    "Human",
    "Unknown",
    "Albacore",
    "Yellowfin tuna",
    "Shortbill spearfish",
    "Opah",
    "Swordfish",
    "Skipjack tuna",
    "No fish",
    "Mahi mahi",
    "Indo Pacific sailfish",
    "Striped marlin",
    "Pomfret",
    "Bigeye tuna",
    "Great barracuda",
    "Wahoo",
    "Tuna",
    "Sickle pomfret",
    "Blue marlin",
    "Shark",
    "Marlin",
    "Escolar",
    "Long snouted lancetfish",
    "Lancetfish",
    "Oilfish",
    "Black marlin",
    "Roudie scolar",
    "Water",
    "Thresher shark",
    "Pelagic stingray",
    "Mola mola",
    "Brama",
    "Snake mackerel",
    "Rainbow runner",
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "train/annotations.json",
        img_prefix=data_root + "train/",
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val/annotations.json",
        img_prefix=data_root + "val/",
        classes=classes,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test/annotations.json",
        img_prefix=data_root + "test/",
        classes=classes,
    ),
)
evaluation = dict(metric=["bbox"])
