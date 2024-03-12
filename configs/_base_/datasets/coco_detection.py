dataset_type = "CocoDataset"
data_root = "data/coco/"
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_norm_cfg = dict(mean=[122.770935, 116.74601, 104.093735], std=[68.500534, 66.63216, 70.323166], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="LoadProposals", num_max_proposals=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Resize",
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768), (1333, 800)],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "proposals", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadProposals", num_max_proposals=None),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Pad", size_divisor=32),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img", "proposals", "objectness"]),
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/ovd_ins_train2017_b.json",
        proposal_file="proposals/train_coco_proposals.pkl",
        proposal_id_map="proposals/train_coco_id_map.json",
        img_prefix=data_root + "train2017/",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/ovd_ins_val2017_all.json",
        proposal_file="proposals/val_coco_proposals.pkl",
        proposal_id_map="proposals/val_coco_id_map.json",
        img_prefix=data_root + "val2017/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/ovd_ins_val2017_all.json",
        proposal_file="proposals/val_coco_proposals.pkl",
        proposal_id_map="proposals/val_coco_id_map.json",
        img_prefix=data_root + "val2017/",
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric="bbox")
