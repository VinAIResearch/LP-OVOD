_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        style='caffe'),
    neck=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True)
    ),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            in_channels=256,
            ensemble=True,
            fc_out_channels=1024,
            roi_feat_size=7,
            with_cls=False,
            num_classes=48,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    test_cfg = dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.000025)
lr_config = dict(step=[13, 18])
total_epochs = 20
evaluation = dict(interval=2,metric=['bbox'])
checkpoint_config = dict(interval=1, create_symlink=True)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
    )
