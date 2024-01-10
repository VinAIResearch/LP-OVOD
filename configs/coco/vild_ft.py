_base_ = './vild_coco_20e_faster_rcnn.py'

model = dict(
    type='FasterRCNNFreezeBackbone',
    roi_head=dict(
        type="StandardRoIHeadFinetune",
        train_temp=0.1,
        test_temp=0.07,
        neg_pos_ub=20,
        bbox_head=dict(num_classes=65)))


dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/ovd_ins_val2017_b.json',
        proposal_file='checkpoints/proposals/instances_train2017_proposals.pkl',
        img_prefix=data_root + 'train2017/',
        proposal_id_map='checkpoints/proposals/train_proposal_id_map.json',
        pipeline=train_pipeline))
optimizer = dict(lr=0.001)
total_epochs = 12
lr_config = dict(
    warmup_iters=20,
    warmup_ratio=0.1,
    step=[8, 11])

evaluation = dict(interval=24,metric=['bbox'])
checkpoint_config = dict(interval=12, create_symlink=True)
find_unused_parameters=True
