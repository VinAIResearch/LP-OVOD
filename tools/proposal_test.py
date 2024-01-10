from mmdet.datasets import build_dataset, build_dataloader
from mmdet.datasets import LVISV1Dataset, LVISV1SplitDataset, CocoDataset
from mmcv import Config, DictAction
from tqdm import tqdm

# Build dataset

# data_root = "data/coco/"
# data_val = CocoDataset(
#     ann_file=data_root + 'annotations/ovd_ins_val2017_t.json',
#     proposal_file='data/coco/annotations/oln/oln_proposals.pkl',
#     img_prefix=data_root + 'val2017/',
#     proposal_id_map='data/coco/annotations/oln/oln_id_map.json',
#     pipeline=[],
#     test_mode=True,
# )
# data_val.evaluate(data_val.proposals, metric='proposal_fast')

data_root = "data/lvis_v1/"
# data_val = LVISV1Dataset(
#     ann_file=data_root + 'annotations/lvis_v1_val.json',
#     proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_val.pkl',
#     img_prefix=data_root,
#     pipeline=[],
#     test_mode=True,
# )
# data_val.evaluate(data_val.proposals, metric='proposal_fast')

data_val = LVISV1SplitDataset(
    ann_file=data_root + 'annotations/lvis_v1_val.json',
    proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_val.pkl',
    img_prefix=data_root,
    pipeline=[],
    img_count_lbl=["r"]
)
data_val.evaluate(data_val.proposals, metric='proposal_fast')

# data_val = CocoDataset(
#     ann_file=data_root + 'annotations/ovd_ins_val2017_t.json',
#     proposal_file=data_root + 'proposals/instances_val2017_proposals.pkl',
#     img_prefix=data_root + 'val2017/',
#     proposal_id_map=data_root + 'annotations/val_proposal_id_map.json',
#     pipeline=[],
#     test_mode=True,
# )
# data_val.evaluate(data_val.proposals, metric='proposal_fast')

# data_val = CocoDataset(
#     ann_file=data_root + 'annotations/ovd_ins_val2017_b.json',
#     proposal_file=data_root + 'proposals/val_proposals_lvis_rpn.pkl',
#     img_prefix=data_root + 'val2017/',
#     proposal_id_map=data_root + 'annotations/val_proposal_id_map_lvis_rpn.json',
#     pipeline=[],
#     test_mode=True,
# )
# data_val.evaluate(data_val.proposals, metric='proposal_fast')

# data_val = CocoDataset(
#     ann_file=data_root + 'annotations/ovd_ins_val2017_t.json',
#     proposal_file=data_root + 'proposals/val_proposals_lvis_rpn.pkl',
#     img_prefix=data_root + 'val2017/',
#     proposal_id_map=data_root + 'annotations/val_proposal_id_map_lvis_rpn.json',
#     pipeline=[],
#     test_mode=True,
# )
# data_val.evaluate(data_val.proposals, metric='proposal_fast')