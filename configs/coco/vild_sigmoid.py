_base_ = './vild_coco_20e_faster_rcnn.py'

model = dict(roi_head=dict(type="StandardRoIHeadSigmoid"))

lr_config = dict(step=[16, 19])
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=15, norm_type=2))

