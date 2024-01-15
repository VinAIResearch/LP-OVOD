_base_ = "./vild_ft.py"

model = dict(
    roi_head=dict(type="StandardRoIHeadSigmoidFinetune", test_temp=1.0, neg_pos_ub=20, bbox_head=dict(num_classes=65))
)
