# optimizer
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])
# total_epochs = 12
lr_config = dict(policy="CosineAnnealing", warmup="linear", warmup_iters=500, warmup_ratio=0.001, min_lr_ratio=2e-4)
