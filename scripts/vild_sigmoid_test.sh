python -m torch.distributed.launch --nproc_per_node=1 ./tools/test.py \
    "configs/coco/vild_sigmoid_ft.py" \
    workdirs/vild_sigmoid_ft/latest.pth --launcher "pytorch" \
    --eval bbox \
    --cfg-options \
        model.roi_head.prompt_path="ovd_coco_text_embedding.pth" \
        model.roi_head.temperature_test=0.01 \
        model.roi_head.alpha=0. \
        model.roi_head.beta=0.8 \
        model.roi_head.clip_root="weights" \
        model.roi_head.bbox_head.ensemble=True \
        data.test.proposal_file=checkpoints/proposals/rpn_rgb.pkl \
        data.test.proposal_id_map=checkpoints/proposals/oln_id_map.json \
    --eval-options \
        classwise=True