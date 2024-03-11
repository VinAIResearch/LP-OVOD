PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=8 --master_port=$PORT tools/train.py \
    configs/coco/vild_sigmoid.py --launcher "pytorch" \
    --work-dir workdirs/vild_sigmoid \
    --no-validate \
    --cfg-options \
        load_from="weights/current_mmdetection_Head.pth" \
        model.roi_head.temperature_novel=0.01 \
        model.roi_head.prompt_path="ovd_coco_text_embedding.pth" \
        model.roi_head.kd_weight=256 \
        model.roi_head.clip_root="weights" \
        model.roi_head.bbox_head.ensemble=True \
        model.roi_head.proposal_id_map=proposals/train_coco_id_map.json \
        model.roi_head.feature_path=coco_clip_emb_train.pth
