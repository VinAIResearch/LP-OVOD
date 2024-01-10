python ./tools/train.py \
    configs/coco/vild_sigmoid_ft.py \
    --work-dir workdirs/vild_sigmoid_ft \
    --no-validate \
    --cfg-options \
        load_from=workdirs/vild_sigmoid/latest.pth \
        model.roi_head.prompt_path="ovd_coco_text_embedding.pth" \
        model.roi_head.temperature_test=0.01 \
        model.roi_head.alpha=0. \
        model.roi_head.beta=0.8 \
        model.roi_head.test_temp=1. \
        model.roi_head.neg_pos_ub=10 \
        model.roi_head.clip_root="weights" \
        model.roi_head.bbox_head.ensemble=True \
        data.train.dataset.ann_file=retrieval/retrieve_proposal_onl100.json
