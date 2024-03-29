import mmcv
import torch
from mmcv import Config, DictAction

import argparse
from mmdet.core import bbox2result
from mmdet.core.bbox.assigners.max_iou_assigner import MaxIoUAssigner
from mmdet.datasets import build_dataset
from torchvision.ops import batched_nms
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate metric of the " "results saved in pkl format")
    parser.add_argument("config", help="Config of the model")
    parser.add_argument("pkl_results", help="Results in pickle format")
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='Evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    assert args.eval or args.format_only, (
        "Please specify at least one operation (eval/format the results) with "
        'the argument "--eval", "--format-only"'
    )
    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl_results)
    assigner = MaxIoUAssigner(0.5, 0.5, 0.5)
    results = []
    for idx in tqdm(range(len(dataset))):
        # img_info = dataset.data_infos[idx]
        ann_info = dataset.get_ann_info(idx)
        gt_bboxes, gt_labels = torch.from_numpy(ann_info["bboxes"]), torch.from_numpy(ann_info["labels"])

        det_bboxes = []
        det_bboxes = torch.tensor(outputs[idx])

        assign_result = assigner.assign(det_bboxes[:, :4], gt_bboxes, None, gt_labels)
        valid = assign_result.labels >= 0
        det_labels = assign_result.labels[valid]
        det_bboxes = det_bboxes[valid]

        indices = batched_nms(det_bboxes[:, :4], det_bboxes[:, 4], idxs=det_labels, iou_threshold=0.5)
        det_bboxes = det_bboxes[indices]
        det_labels = det_labels[indices]

        bbox_result = bbox2result(det_bboxes, det_labels, 65)
        results.append(bbox_result)

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(results, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get("evaluation", {}).copy()
        # hard-code way to remove EvalHook args
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
            eval_kwargs.pop(key, None)
        eval_kwargs.update(dict(metric=args.eval, **kwargs))
        print(dataset.evaluate(results, **eval_kwargs))


if __name__ == "__main__":
    main()
