from ..builder import DETECTORS
from .faster_rcnn import FasterRCNN


@DETECTORS.register_module()
class FasterRCNNFreezeBackbone(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, **kwargs):
        super(FasterRCNNFreezeBackbone, self).__init__(**kwargs)

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.neck.parameters():
            p.requires_grad = False
        for p in self.rpn_head.parameters():
            p.requires_grad = False

    def forward_train(
        self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs
    ):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.backbone.eval()
        self.neck.eval()
        self.rpn_head.eval()
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
            _, proposal_list = self.rpn_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg
            )
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **kwargs
        )
        losses.update(roi_losses)

        return losses
