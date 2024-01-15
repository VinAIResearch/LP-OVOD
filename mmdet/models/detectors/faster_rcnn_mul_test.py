from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class FasterRCNNMulTest(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self, backbone, rpn_head, roi_head, train_cfg, test_cfg, neck=None, pretrained=None):
        super(FasterRCNNMulTest, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """Test without augmentation."""

        assert self.with_bbox, "Bbox head must be implemented."
        x = self.extract_feat(img)
        assert (
            "objectness" in kwargs and "proposals" in kwargs
        ), "Both proposals and objectness score need to be loaded"
        proposals = kwargs.get("proposals", None)
        objectness = kwargs.get("objectness", None)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        refined_bboxes = self.roi_head.get_refined_bboxes(x, proposal_list, img_metas)

        return self.roi_head.multi_test(
            x, img, proposal_list, refined_bboxes, img_metas, rescale=rescale, objectness=objectness
        )
