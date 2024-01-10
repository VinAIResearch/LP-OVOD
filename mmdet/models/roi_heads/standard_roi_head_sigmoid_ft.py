import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head_sigmoid import StandardRoIHeadSigmoid
from .class_name import *
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from .class_name import coco_base_label_ids, coco_novel_label_ids, COCO_OVD_ALL_CLS
from fvcore.nn import sigmoid_focal_loss_jit
import math


@HEADS.register_module()
class StandardRoIHeadSigmoidFinetune(StandardRoIHeadSigmoid):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 test_temp=0.07,
                 neg_pos_ub=20,
                 **kwargs):
        super(StandardRoIHeadSigmoidFinetune, self).__init__(**kwargs)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self.text_features_for_classes = self.text_features_for_classes.float()
        self.mapping_label = {label: i for i, label in enumerate(coco_novel_label_ids)}
        self.mapping_label.update({65: len(coco_novel_label_ids)})
        self.prototype_novel = nn.Parameter(torch.zeros((len(coco_novel_label_ids), 1024)), requires_grad=True)
        self.test_temp = test_temp
        if self.bbox_sampler is not None:
            self.bbox_sampler.neg_pos_ub = neg_pos_ub

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.bbox_head.eval()
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        """Run forward function and calculate loss for box head in training."""

        # -------------Classification loss---------------
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results, region_embeddings = self._bbox_forward(x, rois)

        cls_score_text = region_embeddings @ self.prototype_novel.T
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        labels, _, _, _ = bbox_targets
        labels = labels.cpu().numpy()
        labels = torch.tensor([self.mapping_label[label] for label in labels]).long().to(self.device)

        bin_labels = labels.new_full((labels.size(0), len(coco_novel_label_ids)), 0)
        inds = torch.nonzero(
            (labels >= 0) & (labels < len(coco_novel_label_ids)), as_tuple=False).squeeze()
        if inds.numel() > 0:
            bin_labels[inds, labels[inds]] = 1

        num_pos_bboxes = sum([res.pos_bboxes.size(0) for res in sampling_results])
        cls_loss = sigmoid_focal_loss_jit(
            cls_score_text, bin_labels,
            reduction="sum",
            gamma=2, alpha=0.25) / num_pos_bboxes
        loss_bbox = dict()
        
        loss_bbox.update(cls_loss=cls_loss)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           **kwargs):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        # Get origin input shape to support onnx dynamic input shape
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        rois = bbox2roi(proposals)
        num_proposals_per_img = tuple(len(proposal) for proposal in proposals)
        objectness = kwargs.get('objectness', None)

        # Score for the first head
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        cls_score_text_base = (region_embeddings @ self.prototype.T).float()
        cls_score_text_novel = (region_embeddings @ self.prototype_novel.T) / self.test_temp
        cls_score_text_novel = cls_score_text_novel.float()
        cls_score_text_full = cls_score_text_base.new_full((region_embeddings.size(0), self.num_classes), 0)
        cls_score_text_full[:, coco_base_label_ids] = cls_score_text_base
        cls_score_text_full[:, coco_novel_label_ids] = cls_score_text_novel
        cls_score_text_full = cls_score_text_full.sigmoid()
        # Score for the second head
        if self.ensemble:
            _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
            cls_score_image = region_embeddings_image @ self.text_features_for_classes.T.float()
            cls_score_image = (cls_score_image / self.temperature_test).float()
            cls_score_image = F.softmax(cls_score_image, dim=1)
            
        # Ensemble two heads (default setting)
        if self.ensemble:
            cls_score = torch.where(
                self.novel_index, cls_score_image**(1-self.beta) * cls_score_text_full**self.beta,
                cls_score_text_full**(1-self.alpha) * cls_score_image**self.alpha)
        else:
            cls_score = cls_score_text_full

        if objectness is not None:
            cls_score = (cls_score * objectness.unsqueeze(1)) ** 0.5

        # add score for background class (compatible with mmdet nms)
        cls_score = torch.cat([cls_score, torch.zeros(cls_score.size(0), 1, device=self.device)], dim=1)

        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None,) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        return det_bboxes, det_labels
