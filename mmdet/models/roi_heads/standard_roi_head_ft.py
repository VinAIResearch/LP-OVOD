import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops.roi_align import roi_align
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .standard_roi_head import StandardRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from .class_name import *
from PIL import Image
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from .class_name import coco_base_label_ids, coco_novel_label_ids, COCO_OVD_ALL_CLS


@HEADS.register_module()
class StandardRoIHeadFinetune(StandardRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self, 
                 train_temp=0.1,
                 test_temp=0.07,
                 neg_pos_ub=20,
                 **kwargs):
        super(StandardRoIHeadFinetune, self).__init__(**kwargs)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            
        self.text_features_for_classes = self.text_features_for_classes.float()
        self.mapping_label = {label: i for i, label in enumerate(coco_novel_label_ids)}
        self.mapping_label.update({65: len(coco_novel_label_ids)})
        self.fc_cls = nn.Linear(512, len(coco_novel_label_ids) + 1, bias=True)
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        self.train_temp = train_temp
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
        self.eval()
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

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses
    
    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        
        # -------------Classification loss---------------
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        normalized_weight = F.normalize(self.fc_cls.weight, p=2, dim=-1)

        cls_score_text = region_embeddings @ normalized_weight.T
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        labels, _, _, _ = bbox_targets
        labels = labels.cpu().numpy()
        labels = torch.tensor([self.mapping_label[label] for label in labels]).long().to(self.device)
        
        cls_loss = F.cross_entropy(cls_score_text / self.train_temp, labels, reduction='mean')
        loss_bbox = dict()
        
        loss_bbox.update(cls_loss=cls_loss)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
    
    def classification_branch1(self, region_embeddings, text_embeddings, proposals):
        # origin_cls_text = F.softmax(torch.matmul(region_embeddings, text_embeddings.T)/self.temperature, dim=-1)
        non_base_ids = coco_novel_label_ids + [65]
        region_embeddings_norm = F.normalize(region_embeddings, p=2, dim=-1)
        cls_score_text = torch.matmul(region_embeddings_norm, text_embeddings.T).float()
        # cls_score_text[..., non_base_ids] = -1e11
        cls_score_text = F.softmax(cls_score_text / self.temperature_test, dim=-1) #[N, 66]

        novel_score_text = torch.softmax(self.fc_cls(region_embeddings), dim=-1)
        cls_score_text[..., non_base_ids] = novel_score_text
        return cls_score_text
    
    def classification_branch1_ver2(self, region_embeddings, text_embeddings, proposals):
        # origin_cls_text = F.softmax(torch.matmul(region_embeddings, text_embeddings.T)/self.temperature, dim=-1)

        region_embeddings_norm = F.normalize(region_embeddings, p=2, dim=-1)
        cls_score_text = torch.matmul(region_embeddings_norm, text_embeddings.T).float()
        # cls_score_text[..., non_base_ids] = -1e11
        cls_score_text = F.softmax(cls_score_text / self.temperature_test, dim=-1) #[N, 66]

        novel_weights = F.normalize(self.fc_cls.weight, dim=-1, p=2)
        novel_score_text = F.softmax((region_embeddings_norm @ novel_weights.T) / self.test_temp , dim=-1)
        cls_score_text[..., coco_novel_label_ids] = novel_score_text[...,:-1]
        return cls_score_text

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

        # Text embedding
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
            text_features = torch.cat([self.text_features_for_classes, bg_class_embedding], dim=0)
        else:
            text_features = self.text_features_for_classes
        # Score for the first head
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        region_embeddings = self.projection(region_embeddings)
        cls_score_text = self.classification_branch1_ver2(region_embeddings, text_features, proposals)
        # Score for the second head
        if self.ensemble:
            _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
            cls_score_image = region_embeddings_image @ text_features.T
            cls_score_image = (cls_score_image / self.temperature_test).float()
            cls_score_image[:, -1] = -1e11 # ignore score for background class
            
            cls_score_image = F.softmax(cls_score_image, dim=1)
        if self.ensemble:
            cls_score = torch.where(
                self.novel_index, cls_score_image**(1-self.beta) * cls_score_text**self.beta,
                cls_score_text**(1-self.alpha) * cls_score_image**self.alpha)
            
        else:
            cls_score = cls_score_text

        objectness = kwargs.get('objectness', None)
        if objectness is not None:
            cls_score = torch.sqrt(cls_score * objectness.unsqueeze(1))
        # cls_score = cls_score_text
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
       
    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale, **kwargs)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], len(self.CLASSES))
            for i in range(len(det_bboxes))
        ]

        return bbox_results
