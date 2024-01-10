import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from .class_name import *
from PIL import Image
from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from tqdm import tqdm


@HEADS.register_module()
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 kd_weight=256,
                 prompt_path="",
                 fix_bg=False,
                 feature_path='data/lvis_clip_image_embedding',
                 clip_root="",
                 temperature=0.02,
                 temperature_test=0.02,
                 alpha=0.,
                 beta=0.4,
                 pretrained=None,
                 ):
        super(StandardRoIHead, self).__init__(bbox_roi_extractor=bbox_roi_extractor,
                                              bbox_head=bbox_head,
                                              mask_roi_extractor=mask_roi_extractor,
                                              mask_head=mask_head,
                                              shared_head=shared_head,
                                              train_cfg=train_cfg,
                                              test_cfg=test_cfg,
                                              pretrained=pretrained,
                                              )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        if bbox_head.num_classes == 48:
            self.CLASSES = COCO_OVD_ALL_CLS
        elif bbox_head.num_classes == 65:
            self.CLASSES = COCO_OVD_ALL_CLS
        elif bbox_head.num_classes == 80:
            self.CLASSES = COCO_CLASSES
        elif bbox_head.num_classes == 20:
            self.CLASSES = VOC_CLASSES
        elif bbox_head.num_classes == 1203:
            self.CLASSES = LVIS_CLASSES
        elif bbox_head.num_classes == 365:
            self.CLASSES = Object365_CLASSES
        else:
            raise ValueError(f"{bbox_head.num_classes} is an invalid bbox_head num_classes!")
            
        # NOTE: num_classes (65) != bbox_head.num_classes (48)
        # num_classes for evaluation, bbox_head.num_classes for training
        self.num_classes = len(self.CLASSES)
        
        if self.num_classes == 1203:
            self.base_label_ids = torch.tensor(lvis_base_label_ids, device=device)
            self.novel_label_ids = torch.tensor(lvis_novel_label_ids, device=device)
            self.novel_index = F.pad(torch.bincount(self.novel_label_ids),(0,self.num_classes-self.novel_label_ids.max())).bool()
        elif self.num_classes == 20:
            self.novel_label_ids = torch.tensor(voc_novel_label_ids, device=device)
            self.novel_index = F.pad(torch.bincount(self.novel_label_ids),(0,self.num_classes-self.novel_label_ids.max())).bool()
        elif self.num_classes == 80:
            self.novel_label_ids = torch.tensor(coco_unseen_ids_train, device=device)
            self.unseen_label_ids_test = torch.tensor(coco_unseen_ids_test, device=device)
            self.novel_index = F.pad(torch.bincount(self.novel_label_ids),(0,self.num_classes-self.novel_label_ids.max())).bool()
        elif self.num_classes == 65:
            base_label_ids = [COCO_OVD_ALL_CLS.index(i) for i in COCO_SEEN_CLS]
            self.base_label_ids = torch.tensor(base_label_ids, device=device)
            novel_label_ids = [COCO_OVD_ALL_CLS.index(i) for i in COCO_UNSEEN_CLS]
            self.novel_label_ids = torch.tensor(novel_label_ids, device=device)
            self.novel_index = F.pad(torch.bincount(self.novel_label_ids),(0,self.num_classes-self.novel_label_ids.max())).bool()
            
        # config
        self.kd_weight = kd_weight
        self.fix_bg = fix_bg
        
        # Load text embedding
        self.feature_path = feature_path
        self.text_features_for_classes = []
        self.iters = 0
        self.ensemble = bbox_head.ensemble
        if prompt_path is not None:
            save_path = prompt_path
        else:
            save_path = 'text_embedding.pt'
        
        if osp.exists(save_path):
            if not self.fix_bg:
                self.text_features_for_classes = torch.load(save_path).to(device).squeeze()[:self.num_classes]
            else:
                self.text_features_for_classes = torch.load(save_path).to(device).squeeze()
            print(f"Prompt path exist, load from {save_path} with shape {self.text_features_for_classes.shape}")
        else:
            # Load clip
            clip_model, self.preprocess = clip.load('ViT-B/32', device, download_root=clip_root)
            clip_model.eval()
            for child in clip_model.children():
                for param in child.parameters():
                    param.requires_grad = False
            time_start = time.time()
            for template in tqdm(template_list):
                text_features_for_classes = torch.cat([clip_model.encode_text(clip.tokenize(template.format(c)).to(device)).detach() for c in self.CLASSES])
                self.text_features_for_classes.append(F.normalize(text_features_for_classes,dim=-1))

            self.text_features_for_classes = torch.stack(self.text_features_for_classes).mean(dim=0)
            torch.save(self.text_features_for_classes.detach().cpu(), save_path)
            print('Text embedding finished, {} passed, shape {}'.format(time.time()-time_start, self.text_features_for_classes.shape))
        
        # if not fix_bg, use a learnable back_ground
        if not self.fix_bg:
            self.bg_embedding = nn.Linear(1,512)
            nn.init.xavier_uniform_(self.bg_embedding.weight)
            nn.init.constant_(self.bg_embedding.bias, 0)
            
        # projection layer to map faster-rcnn output dim to clip dim (1024 to 512)
        self.projection = nn.Linear(1024, 512)
        self.temperature = temperature
        self.temperature_test = temperature_test
        self.alpha = alpha
        self.beta = beta
        
        # if ensemble, use a different head to output score for ensembling
        if self.ensemble:
            self.projection_for_image = nn.Linear(1024, 512)
            nn.init.xavier_uniform_(self.projection_for_image.weight)
            nn.init.constant_(self.projection_for_image.bias, 0)

        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      proposals_pre_computed,
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
            bbox_results = self._bbox_forward_train(x, sampling_results,proposals_pre_computed,
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

    def _bbox_forward(self, x, rois):
        """
        First head
        Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.forward_embedding(bbox_feats)
        bbox_pred = self.bbox_head(region_embeddings)
        bbox_results = dict(
            bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results, region_embeddings

    def _bbox_forward_for_image(self, x, rois):
        """
        Second head
        Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        rois = rois.float()
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        region_embeddings = self.bbox_head.forward_embedding_for_image(bbox_feats)

        return None, region_embeddings

    def _bbox_forward_train(self, x, sampling_results, proposals_pre_computed, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        
        # -------------Classification loss---------------
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings, p=2, dim=1)
        
        # get class embedding
        # with background (default)
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).reshape(1, 512)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
            text_features = torch.cat([self.text_features_for_classes[self.base_label_ids], bg_class_embedding], dim=0)
        # without background
        else:
            text_features = self.text_features_for_classes[self.base_label_ids]

        cls_score_text = region_embeddings @ text_features.T
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        labels, _, _, _ = bbox_targets
        
        text_cls_loss = F.cross_entropy(cls_score_text / self.temperature, labels, reduction='mean')
        loss_bbox = self.bbox_head.loss(
            bbox_results['bbox_pred'], rois,
            *bbox_targets)
        
        loss_bbox.update(text_cls_loss=text_cls_loss)
        
        # ---------------Knowledge distillation------------------------
        if self.kd_weight > 0: 
            num_proposals_per_img = tuple(len(proposal) for proposal in proposals_pre_computed)
            rois_image = torch.cat(proposals_pre_computed, dim=0)
            batch_index = torch.cat([x[0].new_full((num_proposals_per_img[i],1),i) for i in range(len(num_proposals_per_img))],0)
            bboxes = torch.cat([batch_index, rois_image[..., :4]], dim=-1)
            
            if self.ensemble:
                _, region_embeddings_image = self._bbox_forward_for_image(x, bboxes)
                region_embeddings_image = self.projection_for_image(region_embeddings_image)
                region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
            else:
                _, region_embeddings_image = self._bbox_forward(x, bboxes)
                region_embeddings_image = self.projection(region_embeddings_image)
                region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)

            clip_image_features_ensemble = []
            for i in range(len(img_metas)):
                save_path = os.path.join(self.feature_path, img_metas[i]['ori_filename'].split('.')[0] + '.pth')
                # TODO: load image region embedding with data
                # potential bottleneck
                try:
                    tmp = torch.load(save_path)
                    clip_image_features_ensemble.append(tmp.to(self.device))
                except:
                    raise ValueError(f"{save_path} does not exist")
            clip_image_features_ensemble = torch.cat(clip_image_features_ensemble, dim=0)
            kd_loss = F.l1_loss(region_embeddings_image, clip_image_features_ensemble) * self.kd_weight
            loss_bbox.update(kd_loss=kd_loss)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.bool))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.bool))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results


    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results
    
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
        # img_norm_mean, img_norm_std = img_metas[0]['img_norm_cfg']['mean'], img_metas[0]['img_norm_cfg']['std']
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
        rois = bbox2roi(proposals)
        num_proposals_per_img = tuple(len(proposal) for proposal in proposals)
        objectness = kwargs.get('objectness', None)
        
        # Text embedding
        if not self.fix_bg:
            input_one = x[0].new_ones(1)
            bg_class_embedding = self.bg_embedding(input_one).unsqueeze(0)
            bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
            # bg_class_embedding = torch.zeros((1, self.text_features_for_classes.size(1)), device=self.text_features_for_classes.device)
            text_features = torch.cat([self.text_features_for_classes, bg_class_embedding], dim=0)
        else:
            text_features = self.text_features_for_classes
        
        # Score for the first head
        bbox_results, region_embeddings = self._bbox_forward(x, rois)
        region_embeddings = self.projection(region_embeddings)
        region_embeddings = torch.nn.functional.normalize(region_embeddings,p=2,dim=1)
        cls_score_text = region_embeddings @ text_features.T
        
        cls_score_text = cls_score_text / self.temperature_test
        cls_score_text = F.softmax(cls_score_text, dim=1)
        
        # Score for the second head
        if self.ensemble:
            _, region_embeddings_image = self._bbox_forward_for_image(x, rois)
            region_embeddings_image = self.projection_for_image(region_embeddings_image)
            region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
            cls_score_image = region_embeddings_image @ text_features.T
            cls_score_image = (cls_score_image / self.temperature_test).float()
            cls_score_image[:, -1] = -1e11 # ignore score for background class
            
            cls_score_image = F.softmax(cls_score_image, dim=1)
            
        # Ensemble two heads (default setting)
        if self.ensemble:
            cls_score = torch.where(
                self.novel_index, cls_score_image**(1-self.beta) * cls_score_text**self.beta,
                cls_score_text**(1-self.alpha) * cls_score_image**self.alpha)
        else:
            cls_score = cls_score_text
            
        if objectness is not None:
            cls_score = (cls_score * objectness.unsqueeze(1)) ** 0.5
        
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
                    proposals=None,
                    rescale=False,
                    **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale, **kwargs)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale, **kwargs)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))
