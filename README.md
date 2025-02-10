# Implementation of a Weighting Loss Approach for Transformer-Based Object Detection

This repository contains updates to the loss criterion used in transformer-based object detection models. The updates integrate a size-balanced L1 loss to improve performance.

### Original Criterion

The `old_criterion.py` file is the original code used by the following repositories:
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- [Detrex](https://github.com/IDEA-Research/detrex)

### Updated Criterion

The `criterion.py` file includes the integration of size-balanced L1 loss for improved bounding box regression.

#### File Paths:
- **Detrex:** `detrex/modeling/criterion/criterion.py`
- **RT-DETR:** `rtdetr_pytorch/src/zoo/rtdetr/rtdetr_criterion.py`

### Changes in Loss Calculation

**Original Implementation:**
```python
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes: the L1 regression loss and the GIoU loss.
        
        The targets dict must contain the key "boxes" with a tensor of shape [nb_target_boxes, 4].
        The target boxes are expected in the format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
```

**Updated Implementation with Size-Balanced L1 Loss:**
```python
    def calculate_areas(self, bboxes):
        """
        Calculate the areas of bounding boxes given in the format [center_x, center_y, width, height].

        :param bboxes: A PyTorch tensor of shape (N, 4) where each row is [center_x, center_y, width, height].
        :return: A tensor containing the area of each bounding box.
        """
        widths = bboxes[:, 2]
        heights = bboxes[:, 3]
        areas = widths * heights
        return areas

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses using a bounding box size-weighting mechanism for the L1 loss.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]

        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        areas = self.calculate_areas(src_boxes)
        areas = areas.clone().detach().requires_grad_(False)

        if len(areas) > 0:
            area_max = torch.max(areas)
            area_contribution = F.softmax(-areas / area_max, dim=0)
            box_weighting = area_contribution.unsqueeze(1).repeat(1, 4)
            losses['loss_bbox'] = (loss_bbox * box_weighting.to('cuda')).sum()
        else:
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
```

