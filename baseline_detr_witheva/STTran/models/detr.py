# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
import numpy as np
from getgt import AverageMeter
import time
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        ##class remove
        ##self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        ## change to obj_cls_embed , object class is 35, if calculate with person ,the class is 36, AG ->37
        self.obj_cls_embed = nn.Linear(hidden_dim, 38)
        ##bbox remove
        ##self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        ##change to obj_box_embed
        self.obj_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        ##so as to sub
        self.sub_cls_embed = nn.Linear(hidden_dim, 1)
        self.sub_box_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        ##three kinds of relation
        self.a_rel_compress = nn.Linear(hidden_dim, 4) ##input tensor dimesion is 512
        self.s_rel_compress = nn.Linear(hidden_dim, 7)
        self.c_rel_compress = nn.Linear(hidden_dim, 18) ##17?

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        ## later build
        self.obj_loss_cls = None
        self.obj_loss_bbox = None
        self.obj_loss_iou = None
        self.sub_loss_cls = None
        self.sub_loss_bbox = None
        self.sub_loss_iou = None

        self.rel_loss_cls = None

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        entry_time=time.time()
        backbone_time= AverageMeter()
        features, pos = self.backbone(samples)
        backbone_time.update(time.time()-entry_time)
        print("backbone_time:%.2f"%(backbone_time.val))
        ### 将feature和mask分开
        src, mask = features[-1].decompose()


        assert mask is not None
        # 
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # hs = self.transformer(src, mask, query_embed, pos_embed,bs,c,h,w)[0]

        outputs_class = self.obj_cls_embed(hs)
        outputs_coord = self.obj_box_embed(hs).sigmoid()

        # match dimesion is error ,I have change some dimesion
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

        ## add according to the task
        # out["pred_sub_bbox"]=self.sub_box_embed(hs)

        outputs_arls = self.a_rel_compress(hs)
        outputs_srls = self.s_rel_compress(hs).sigmoid()
        outputs_crls = self.c_rel_compress(hs).sigmoid()
        out["attention_distribution"] = outputs_arls[-1]
        out["spatial_distribution"] = outputs_srls[-1]
        out["contacting_distribution"] = outputs_crls[-1]

        # out["spatial_distribution"] = torch.sigmoid(out["spatial_distribution"])
        # out["contacting_distribution"] = torch.sigmoid(out["contacting_distribution"])

        #out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord,outputs_arls,outputs_srls,outputs_crls)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord,outputs_arls,outputs_srls,outputs_crls):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b ,'attention_distribution': c,'spatial_distribution':d,'contacting_distribution':e}
                for a, b,c,d,e in zip(outputs_class[:-1], outputs_coord[:-1],outputs_arls[:1],outputs_srls[:-1],outputs_crls[:-1])]
        # return [{'pred_logits': a, 'pred_boxes': b}
        #         for a, b in zip(outputs_class, outputs_coord)]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        ### rls empty_weight
        self.num_att = 3
        self.num_spa = 6
        self.num_con = 17
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight = torch.ones(self.num_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
        empty_weight_att = torch.ones(self.num_att + 1)
        # empty_weight_att = torch.ones(self.num_att)
        empty_weight_att[-1] = self.eos_coef
        self.register_buffer('empty_weight_att', empty_weight_att)
        
        empty_weight_spa = torch.ones(self.num_spa + 1)
        # empty_weight_spa = torch.ones(self.num_spa)
        empty_weight_spa[-1] = self.eos_coef
        self.register_buffer('empty_weight_spa', empty_weight_spa)
        
        empty_weight_con = torch.ones(self.num_con + 1)
        # empty_weight_con = torch.ones(self.num_con)
        empty_weight_con[-1] = self.eos_coef
        self.register_buffer('empty_weight_con', empty_weight_con)
    ### 计算rls的loss
    def loss_att_rls(self, outputs, targets, indices, num_boxes, nopeople_indx,log=True):
        """ATTention_rls 注意字符串要根据输出和gt改
        """
        assert 'attention_distribution' in outputs
        pred_att = outputs['attention_distribution']

        idx = self._get_src_permutation_idx(nopeople_indx)
        ### 因为nopeopleindex删掉了0，所以是从1开始的，j->j-1
        target_rls_o = torch.cat([t["attention_relationship"][J-1] for t, (_, J) in zip(targets, nopeople_indx)])
        target_rls = torch.full(pred_att.shape[:2], self.num_att,
                                    dtype=torch.int64, device=pred_att.device)
        target_rls[idx] = target_rls_o

        loss_ce = F.cross_entropy(pred_att.transpose(1, 2), target_rls, self.empty_weight_att)
        losses = {'loss_ce_att_rls': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['att_rls_error'] = 100 - accuracy(pred_att[idx], target_rls_o)[0]
        return losses
    def loss_con_rls(self, outputs, targets, indices,num_boxes, nopeople_indx, log=True):
        """Contact_rls 注意字符串要根据输出和gt改
        """
        assert 'contacting_distribution' in outputs
        pred_con = outputs['contacting_distribution']
        
        ###mlm
        idx = self._get_src_permutation_idx(nopeople_indx)
        pred_rls = pred_con[idx]
        label_rls = -torch.ones([len(pred_rls), 18], dtype=torch.long).to(device=pred_con.device)
        z = 0
        for m in range(len(targets)):
            for n in range(len(targets[m]["contacting_relation"])):
                label_rls[z, : len(targets[m]["contacting_relation"][n])] = targets[m]["contacting_relation"][n]
                z +=1
        
        mlm_loss = nn.MultiLabelMarginLoss()
        loss_mlm = mlm_loss(pred_rls, label_rls)
        losses = {'loss_ce_con_rls': loss_mlm}
        return losses
       
        ###用于log
        # target_rls_o = torch.cat([t["contacting_relation"][J-1] for t, (_, J) in zip(targets, nopeople_indx)])
        # target_rls_o=target_rls_o.squeeze()        

        # idx = self._get_src_permutation_idx(nopeople_indx)
        # target_rls_o = torch.cat([t["contacting_relation"][J] for t, (_, J) in zip(targets, nopeople_indx)])
        # target_rls = torch.full(pred_con.shape[:2], self.num_con,
        #                             dtype=torch.int64, device=pred_con.device)
        ###不知道为什么会多出来一维
        # target_rls_o=target_rls_o.squeeze()
        # target_rls[idx] = target_rls_o

        # loss_ce = F.cross_entropy(pred_con.transpose(1, 2), target_rls, self.empty_weight_con)
        # losses = {'loss_ce_con_rls': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['att_rls_error'] = 100 - accuracy(pred_con[idx], target_rls_o)[0]
        # return losses 
            pass   
    def loss_spa_rls(self, outputs, targets, indices,num_boxes, nopeople_indx,log=True):
        """Spatial_rls 注意字符串要根据输出和gt改
        """
        assert 'spatial_distribution' in outputs
        pred_spa = outputs['spatial_distribution']
        idx = self._get_src_permutation_idx(nopeople_indx)
        pred_rls = pred_spa[idx]
        label_rls = -torch.ones([len(pred_rls), 7], dtype=torch.long).to(device=pred_spa.device)
        z = 0
        for m in range(len(targets)):
            for n in range(len(targets[m]["spatial_relationship"])):
                label_rls[z, : len(targets[m]["spatial_relationship"][n])] = targets[m]["spatial_relationship"][n]
                z +=1
        mlm_loss = nn.MultiLabelMarginLoss()
        loss_mlm = mlm_loss(pred_rls, label_rls)
        losses = {'loss_ce_spa_rls': loss_mlm}
        return losses
       
       
        ###用于log
        # target_rls_o = torch.cat([t["spatial_relationship"][J-1] for t, (_, J) in zip(targets, nopeople_indx)])
        # target_rls_o=target_rls_o.squeeze()
        # target_rls[idx] = target_rls_o

        # loss_ce = F.cross_entropy(pred_spa.transpose(1, 2), target_rls, self.empty_weight_spa)
        # losses = {'loss_ce_spa_rls': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['att_rls_error'] = 100 - accuracy(pred_spa[idx], target_rls_o)[0]
            pass
        # return losses
    def loss_labels(self, outputs, targets, indices, num_boxes,nopeople_indx, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, nopeople_indx):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, nopeople_indx):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(box_ops.box_xyxy_to_cxcywh(target_boxes)))) ##
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    ### 加入rls
    def get_loss(self, loss, outputs, targets, indices, num_boxes, nopeople_indx,**kwargs):
        loss_map = {
            'att_rls':self.loss_att_rls,
            'spa_rls':self.loss_spa_rls,
            'con_rls':self.loss_con_rls,
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, nopeople_indx,**kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        b=[{i[1].tolist().index(0)}for i in indices]
        k=[list(indices[i])for i in range(len(indices))]
        nopeople_indx=[tuple([np.delete(m, list(b[i])[0])for m in j])for i,j in enumerate(k)]

        # Compute the average number of target boxes accross all nodes, for normalization purposes

        # num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = sum(len(t)-1 for t in targets[:4])
        
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes,nopeople_indx))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes,nopeople_indx, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    
                    # if loss == 'att_rls':
                    #     # Logging is enabled only for the last layer
                    #     kwargs = {'log': False}
                    # l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes,nopeople_indx, **kwargs)
                    # l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    # losses.update(l_dict)
                    # if loss == 'con_rls':
                    #     # Logging is enabled only for the last layer
                    #     kwargs = {'log': False}
                    # l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes,nopeople_indx, **kwargs)
                    # l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    # losses.update(l_dict)
                    # if loss == 'spa_rls':
                    #     # Logging is enabled only for the last layer
                    #     kwargs = {'log': False}
                    # l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes,nopeople_indx, **kwargs)
                    # l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    # losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    
    num_classes = 37 #0->background 1->person 2-36 ->class

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1,'loss_ce_att_rls':1,'loss_ce_spa_rls':1,'loss_ce_con_rls':1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['att_rls','spa_rls','con_rls','labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
