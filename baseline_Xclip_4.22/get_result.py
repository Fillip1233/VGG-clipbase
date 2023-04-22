import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import distributed as dist
import numpy as np
from result_class import Result

def get_triplets(pred):
    result_list = []
    for img_id in range(pred['pred_logits'].shape[0]):
        # need to change sub obj
        s_cls_score = pred['pred_logits'][img_id,...]
        o_cls_score = pred['pred_logits'][img_id,...]
        att_score = pred['attention_distribution'][img_id,...]
        spa_score = pred['spatial_distribution'][img_id,...]
        con_score = pred['contacting_distribution'][img_id,...]
        s_bbox_pred = pred['pred_boxes'][img_id,...]
        o_bbox_pred = pred['pred_boxes'][img_id,...]
        triplets = get_triplets_single(s_cls_score,o_cls_score,
                                       att_score,spa_score,con_score,
                                       s_bbox_pred,o_bbox_pred)
        result_list.append(triplets)
    return result_list

def get_triplets_single(s_cls_score,o_cls_score,
                                    att_score,spa_score,con_score,
                                    s_bbox_pred,o_bbox_pred):
    assert len(s_cls_score) == len(o_cls_score)
    assert len(s_cls_score) == len(s_bbox_pred)
    assert len(s_cls_score) == len(o_bbox_pred)
    s_logits = F.softmax(s_cls_score, dim=-1)
    o_logits = F.softmax(o_cls_score, dim=-1)
    att_log = F.softmax(att_score,dim=-1)
    spa_log = F.softmax(spa_score,dim=-1)
    con_log = F.softmax(con_score,dim=-1)
    att_logits = att_log[...,1:]
    s_scores, s_labels = s_logits.max(-1)
    o_scores, o_labels = o_logits.max(-1)
    att_score,att_indexes = att_logits.reshape(-1).topk(100)
    att_labels = att_indexes % 4
    triplets_index = att_indexes// 3
    spa_logits = spa_log[...,1:]
    _,spa_indexes = spa_logits.reshape(-1).topk(100)
    spa_labels =spa_indexes % 7
    con_logits = con_log[...,1:]
    _,con_indexes = con_logits.reshape(-1).topk(100)
    con_labels =con_indexes % 18
    s_scores = s_scores[triplets_index]
    s_labels = s_labels[triplets_index]+1
    s_bbox_pred = s_bbox_pred[triplets_index]

    o_scores = o_scores[triplets_index]
    o_labels = o_labels[triplets_index]+1
    o_bbox_pred = o_bbox_pred[triplets_index]

    att_dist = att_log.reshape(-1,4)[triplets_index]
    spa_dist = spa_log.reshape(-1,7)[triplets_index]
    con_dist = con_log.reshape(-1,18)[triplets_index]

    labels = torch.cat((s_labels,o_labels),0)
    rel_pairs = torch.arange(len(labels),dtype=torch.int).reshape(2,-1).T
    det_bboxes = torch.cat((s_bbox_pred,o_bbox_pred),0)

    return det_bboxes,labels,rel_pairs,att_labels,spa_labels,con_labels,att_dist,spa_dist,con_dist

def triplet2Result(triplets):
    bboxes,labels,rel_pairs,att_labels,spa_labels,con_labels,att_dist,spa_dist,con_dist = triplets
    labels = labels.detach().cpu().numpy()
    bboxes = bboxes.detach().cpu().numpy()
    rel_pairs = rel_pairs.detach().cpu().numpy()
    att_labels = att_labels.detach().cpu().numpy()
    spa_labels = spa_labels.detach().cpu().numpy()
    con_labels = con_labels.detach().cpu().numpy()
    att_dists = att_dist.detach().cpu().numpy()
    spa_dists = spa_dist.detach().cpu().numpy()
    con_dists = con_dist.detach().cpu().numpy()
    return Result(
        refine_bboxes=bboxes,
        labels=labels,
        formatted_masks=dict(pan_results=None),
        rel_pair_idxes=rel_pairs,
        att_dists=att_dists,
        spa_dists=spa_dists,
        con_dists=con_dists,
        # rel_labels=r_labels,
        att_labels = att_labels,
        spa_labels = spa_labels,
        con_labels = con_labels,
        pan_results=None,
    )
def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def evaluate(AG_dataset,
        gt_annotation,
        sg_result,metric='sgdet',
        logger=None,
        jsonfile_prefix=None,
        classwise=True,
        multiple_preds=False,
        iou_thrs=0.5,
        nogc_thres_num=None,
        detection_method='bbox'):
    print('\nLoading testing groundtruth...\n')
    gt_result = []
    for i in range(len(AG_dataset.gt_annotations)):
        ann = get_ann_info(AG_dataset,gt_annotation,i)


def get_ann_info(AG_dataset,gt_annotation,idx):
    d = AG_dataset.gt_annotations[idx]
    # Process bbox annotations
    gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
    gt_bboxes = np.array()