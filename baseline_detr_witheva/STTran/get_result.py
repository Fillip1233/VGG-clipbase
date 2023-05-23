import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import distributed as dist
import numpy as np
from result_class import Result

from functools import reduce
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from coco_eval import CocoEvaluator
import time
from getgt import AverageMeter
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class BasicSceneGraphEvaluator:
    def __init__(self, mode, multiple_preds=False):
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {20: [], 50: [], 100: []}
        self.multiple_preds = multiple_preds

    @classmethod
    def all_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, **kwargs) for m in ('sgdet', 'sgcls', 'predcls')}
        return evaluators

    @classmethod
    def vrd_modes(cls, **kwargs):
        evaluators = {m: cls(mode=m, multiple_preds=True, **kwargs) for m in ('preddet', 'phrdet')}
        return evaluators

    def evaluate_scene_graph_entry(self, gt_entry, pred_scores, viz_dict=None, iou_thresh=0.5):
        res = evaluate_from_dict(gt_entry, pred_scores, self.mode, self.result_dict,
                                  viz_dict=viz_dict, iou_thresh=iou_thresh, multiple_preds=self.multiple_preds)
        # self.print_stats()
        return res

    def save(self, fn):
        np.save(fn, self.result_dict)

    def print_stats(self):
        output = {}
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))
            output['R@%i' % k] = np.mean(v)
        return output

def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, multiple_preds=False,
                       viz_dict=None, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param mode: 'det' or 'cls'
    :param result_dict: 
    :param viz_dict: 
    :param kwargs: 
    :return: 
    """
    gt_rels = gt_entry['gt_relations']


    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']

    rel_scores = pred_entry['rel_scores']

    pred_rels = 1+rel_scores.argmax(1)
    predicate_scores = rel_scores.max(1)

    sub_boxes = pred_entry['sub_boxes']
    obj_boxes = pred_entry['obj_boxes']
    sub_score = pred_entry['sub_scores']
    obj_score = pred_entry['obj_scores']
    sub_class = pred_entry['sub_classes']
    obj_class = pred_entry['obj_classes']

    pred_to_gt, _, rel_scores = evaluate_recall(
                gt_rels, gt_boxes, gt_classes,
                pred_rels, sub_boxes, obj_boxes, sub_score, obj_score, predicate_scores, sub_class, obj_class, phrdet= mode=='phrdet',
                **kwargs)

    for k in result_dict[mode + '_recall']:

        match = reduce(np.union1d, pred_to_gt[:k])

        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)
    return pred_to_gt, _, rel_scores

def evaluate_recall(gt_rels, gt_boxes, gt_classes,
                    pred_rels, sub_boxes, obj_boxes, sub_score, obj_score, predicate_scores, sub_class, obj_class,
                    iou_thresh=0.5, phrdet=False):
    """
    Evaluates the recall
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0,5)), np.zeros(0)

    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)


    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])

    pred_triplets = np.column_stack((sub_class, pred_rels, obj_class))
    pred_triplet_boxes =  np.column_stack((sub_boxes, obj_boxes))
    relation_scores = np.column_stack((sub_score, obj_score, predicate_scores))  #TODO!!!! do not * 0.1 finally


    sorted_scores = relation_scores.prod(1)
    pred_triplets = pred_triplets[sorted_scores.argsort()[::-1],:]
    pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1],:]
    relation_scores = relation_scores[sorted_scores.argsort()[::-1],:]
    scores_overall = relation_scores.prod(1)


    if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
        print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
        # raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )

    return pred_to_gt, None, relation_scores

def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thresh, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets: 
    :param pred_triplets: 
    :param gt_boxes: 
    :param pred_boxes: 
    :param iou_thresh: 
    :return: 
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt

def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res

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


def _triplet(predicates, relations, classes, boxes,
             predicate_scores=None, class_scores=None):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])

    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))

    return triplets, triplet_boxes, triplet_scores

def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list):
    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        gt_entry = {'gt_classes': target['labels'].cpu().clone().numpy(),
                    'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
                    'gt_boxes': target_bboxes_scaled}

        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy()

        pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
        pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)
        rel_scores = outputs['rel_logits'][batch][:,1:-1].softmax(-1)

        pred_entry = {'sub_boxes': sub_bboxes_scaled,
                      'sub_classes': pred_sub_classes.cpu().clone().numpy(),
                      'sub_scores': pred_sub_scores.cpu().clone().numpy(),
                      'obj_boxes': obj_bboxes_scaled,
                      'obj_classes': pred_obj_classes.cpu().clone().numpy(),
                      'obj_scores': pred_obj_scores.cpu().clone().numpy(),
                      'rel_scores': rel_scores.cpu().clone().numpy()}

        evaluator['sgdet'].evaluate_scene_graph_entry(gt_entry, pred_entry)

        if evaluator_list is not None:
            for pred_id, _, evaluator_rel in evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
                gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
                if gt_entry_rel['gt_relations'].shape[0] == 0:
                    continue
                evaluator_rel['sgdet'].evaluate_scene_graph_entry(gt_entry_rel, pred_entry)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# ###
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

# def evaluate(AG_dataset,
#         gt_annotation,
#         sg_result,metric='sgdet',
#         logger=None,
#         jsonfile_prefix=None,
#         classwise=True,
#         multiple_preds=False,
#         iou_thrs=0.5,
#         nogc_thres_num=None,
#         detection_method='bbox'):
#     print('\nLoading testing groundtruth...\n')
#     gt_result = []
#     for i in range(len(AG_dataset.gt_annotations)):
#         ann = get_ann_info(AG_dataset,gt_annotation,i)


def get_ann_info(AG_dataset,gt_annotation,idx):
    d = AG_dataset.gt_annotations[idx]
    # Process bbox annotations
    gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
    gt_bboxes = np.array()




def evaluate(model, criterion, postprocessors, data_loader, base_ds,device, args,AG_dataset_test,test_iter):
    model.eval()
    criterion.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
    evaluator_list = []
    for index, name in enumerate(data_loader.dataset.rel_categories):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    for b in tqdm(range(len(data_loader))):
        # measure data loading time
        data = next(test_iter)
        im_data = data[0].cuda()
        print(im_data.shape)
        gt_annotation = AG_dataset_test.gt_annotations[data[1]]
        torch.cuda.synchronize()
        ##convinice to read
        gt_annotation=gt_annotation[:4]
        ##add no_relation to person
        # for i,j in enumerate(gt_annotation):
        #     for m in j:
        #         if 'person_bbox' in m.keys():
        #             m["attention_relationship"]=torch.tensor([3])
        #             m["spatial_relationship"]=torch.tensor([6])
        #             m["contacting_relationship"]=torch.tensor([17])

        ##change annotation to detr format
        dictionaries=[]
        gt_protime=AverageMeter()
        torch.cuda.synchronize()
        pro1=time.time()
        for i,j in enumerate(gt_annotation):
            t_bbox=np.array([[0,0,0,0]])
            a_relation= None
            s_relation= []
            c_relation= []
            for m in j:#j-> 1 frame , m->1 class dist
                if 'person_bbox' in m.keys():
                    m['class'] = 1  ##35 object class ,1 stand for human
                    m['person_bbox'][:1]/=im_data.shape[3]
                    m['person_bbox'][1:2]/=im_data.shape[2]
                    m['person_bbox'][2:3]/=im_data.shape[3]
                    m['person_bbox'][3:4]/=im_data.shape[2]
                    # m["attention_relationship"]=torch.tensor([3])
                    # m["spatial_relationship"]=torch.tensor([6])
                    # m["contacting_relationship"]=torch.tensor([17])
                    t_bbox = np.concatenate((t_bbox, m['person_bbox']), axis=0)
                if 'bbox' in m.keys():
                    m['bbox'][:1]/=im_data.shape[3]
                    m['bbox'][1:2]/=im_data.shape[2]
                    m['bbox'][2:3]/=im_data.shape[3]
                    m['bbox'][3:4]/=im_data.shape[2]
                    t_bbox = np.concatenate((t_bbox, [m['bbox']]), axis=0)
                if a_relation is None and 'attention_relationship'in m.keys():
                    a_relation=m["attention_relationship"]
                elif 'attention_relationship' in m.keys():
                    a_relation=torch.cat((a_relation,m["attention_relationship"]))
                if 'spatial_relationship' in m.keys():
                    s_relation.append(m["spatial_relationship"])
                if 'contacting_relationship' in m.keys():
                    c_relation.append(m["contacting_relationship"])
            # s_relation = torch.stack(s_relation)
            s_relation=pad_sequence(s_relation, batch_first=True)
            # c_relation = torch.stack(c_relation)
            c_relation=pad_sequence(c_relation, batch_first=True)
            class_value = [d["class"] for d in j]
            class_value = torch.tensor(class_value)
            t_bbox = t_bbox[1:]
            t_bbox =torch.from_numpy(t_bbox)
            t_bbox=t_bbox.to(dtype=torch.float32)
            ##contacting_relationship->contacting_relation
            dictionaries.append({"labels":class_value,"boxes":t_bbox,"attention_relationship":a_relation,"spatial_relationship":s_relation,"contacting_relation":c_relation})
        gt_annotation=dictionaries
        gt_annotation = [{k: v.cuda() for k, v in t.items()} for t in gt_annotation]
        torch.cuda.synchronize()
        gt_protime.update(time.time()-pro1)
        print("gt_protime:%.2f"%(gt_protime.val))
        pred = model(im_data)
        evaluate_rel_batch(pred, gt_annotation, evaluator, evaluator_list)
        orig_target_sizes = torch.stack([t["orig_size"] for t in gt_annotation], dim=0)
        results = postprocessors['bbox'](pred, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(gt_annotation, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
    evaluator['sgdet'].print_stats()
            
