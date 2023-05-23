import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Result(object):
    """ little container class for holding the detection result
        od: object detector, rm: rel model"""
    def __init__(
        self,
        bboxes=None,  # gt bboxes / OD: det bboxes
        dists=None,  # OD: predicted dists
        labels=None,  # gt labels / OD: det labels
        masks=None,  # gt masks  / OD: predicted masks
        formatted_masks=None,  # OD: Transform the masks for object detection evaluation
        points=None,  # gt points / OD: predicted points
        rels=None,  # gt rel triplets / OD: sampled triplets (training) with target rel labels
        key_rels=None,  # gt key rels
        relmaps=None,  # gt relmaps
        refine_bboxes=None,  # RM: refined object bboxes (score is changed)
        formatted_bboxes=None,  # OD: Transform the refine_bboxes for object detection evaluation
        refine_scores=None,  # RM: refined object scores (before softmax)
        refine_dists=None,  # RM: refined object dists (after softmax)
        refine_labels=None,  # RM: refined object labels
        target_labels=None,  # RM: assigned object labels for training the relation module.
        rel_scores=None,  # RM: predicted relation scores (before softmax)
        rel_dists=None,  # RM: predicted relation prob (after softmax)
        att_dists=None,
        spa_dists=None,
        con_dists=None,
        att_labels=None,
        spa_labels=None,
        con_labels=None,
        triplet_scores=None,  # RM: predicted triplet scores (the multiplication of sub-obj-rel scores)
        ranking_scores=None,  # RM: predicted ranking scores for rank the triplet
        rel_pair_idxes=None,  # gt rel_pair_idxes / RM: training/testing sampled rel_pair_idxes
        rel_labels=None,  # gt rel_labels / RM: predicted rel labels
        target_rel_labels=None,  # RM: assigned target rel labels
        target_key_rel_labels=None,  # RM: assigned target key rel labels
        saliency_maps=None,  # SAL: predicted or gt saliency map
        attrs=None,  # gt attr
        rel_cap_inputs=None,  # gt relational caption inputs
        rel_cap_targets=None,  # gt relational caption targets
        rel_ipts=None,  # gt relational importance scores
        tgt_rel_cap_inputs=None,  # RM: assigned target relational caption inputs
        tgt_rel_cap_targets=None,  # RM: assigned target relational caption targets
        tgt_rel_ipts=None,  # RM: assigned target relational importance scores
        rel_cap_scores=None,  # RM: predicted relational caption scores
        rel_cap_seqs=None,  # RM: predicted relational seqs
        rel_cap_sents=None,  # RM: predicted relational decoded captions
        rel_ipt_scores=None,  # RM: predicted relational caption ipt scores
        cap_inputs=None,
        cap_targets=None,
        cap_scores=None,
        cap_scores_from_triplet=None,
        alphas=None,
        rel_distribution=None,
        obj_distribution=None,
        word_obj_distribution=None,
        cap_seqs=None,
        cap_sents=None,
        img_shape=None,
        scenes=None,  # gt scene labels
        target_scenes=None,  # target_scene labels
        add_losses=None,  # For Recording the loss except for final object loss and rel loss, e.g.,
        # use in causal head or VCTree, for recording auxiliary loss
        head_spec_losses=None,  # For method-specific loss
        pan_results=None,
    ):
        self.__dict__.update(locals())
        del self.__dict__['self']

    def is_none(self):
        return all(
            [v is None for k, v in self.__dict__.items() if k != 'self'])

    # HACK: To turn this object into an iterable
    def __len__(self):
        return 1

    # HACK:
    def __getitem__(self, i):
        return self

    # HACK:
    def __iter__(self):
        yield self
