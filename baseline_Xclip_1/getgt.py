import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
def get_gt(im_data, im_info, gt_boxes, num_boxes, gt_annotation, gpu_num,im_all=None,):
    bbox_num = 0

    im_idx = []  # which frame are the relations belong to
    pair = []
    a_rel = []
    s_rel = []
    c_rel = []

    for i in gt_annotation:
        bbox_num += len(i)
    FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32).cuda(gpu_num)
    FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).cuda(gpu_num)
    FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32).cuda(gpu_num)
    HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64).cuda(gpu_num)

    bbox_idx = 0
    for i, j in enumerate(gt_annotation):
        for m in j:
            if 'person_bbox' in m.keys():
                FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['person_bbox'][0])
                FINAL_BBOXES[bbox_idx, 0] = i
                FINAL_LABELS[bbox_idx] = 1
                HUMAN_IDX[i] = bbox_idx
                bbox_idx += 1
            else:
                FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['bbox'])
                FINAL_BBOXES[bbox_idx, 0] = i
                FINAL_LABELS[bbox_idx] = m['class']
                im_idx.append(i)
                pair.append([int(HUMAN_IDX[i]), bbox_idx])
                a_rel.append(m['attention_relationship'].tolist())
                s_rel.append(m['spatial_relationship'].tolist())
                c_rel.append(m['contacting_relationship'].tolist())
                bbox_idx += 1
    
    ##due to the batchsize ,we only use 4 frame
    a_rel=a_rel[:4]
    s_rel=s_rel[:4]
    c_rel=c_rel[:4]

    pair = torch.tensor(pair).cuda(gpu_num)
    im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(gpu_num)

    counter = 0
    FINAL_BASE_FEATURES = torch.tensor([]).cuda(gpu_num)

    # while counter < im_data.shape[0]:
    #     #compute 10 images in batch and  collect all frames data in the video
    #     if counter + 10 < im_data.shape[0]:
    #         inputs_data = im_data[counter:counter + 10]
    #     else:
    #         inputs_data = im_data[counter:]
    #     base_feat = self.fasterRCNN.RCNN_base(inputs_data)
    #     FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat), 0)
    #     counter += 10

    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] * im_info[0, 2]
    # FINAL_FEATURES = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, FINAL_BBOXES)
    # FINAL_FEATURES = self.fasterRCNN._head_to_tail(FINAL_FEATURES)

    union_boxes = torch.cat((im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:6][pair[:, 0]], FINAL_BBOXES[:, 1:6][pair[:, 1]]),
                            torch.max(FINAL_BBOXES[:, 6:5][pair[:, 0]], FINAL_BBOXES[:, 6:5][pair[:, 1]])), 1)
    # union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
    FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
    pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
                        1).data.cpu().numpy()
    # spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

    ## remove features and union_feat, spatial_mask
    entry = {'boxes': FINAL_BBOXES,
            'labels': FINAL_LABELS, # here is the groundtruth
            'scores': FINAL_SCORES,
            'im_idx': im_idx,
            'pair_idx': pair,
            'human_idx': HUMAN_IDX,
            # 'features': FINAL_FEATURES,
            # 'union_feat': union_feat,
            'union_box': union_boxes,
            # 'spatial_masks': spatial_masks,
            'attention_gt': a_rel,
            'spatial_gt': s_rel,
            'contacting_gt': c_rel
            }

    return entry
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# class SelfAttention(nn.Module):
#     def __init__(self, embed_size, heads):
#         super(SelfAttention, self).__init__()
#         self.embed_size = embed_size
#         self.heads = heads
#         self.head_dim = embed_size // heads
        
#         assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
#         self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
    
#     def forward(self, values, keys, query, mask):
#         N = query.shape[0] # Batch size
#         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
#         # Split embedding into self.heads pieces
#         values = values.reshape(N, value_len, self.heads, self.head_dim)
#         keys = keys.reshape(N, key_len, self.heads, self.head_dim)
#         queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
#         # Calculate energy for all self.heads
#         energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
#         # Apply mask
#         if mask is not None:
#             energy = energy.masked_fill(mask == 0, float("-1e20"))
        
#         # Apply softmax to get attention scores
#         attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
#         # Multiply attention scores with values to get the weighted sum
#         out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        
#         # Concatenate all self.heads outputs
#         out = self.fc_out(out)
#         return out

class selfatt(nn.Module):
    def __init__(self, embed_size, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_size, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_size)
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: [seq_len, batch_size, d_model]
        src2, self_attn_weights = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.feed_forward(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, self_attn_weights

def process_gt(im_data,gt_annotation):
    targets_my=[]
    dictionaries=[]
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
    return dictionaries[:8]