import torch
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
    
