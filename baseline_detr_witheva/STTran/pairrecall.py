import numpy as np
from tqdm import tqdm
from functools import reduce
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
import copy

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

#读取gt数据
def loadgt(path):
    try:
        data=np.load(path,allow_pickle=True).item()
        return data["triples"],data["boxes"]
    except:
        pass
#读取pred
def loadpre(path):
    data=np.load(path,allow_pickle=True).item()
    return data["triples"],data["boxes"],data['rlsscore']

def topair(triples):
    # for i in range(len(triples)):
    
    x=np.delete(triples,1,axis=1)
    # print(x)
    return x

def pairprocess(triples):
    fb=copy.deepcopy(triples)
    fb=list(fb)
    for i in range(0,len(triples)):      
        while countit(fb,triples[i]) != 1:
            fb=removeit(fb,triples[i])
    fb2=copy.deepcopy(fb)
    for m in fb2:
        if m[0]==1:
            fb=removeit(fb,m)
        
    return fb

def countit(counted,countwhat):
    num=0
    for i in counted:
        if ifequal(i,countwhat):
            num=num+1
    return num

def ifequal(x,y):
    equalnum=0
    for m in range(2):

        if x[m]==y[m]:
            equalnum=equalnum+1
    return equalnum == 2

def removeit(name,item):
    for i in range(len(name)):
        if ifequal(name[i],item):
            global zz
            zz=np.delete(name,i,axis=0)
    return zz

def findindex(list,findwhat):
    for i in range(len(list)):
        if ifequal(list[i],findwhat):
            return i

def findbox(gt_boxes,gt_triplets,gt_has_match,gt_pair_0):
    boxes=[]
    for i in gt_triplets[gt_has_match]:
        boxes.append(gt_boxes[findindex(gt_pair_0,i)])
    return boxes


                
            

class BasicSceneGraphEvaluator:
    def __init__(self, mode):
        self.result_dict = {}
        self.mode = mode
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}

    def reset_result(self):
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}

    def print_stats(self):
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))
    def evaluate_scene_graph(self,gt_triplets, pred_triplets,gt_triplet_boxes,pred_triplet_boxes):
        evaluate_from_dict(gt_triplets, pred_triplets,gt_triplet_boxes,pred_triplet_boxes,self.result_dict,self.mode)

def evaluate_from_dict(gt_triplets, pred_triplets,gt_triplet_boxes,pred_triplet_boxes,result_dict,mode):
    pred_to_gt = _compute_pred_matches(
            gt_triplets,
            pred_triplets,
            gt_triplet_boxes,
            pred_triplet_boxes,
            iou_thresh=0.5
        )
    for k in result_dict[mode + '_recall']:
#union1d,求并集
        match = reduce(np.union1d, pred_to_gt[:k])

        rec_i = float(len(match)) / float(gt_triplets.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)
    return pred_to_gt

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
                                         findbox(gt_boxes,gt_triplets,gt_has_match,gt_pair_0),
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
evaluator=BasicSceneGraphEvaluator('sgdet')
for i in tqdm(range(1,1738)):
    for z in tqdm(range(100)):
        # try:
            
        gt_triples,gt_boxes=loadgt('/mnt/cephfs/home/alvin/yingqi/STTran/video.predcls/video{}/frame{}/gt.npy'.format(i,z))
        sgdet_t,sgdet_box,sgdet_socre=loadpre('/mnt/cephfs/home/alvin/yingqi/STTran/video.sgdet/video{}/frame{}/with.npy'.format(i,z))
        gt_pair_0=topair(gt_triples)
        gt_pair=pairprocess(gt_pair_0)
        sgdet_pair_0=topair(sgdet_t)
        sgdet_pair=pairprocess(sgdet_pair_0)
        evaluator.evaluate_scene_graph(gt_pair,sgdet_pair,gt_boxes,sgdet_box)
        # except:
        #     pass
print('-------------------------with constraint-------------------------------')
evaluator.print_stats()