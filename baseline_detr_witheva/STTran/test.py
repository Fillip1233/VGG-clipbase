import numpy as np
np.set_printoptions(precision=4)
import copy
import torch

from dataloader.action_genome import AG, cuda_collate_fn

from lib.config import Config
# from lib.evaluation_recall import BasicSceneGraphEvaluator
from models.backbone import build_backbone
from models.transformer import build_transformer
from models.detr import DETR
# from lib.object_detector import detector
# from lib.sttran import STTran
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

import os

from get_result import get_triplets,triplet2Result,get_dist_info,evaluate
# from result_class import Configclass
conf = Config()
# for i in conf.args:
#     print(i,':', conf.args[i])
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda:0')

# cfg = Configclass.fromfile(conf.config)

backbone = build_backbone(conf)
transformer = build_transformer(conf)
model = DETR(
        backbone,
        transformer,
        num_classes=37,
        num_queries=conf.num_queries,
        aux_loss=conf.aux_loss,
    )
model.eval()
model.to(gpu_device)
# ckpt = torch.load(conf.model_path, map_location=gpu_device)
# model.load_state_dict(ckpt['state_dict'], strict=False)
# print('*'*50)
# print('CKPT {} is loaded'.format(conf.model_path))
#

with torch.no_grad():
    for b, data in tqdm(enumerate(dataloader)):
        im_data = copy.deepcopy(data[0].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[1]]
        gt_annotation=gt_annotation[:4]
        ## Processing gt
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
        ##

        pred=model(im_data)
        results_list = get_triplets(pred)
        sg_result = [triplet2Result(triplets)for triplets in results_list]
        print(b)
        if b == 1:
            break
    rank, _ = get_dist_info()
    # metric = evaluate(AG_dataset,gt_annotation,sg_result)
    # print(metric)
        


print('-------------------------with constraint-------------------------------')

print('-------------------------semi constraint-------------------------------')

print('-------------------------no constraint-------------------------------')
