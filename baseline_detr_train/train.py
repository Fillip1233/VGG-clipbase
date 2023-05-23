import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import copy
from tqdm import tqdm

### import detr match loss backward
from engine import detr_backward

from dataloader.action_genome import AG, cuda_collate_fn
# from lib.object_detector import detector
# from lib.config import Config
# from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
# from lib.sttran import STTran
# from save_check_point import save_sh_n_codes
# from torchstat import stat

## some import according to detr
import util.misc as utils
import argparse
import random
import wandb

## some function need to add
from getgt import AverageMeter
import numpy as np
from models import build_model
import math
import os
import sys
from typing import Iterable
import util.misc as utils
from torch.autograd import profiler
from torch.nn.utils.rnn import pad_sequence

"""------------------------------------some settings----------------------------------------"""
# conf = Config()
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    ###改成了resnet18，原本为resnet50
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    ###STTran
    parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
    parser.add_argument('-save_path', default='data1/', type=str)
    parser.add_argument('-model_path', default=None, type=str)
    parser.add_argument('-data_path', default='/data/scene_understanding/action_genome/', type=str)
    parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
    parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
    parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
    parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
    parser.add_argument('-nepoch', help='epoch number', default=10, type=float)
    parser.add_argument('-bce_loss', action='store_true')
    parser.add_argument('-checkpoint_path', type=str,default='./checkpoint')

    #
    parser.add_argument('-use_wandb', default=None,type=str)
    return parser

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3,4,5,6,7"
device_ids=[0,1,2,3,4,5,6]

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
print('The CKPT saved here:', args.save_path)
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(args.enc_layers, args.dec_layers))
# for i in args.args:
#     print(i,':', args.args[i])
"""-----------------------------------------------------------------------------------------"""
if args.use_wandb is not None:
    wandb.init(project="VGG-baselinefor4", entity="vggbaseline")
    wandb.config = {
    "learning_rate": args.lr,
    "epochs": args.nepoch
    }
# profiler = torch.autograd.profiler.Profile()
AG_dataset_train = AG(mode="train", datasize=args.datasize, data_path=args.data_path, filter_nonperson_box_frame=True,
                      filter_small_box=False if args.mode == 'predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=4,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=args.datasize, data_path=args.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if args.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=4,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#gpu setting
# gpu_device = torch.device("cuda:0")
# gpu_num=0

##distributed training from detr, we can also unused it
utils.init_distributed_mode(args)
print("git:\n  {}\n".format(utils.get_sha()))

## fix the seed for reproducibility  copy from detr ,use as distributed training
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

## I have removed the object detector 

##built model ,use the method from detr
model, criterion, postprocessors = build_model(args)
model = nn.DataParallel(model, device_ids=device_ids)
model.to(device)
criterion.to(device)
model_without_ddp = model
if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

param_dicts = [
    {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,
    },
]
optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

## maybe STTran's model has some use to us ,I haven't delete it

# model = STTran(mode=conf.mode,
#                attention_class_num=len(AG_dataset_train.attention_relationships),
#                spatial_class_num=len(AG_dataset_train.spatial_relationships),
#                contact_class_num=len(AG_dataset_train.contacting_relationships),
#                obj_classes=AG_dataset_train.object_classes,
#                enc_layer_num=conf.enc_layer,
#                dec_layer_num=conf.dec_layer).to(device=gpu_device)


## use as distributed training , copy from detr
model_without_ddp = model
if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
if args.frozen_weights is not None:
    assert args.masks, "Frozen training is meant for segmentation only"

## use to see the model parameter
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameters: %.2fM" % (total/1e6))

## just copy from detr ,I haven't understand this part
param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
if args.frozen_weights is not None:
    checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    model_without_ddp.detr.load_state_dict(checkpoint['model'])

# evaluator =BasicSceneGraphEvaluator(mode=args.mode,
#                                     AG_object_classes=AG_dataset_train.object_classes,
#                                     AG_all_predicates=AG_dataset_train.relationship_classes,
#                                     AG_attention_predicates=AG_dataset_train.attention_relationships,
#                                     AG_spatial_predicates=AG_dataset_train.spatial_relationships,
#                                     AG_contacting_predicates=AG_dataset_train.contacting_relationships,
#                                     iou_threshold=0.5,
#                                     constraint='with')


# # loss function, default Multi-label margin loss 
# if args.bce_loss:
#     ce_loss = nn.CrossEntropyLoss()
#     bce_loss = nn.BCELoss()
# else:
#     ce_loss = nn.CrossEntropyLoss()
#     mlm_loss = nn.MultiLabelMarginLoss()

# # optimizer
# if args.optimizer == 'adamw':
#     optimizer = AdamW(model.parameters(), lr=args.lr)
# elif args.optimizer == 'adam':
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
# elif args.optimizer == 'sgd':
#     optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)

# scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

# some parameters
tr = []
data_time = AverageMeter()
batch_time =AverageMeter()
for epoch in tqdm(range(args.nepoch)):
    model.train()
    start = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    torch.cuda.synchronize()
    end1=time.time()
    with profiler.profile(use_cuda=True) as prof:
        for b in tqdm(range(len(dataloader_train))):
            # measure data loading time
            
            data = next(train_iter)

            im_data = data[0].cuda()
            print(im_data.shape)
            gt_annotation = AG_dataset_train.gt_annotations[data[1]]
            torch.cuda.synchronize()
            data_time.update(time.time() - end1)
            print("data_time: %.2f" % (data_time.val))
            print("Average_data_time: %.2f"% (data_time.avg))
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
            targets_my=[]
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
            
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

            loss_dict = criterion(pred, gt_annotation)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            for i,(key,value) in enumerate(loss_dict.items()):
                print(f'{key}:{value} ',end='')
                if i<len(loss_dict)-1:
                    print(',',end='')
            print(losses.item())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                            for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            loss_value = losses_reduced_scaled.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)
            optimizer.zero_grad()
            loss_backtime=AverageMeter()
            torch.cuda.synchronize()
            backtime=time.time()
            losses.backward()
            torch.cuda.synchronize()
            loss_backtime.update(time.time()-backtime)
            print("loss_backtime:%.2f"%(loss_backtime.val))
            normtime=AverageMeter()
            torch.cuda.synchronize()
            no1=time.time()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            torch.cuda.synchronize()
            normtime.update(time.time()-no1)
            print("norm_time:%.2f"%(normtime.val))
            optimizer.step()
            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end1)
            print("batch_time: %.2f" % (batch_time.val))
            print("Average_batch_time: %.2f"% (batch_time.avg))
            end1 = time.time()
            if args.use_wandb is not None:
                wandb.log({"loss": losses.item(),"loss_ce_att_rls":loss_dict["loss_ce_att_rls"].item(),"att_rls_error":loss_dict["att_rls_error"].item(),"loss_ce_spa_rls": \
                loss_dict["loss_ce_spa_rls"].item(),"loss_ce_con_rls":loss_dict["loss_ce_con_rls"].item(),"loss_ce":loss_dict["loss_ce"].item(),"class_error":loss_dict["class_error"].item(), \
                "loss_bbox":loss_dict["loss_bbox"].item(),"loss_giou":loss_dict["loss_giou"].item(),"cardinality_error":loss_dict["cardinality_error"].item(),"loss_ce_att_rls_0": \
                loss_dict["loss_ce_att_rls_0"].item(),"att_rls_error_0":loss_dict["att_rls_error_0"].item(),"loss_ce_spa_rls_0":loss_dict["loss_ce_spa_rls_0"].item(),"loss_ce_con_rls_0":loss_dict["loss_ce_con_rls_0"].item(), \
                "loss_ce_0":loss_dict["loss_ce_0"].item(),"loss_bbox_0":loss_dict["loss_bbox_0"].item(),"loss_giou_0":loss_dict["loss_giou_0"].item(),"cardinality_error_0":loss_dict["cardinality_error_0"].item(), \
                "data_time":data_time.val,"data_time_avg":data_time.avg,"batch_time":batch_time.val,"batch_time_avg":batch_time.avg})
                wandb.watch(model)
    print(prof)
    prof.export_chrome_trace("results.json")        
    metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
    metric_logger.update(class_error=loss_dict_reduced['class_error'])
    metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    lr_scheduler.step()
    
    torch.save({"state_dict": model.state_dict()}, os.path.join(args.save_path, "model_{}.tar".format(epoch)))
    # if args.output_dir:
    #         checkpoint_paths = [output_dir / 'checkpoint.pth']
    #         # extra checkpoint before LR drop and every 100 epochs
    #         if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
    #             checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
    #         for checkpoint_path in checkpoint_paths:
    #             utils.save_on_master({
    #                 'model': model_without_ddp.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'lr_scheduler': lr_scheduler.state_dict(),
    #                 'epoch': epoch,
    #                 'args': args,
    #             }, checkpoint_path
        
        
    #     attention_distribution = pred["attention_distribution"]
    #     spatial_distribution = pred["spatial_distribution"]
    #     contact_distribution = pred["contacting_distribution"]

    #     ## the method to get the gt ,the method is change according to the obj_detector
    #     pred=get_gt(im_data, im_info, gt_boxes, num_boxes, gt_annotation ,gpu_num,im_all=None)

    #     attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=attention_distribution.device).squeeze()
    #     if not args.bce_loss:
    #         # multi-label margin loss or adaptive loss
    #         spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=attention_distribution.device)
    #         contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=attention_distribution.device)
    #         for i in range(len(pred["spatial_gt"])):
    #             spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
    #             contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])

    #     else:
    #         # bce loss
    #         spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=attention_distribution.device)
    #         contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=attention_distribution.device)
    #         for i in range(len(pred["spatial_gt"])):
    #             spatial_label[i, pred["spatial_gt"][i]] = 1
    #             contact_label[i, pred["contacting_gt"][i]] = 1

    #     losses = {}
    #     # we can use this after we change the match function
    #     # losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

    #     #delete it when we complete the match function work
    #     attention_distribution=attention_distribution[:,0,:]
    #     spatial_distribution=spatial_distribution[:,0,:]
    #     contact_distribution=contact_distribution[:,0,:]

    #     losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
    #     if not args.bce_loss:
    #         losses["spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label)
    #         losses["contact_relation_loss"] = mlm_loss(contact_distribution, contact_label)

    #     else:
    #         losses["spatial_relation_loss"] = bce_loss(spatial_distribution, spatial_label)
    #         losses["contact_relation_loss"] = bce_loss(contact_distribution, contact_label)

    #     optimizer.zero_grad()
    #     loss = sum(losses.values())
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
    #     optimizer.step()

    #     tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

    #     if b % 1000 == 0 and b >= 1000:
    #         time_per_batch = (time.time() - start) / 1000
    #         print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
    #                                                                             time_per_batch, len(dataloader_train) * time_per_batch / 60))

    #         mn = pd.concat(tr[-1000:], axis=1).mean(1)
    #         print(mn)
    #         start = time.time()
    #     ## try to clear the cuda
    #     # torch.cuda.empty_cache()
    #     # gc.collect()

    # torch.save({"state_dict": model.state_dict()}, os.path.join(args.save_path, "model_{}.tar".format(epoch)))
    # print("*" * 40)
    # print("save the checkpoint after {} epochs".format(epoch))

#     model.eval()
#     with torch.no_grad():
#         for b in range(len(dataloader_test)):
#             data = next(test_iter)

#             im_data = copy.deepcopy(data[0].cuda(0))
#             im_info = copy.deepcopy(data[1].cuda(0))
#             gt_boxes = copy.deepcopy(data[2].cuda(0))
#             num_boxes = copy.deepcopy(data[3].cuda(0))
#             gt_annotation = AG_dataset_test.gt_annotations[data[4]]

#             # entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
#             pred = model(im_data)
#             evaluator.evaluate_scene_graph(gt_annotation, pred)
#         print('-----------', flush=True)
#     score = np.mean(evaluator.result_dict[args.mode + "_recall"][20])
#     evaluator.print_stats()
#     evaluator.reset_result()
#     scheduler.step(score)
# # opt1=vars(conf)
# # save_sh_n_codes(opt1)


