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
from dataloader.get_dataset import get_datasets
### import detr match loss backward
from engine import detr_backward

from dataloader.action_genome import AG
from lib.AdamW import AdamW

## some import according to detr
import util.misc as utils
import argparse
import random
# import wandb

## some function need to add
import numpy as np
from models import build_model
import math
import os
import sys
from typing import Iterable
import util.misc as utils
from torch.autograd import profiler
from torch.nn.utils.rnn import pad_sequence

##Xclip_vision_model
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download
import requests
from newmodel import base_Model
from transformers import CLIPModel, CLIPTextModel, AutoTokenizer,CLIPProcessor,AutoProcessor
from getgt import AverageMeter,selfatt,save_checkpoint
# from transformers import AutoProcessor, XCLIPVisionModel,XCLIPVisionConfig

from getgt import process_gt
import evaluation
# import trainsample
import wandb,map
from metric import voc_mAP
import aslloss
import torch.utils.data
from datetime import datetime
# from saveit import savegt
# from saveit import savepred
"""------------------------------------some settings----------------------------------------"""
# conf = Config()

SEED = 777
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--lr_backbone', default=1e-3, type=float)
    parser.add_argument('-batch_size', default=16, type=int)
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
    parser.add_argument('-data_path', default='/data/scene_understanding/action_genome/', type=str)   #！
    parser.add_argument('--dataset_dir', help='dir of dataset', default='/mnt/cephfs/dataset/zhenjie/agtraindata')  #！
    parser.add_argument('--dataname', help='dataname', default='agtraindata')
    parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
    parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
    parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
    parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
    parser.add_argument('-nepoch', help='epoch number', default=80, type=float)
    parser.add_argument('-bce_loss', action='store_true')
    parser.add_argument('-checkpoint_path', type=str,default='./checkpoint')

    #
    parser.add_argument('-use_wandb', default=None,type=str)
   
    parser.add_argument('-gpu', default="1",type=str)
    parser.add_argument('-save_test', default=None,type=str)
    parser.add_argument('-num_class', default=35,type=str)
    parser.add_argument('--output',default='./output',metavar='DIR', 
                        help='path to output folder')
    
    #data
    parser.add_argument('--img_size_h', default=360, type=int,
                        help='size of input images')
    parser.add_argument('--img_size_w', default=640, type=int,
                        help='size of input images')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')              
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')
    
    #loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False, 
                        help='disable_torch_grad_focal_loss in asl')              
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                                            help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                                            help='scale factor for clip')

    parser.add_argument('--expname', default='default', type=str,
                                            help='experiment_name') 
    return parser



parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
print('The CKPT saved here:', args.save_path)
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device_ids=[0]
"""-----------------------------------------------------------------------------------------"""

model = base_Model()

# param_dicts = [
#     {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
#     {
#         "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
#         "lr": args.lr_backbone,
#     },
# ]
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


train_dataset, val_dataset = get_datasets(args)


idx=[56,4249,57,60,62,64,66,67,71,78]
sampler = torch.utils.data.SubsetRandomSampler(idx) 


dataloader_train = torch.utils.data.DataLoader(train_dataset, shuffle=False, num_workers=4,sampler=idx,
                                                batch_size=args.batch_size,drop_last=True,pin_memory=False)

dataloader_test = torch.utils.data.DataLoader(val_dataset, shuffle=False, num_workers=4,sampler=idx,
                                              batch_size=args.batch_size, drop_last=True, pin_memory=False)

# dataloader_train = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=4,
#                                                 batch_size=args.batch_size,drop_last=True,pin_memory=False)

# dataloader_test = torch.utils.data.DataLoader(val_dataset, shuffle=False, num_workers=4,
#                                               batch_size=args.batch_size, drop_last=True, pin_memory=False) 


np.random.seed(0)


if args.use_wandb is not None:
    wandb.init(
    # set the wandb project where this run will be logged
    project="shuning-vgg",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "epochs": 80,
    }
    )
re_dir = "./result_log"
if not os.path.exists(re_dir):
    os.makedirs(re_dir)
# some parameters
tr = []
# mlm = nn.MultiLabelSoftMarginLoss()
# mlm = nn.BCEWithLogitsLoss()
criterion = aslloss.AsymmetricLossOptimized(
    gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
    clip=args.loss_clip,
    disable_torch_grad_focal_loss=args.dtgfl,
    eps=args.eps,
)

tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# text1_input = ["a photo of a background","a photo of a person","a photo of a bag", "a photo of a bed","a photo of a blanket","a photo of a book","a photo of a box",
#             "a photo of a broom","a photo of a chair","a photo of a closetcabinet","a photo of a clothes","a photo of a cupglassbottle",
#             "a photo of a dish","a photo of a door","a photo of a doorknob","a photo of a doorway","a photo of a floor",
#             "a photo of a food","a photo of a groceries","a photo of a laptop","a photo of a light","a photo of a medicine",
#             "a photo of a mirror","a photo of a papernotebook","a photo of a phonecamera","a photo of a picture","a photo of a pillow",
#             "a photo of a refrigerator","a photo of a sandwich","a photo of a shelf","a photo of a shoe","a photo of a sofacouch",
#             "a photo of a table","a photo of a television","a photo of a towel","a photo of a vacuum","a photo of a window"]

text1_input = ["a photo of a bag", "a photo of a bed","a photo of a blanket","a photo of a book","a photo of a box",    #[1-5]
            "a photo of a broom","a photo of a chair","a photo of a closetcabinet","a photo of a clothes","a photo of a cupglassbottle",   #[6-10]
            "a photo of a dish","a photo of a door","a photo of a doorknob","a photo of a doorway","a photo of a floor",    #[11-15]
            "a photo of a food","a photo of a groceries","a photo of a laptop","a photo of a light","a photo of a medicine",   #[16-20]
            "a photo of a mirror","a photo of a papernotebook","a photo of a phonecamera","a photo of a picture","a photo of a pillow",   #[21-25]
            "a photo of a refrigerator","a photo of a sandwich","a photo of a shelf","a photo of a shoe","a photo of a sofacouch",   #[26-30]
            "a photo of a table","a photo of a television","a photo of a towel","a photo of a vacuum","a photo of a window"]  #[30-35]

filepathre = os.path.join(re_dir, args.expname)
if not os.path.exists(filepathre):
    os.makedirs(filepathre)
filename1 = 'log.txt'
f1=os.path.join(filepathre, filename1)
filename2 = 'mAP_result.txt'
f2=os.path.join(filepathre, filename2)
best_regular_mAP = 0
best_regular_epoch = -1
for epoch in tqdm(range(args.nepoch)):
    model.train()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)

    text_input = tokenizer(text1_input, padding=True, return_tensors="pt").to(device)
    
    ss=0

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for b in tqdm(range(len(dataloader_train))):
        data_time.update(time.time() - end)
        data = next(train_iter)
        im_data = data[0].cuda()
        t1=data[1].cuda()
        obj_gt =data[2].cuda()
        obj_gt = obj_gt.reshape((8*args.batch_size, 35)) # num_frame*bs, 37
        # if 4249 in t1 and ss==0 and args.save_test is not None:
        #     inde= torch.where(t1==4249)[0].cpu()
        #     obj_gt_1=data[2].cpu()
        #     oo=obj_gt_1[inde][0,:,:]
        #     np.savetxt("output_gt.txt", oo,fmt='%.4f')
        #     ss=ss+1
        # im_data = im_data.view(args.batch_size*8,3,600,1000)
        im_data.permute(0,1,3,4,2)
        im_data=im_data.reshape(8*args.batch_size,3,360,640)
        image_input = list(im_data)
        pixel_values = processor(images=image_input, return_tensors="pt").pixel_values.to(device)
        # pixel_values = processor(videos=video_input, return_tensors="pt").pixel_values.to(device)
        pixel_values = pixel_values.reshape(args.batch_size,8,3,224,224)
        output = model(text_input, pixel_values,device)
        # loss = mlm(output[2], obj_gt)
        # print("loss:",loss)
        loss = criterion(output[2], obj_gt)
        # losses.update(loss.item(), im_data.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), args.batch_size*8)

        print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(
                   epoch, b, len(dataloader_train), batch_time=batch_time,
                   data_time=data_time, loss=losses, lr=optimizer.param_groups[0]['lr'])))
        with open(f1, 'a') as f:
            f.write(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(
                   epoch, b, len(dataloader_train), batch_time=batch_time,
                   data_time=data_time, loss=losses, lr=optimizer.param_groups[0]['lr'])))
        if b%10==0 and args.use_wandb is not None:
            wandb.log({"loss": loss})
            wandb.watch(model)
        if args.save_test is not None and 4249 in t1:
            inde1= torch.where(t1==4249)[0].cpu()
            s1=output[2].cpu()
            s1=s1[inde1*8:inde1*8+8,:]
            np.savetxt(filepathre, s1.detach().numpy(),fmt='%.4f')
    lr_scheduler.step()
    print("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch))
    # if (epoch+1) % 5 == 0:
    # torch.save({"state_dict": model.state_dict()}, os.path.join(args.save_path, "model_{}.tar".format(epoch)))
        
    # if args.use_wandb is not None:
    #     wandb.log({"mAP": mAP})
    #     wandb.watch(model)
    # mAP, _, ap = map.charades_map(output[2],obj_gt)
    # print("map:",mAP)
    if epoch % 1 == 0:
        model.eval()
        # evaluator = trainsample.validate_epoch(val_loader, model, evaluator)
        # evaluator = Evaluator(num_validation_samples, num_classes)
        output_list = []
        gt_list = []
        saved_data =[]
        with torch.no_grad():
            # outputs1 = torch.Tensor().to(device)
            # obj_gt1 = torch.Tensor().to(device)
            # if epoch % 5 == 0:
            for b in tqdm(range(len(dataloader_train))):
                # data = next(train_iter)
                print(data[0].shape)
                im_data = data[0].cuda()
                obj_gt =data[2].cuda()
                obj_gt = obj_gt.reshape((8*args.batch_size,35))
                    # obj_gt = obj_gt.reshape(args.batch_size,37)
                # im_data=im_data.view(args.batch_size*8,3,600,1000)
                im_data.permute(0,1,3,4,2)
                im_data=im_data.reshape(8*args.batch_size,3,360,640)
                image_input = list(im_data)
                pixel_values = processor(images=image_input, return_tensors="pt").pixel_values.to(device)
                # pixel_values = processor(videos=video_input, return_tensors="pt").pixel_values.to(device)
                pixel_values = pixel_values.reshape(args.batch_size,8,3,224,224)
                output = model(text_input, pixel_values,device)
                # torch.cat((concatenated_obj_gt, obj_gt), dim=0)
                # obj_gt1 = torch.cat((obj_gt1,obj_gt),dim=0)
                    # outputs = outputs.append(output[2])
                # outputs1 = torch.cat((outputs1,output[2]),dim=0)
                # output_list.append(output[2].cpu())
                # gt_list.append(obj_gt.cpu())
                output_sm = nn.functional.sigmoid(output[2])
                _item = torch.cat((output_sm.detach().cpu(), obj_gt.detach().cpu()), 1)
                saved_data.append(_item)
                
            saved_data = torch.cat(saved_data, 0).numpy()
            saved_name = 'saved_data_tmp.txt'
            np.savetxt(os.path.join(args.output, saved_name), saved_data)
            filenamelist = ['saved_data_tmp.txt']
            # mAP, _, ap = evaluation.charades_map(torch.vstack(tuple(output_list)), torch.vstack(tuple(gt_list)))
            # mAP, _, ap = evaluation.charades_map(torch.cat(output_list), torch.cat(gt_list))
            print("Calculating mAP:")
            metric_func = voc_mAP 
            mAP, aps = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class, return_each=True)
            is_best = mAP > best_regular_mAP
            if mAP > best_regular_mAP:
                best_regular_mAP = max(best_regular_mAP, mAP)
                best_regular_epoch = epoch
                with open(f2, 'a') as f:
                    f.write(f"save the checkpoint in epoch:{best_regular_epoch}, mAP value:{best_regular_mAP}\n")
            save_checkpoint({"state_dict": model.state_dict(),}, is_best=is_best, filename=os.path.join(args.save_path, 'checkpoint.pth.tar'))
            now = datetime.now()
            current_time = now.strftime("%Y-%m-%d %H:%M:%S")
            with open(f2, 'a') as f:
                f.write(f"[{current_time}] Epoch: {epoch}, mAP value: {mAP}\n")
            print("mAP: {}".format(mAP))
            if args.use_wandb is not None:
                wandb.log({"mAP": mAP})
                wandb.watch(model)
            
            

        # torch.save({"state_dict": model.state_dict()}, os.path.join(args.save_path, "model_{}.tar".format(epoch)))

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
