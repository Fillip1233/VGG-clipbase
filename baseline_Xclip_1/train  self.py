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
from transformers import CLIPModel, CLIPTextModel, AutoTokenizer
from getgt import AverageMeter,selfatt
from transformers import AutoProcessor, XCLIPVisionModel,XCLIPVisionConfig

from getgt import process_gt
import wandb,map
# from saveit import savegt
# from saveit import savepred
"""------------------------------------some settings----------------------------------------"""
# conf = Config()
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
    parser.add_argument('-data_path', default='/data/scene_understanding/action_genome/', type=str)
    parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
    parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
    parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
    parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
    parser.add_argument('-nepoch', help='epoch number', default=50, type=float)
    parser.add_argument('-bce_loss', action='store_true')
    parser.add_argument('-checkpoint_path', type=str,default='./checkpoint')

    #
    parser.add_argument('-use_wandb', default=None,type=str)
   
    parser.add_argument('-gpu', default="1",type=str)
    parser.add_argument('-save_test', default=None,type=str)
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

AG_dataset_train = AG(mode="train", datasize=args.datasize, data_path=args.data_path, filter_nonperson_box_frame=True,
                      filter_small_box=False if args.mode == 'predcls' else True)
# dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=4,
#                                                 collate_fn=cuda_collate_fn, batch_size=128,pin_memory=False)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=4,
                                                batch_size=args.batch_size,drop_last=True,pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=args.datasize, data_path=args.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if args.mode == 'predcls' else True)
# AG_dataset_test = AG(mode="test",data_path=args.data_path, filter_nonperson_box_frame=True,
#                      filter_small_box=False if args.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=4,
                                              batch_size=args.batch_size, drop_last=True, pin_memory=False)
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
model_without_ddp = base_Model()
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

if args.use_wandb is not None:
    wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.lr,
    "epochs": 50,
    }
    )
re_dir = "./re2"
if not os.path.exists(re_dir):
    os.makedirs(re_dir)
# some parameters
tr = []
mlm = nn.MultiLabelSoftMarginLoss()
for epoch in tqdm(range(args.nepoch)):
    # model.train()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    model = base_Model()
    model.to(device)
    text1_input = ["a photo of a __background__","a photo of a person","a photo of a bag", "a photo of a bed","a photo of a blanket","a photo of a book","a photo of a box",
                "a photo of a broom","a photo of a chair","a photo of a closetcabinet","a photo of a clothes","a photo of a cupglassbottle",
                "a photo of a dish","a photo of a door","a photo of a doorknob","a photo of a doorway","a photo of a floor",
                "a photo of a food","a photo of a groceries","a photo of a laptop","a photo of a light","a photo of a medicine",
                "a photo of a mirror","a photo of a papernotebook","a photo of a phonecamera","a photo of a picture","a photo of a pillow",
                "a photo of a refrigerator","a photo of a sandwich","a photo of a shelf","a photo of a shoe","a photo of a sofacouch",
                "a photo of a table","a photo of a television","a photo of a towel","a photo of a vacuum","a photo of a window"]
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_input = tokenizer(text1_input,padding=True, return_tensors="pt").to(device)
    filenamere = "outputre{}.txt".format(epoch)
    filepathre = os.path.join(re_dir, filenamere)
    ss=0
    for b in tqdm(range(len(dataloader_train))):
        data = next(train_iter)
        im_data = data[0].cuda()
        t1=data[1].cuda()
        obj_gt =data[2].cuda()
        obj_gt = obj_gt.reshape((8*args.batch_size,37))
        if 4249 in t1 and ss==0 and args.save_test is not None:
            inde= torch.where(t1==4249)[0].cpu()
            obj_gt_1=data[2].cpu()
            oo=obj_gt_1[inde][0,:,:]
            np.savetxt("output_gt.txt", oo,fmt='%.4f')
            ss=ss+1
        im_data=im_data.view(args.batch_size*8,3,600,1000)
        video_input = list(im_data)
        processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        # pixel_values = processor(videos=video_input, return_tensors="pt").pixel_values
        # pixel_values.to(torch.device("cuda"))
        pixel_values = processor(videos=video_input, return_tensors="pt").pixel_values.to(device)
        with torch.set_grad_enabled(True):
        # with torch.no_grad():
            # savegt(obj_gt,b)
            output = model(text_input, pixel_values,device)
            loss = mlm(output[2],obj_gt)
            print("loss:",loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
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
    model.eval()
    with torch.no_grad():
        outputs1 = torch.Tensor().to(device)
        obj_gt1 = torch.Tensor().to(device)
        # if epoch % 5 == 0:
        for b in range(len(dataloader_test)):
            data = next(test_iter)
            im_data = data[0].cuda()
            obj_gt =data[2].cuda()
            obj_gt = obj_gt.reshape((8*args.batch_size,37))
                # obj_gt = obj_gt.reshape(args.batch_size,37)
            im_data=im_data.view(args.batch_size*8,3,600,1000)
            video_input = list(im_data)
            processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
            pixel_values = processor(videos=video_input, return_tensors="pt").pixel_values.to(device)
            output = model(text_input, pixel_values,device)
                # torch.cat((concatenated_obj_gt, obj_gt), dim=0)
            # obj_gt = torch.zeros((8,35)).to(device)
            obj_gt1 = torch.cat((obj_gt1,obj_gt),dim=1).to(device)
            # pred = torch.zeros((8,35)).to(device)
                # outputs = outputs.append(output[2])
            outputs1 = torch.cat((outputs1,output[2]),dim=1).to(device)
        # mAP, _, ap = map.charades_map(torch.vstack(outputs1), torch.vstack(obj_gt1))
        mAP, _, ap = map.charades_map(torch.vstack(tuple(outputs1)), torch.vstack(tuple(obj_gt1)))

        
        if args.use_wandb is not None:
            wandb.log({"mAP": mAP})
            wandb.watch(model)
        # mAP, _, ap = map.charades_map(output[2],obj_gt)
        print(mAP)
    torch.save({"state_dict": model.state_dict()}, os.path.join(args.save_path, "model_{}.tar".format(epoch)))