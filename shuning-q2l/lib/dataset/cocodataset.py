import torch
import sys, os

from PIL import Image
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
from tqdm import tqdm
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import random
# from scipy.misc import imread
import imageio
import numpy as np
import pickle
import os
from dataloader.blob import prep_im_for_blob, im_list_to_blob

category_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, 
"11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20, 
"21": 21, "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "27": 27, "28": 28, "29": 29, "30": 30, 
"31": 31, "32": 32, "33": 33, "34": 34, "35": 35}

class AG(data.Dataset):
    def __init__(self, image_dir, anno_path, mode, datasize, input_transform=None, 
                filter_nonperson_box_frame=True,filter_small_box=False,
                labels_path=None,data_path=None,
                used_category=-1):
        root_path = data_path
        root = image_dir
        # self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        # with open('./data/coco/category.json','r') as load_category:
        #     self.category_map = json.load(load_category)
        
        self.category_map = category_map  #baoliu映射
        self.input_transform = input_transform    #baoliu
        self.labels_path = labels_path      #baoliu标签
        self.used_category = used_category            #baoliu

	    
        self.frames_path = os.path.join(root_path, 'frames/')
        self.object_classes = ['__background__']
        with open(os.path.join(root_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

# collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]


        print('-------loading annotations---------slowly-----------')

        if filter_small_box:
            with open(root_path + 'annotations/person_bbox.pkl', 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open('/mnt/cephfs/home/alvin/yingqi/New/STTran/dataloader/object_bbox_and_relationship_filtersmall.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
        else:
            with open(root_path + 'annotations/person_bbox.pkl', 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(root_path+'annotations/object_bbox_and_relationship.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
            f.close()
        print('--------------------finish!-------------------------')

        if datasize == 'mini':
            small_person = {}
            small_object = {}
            for i in list(person_bbox.keys())[:80000]:
                small_person[i] = person_bbox[i]
                small_object[i] = object_bbox[i]
            person_bbox = small_person
            object_bbox = small_object


        # collect valid frames
        video_dict = {}
        for i in person_bbox.keys():
            if object_bbox[i][0]['metadata']['set'] == mode: #train or testing?
                frame_valid = False
                for j in object_bbox[i]: # the frame is valid if there is visible bbox
                    if j['visible']:
                        frame_valid = True
                if frame_valid:
                    video_name, frame_num = i.split('/')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]

        self.video_list = []
        self.video_size = [] # (w,h)
        self.gt_annotations = []
        self.non_gt_human_nums = 0
        self.non_heatmap_nums = 0
        self.non_person_video = 0
        self.one_frame_video = 0
        self.valid_nums = 0
        self.obj_gt=[]

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            gt_obj_video=[]
            for j in video_dict[i]:
                if filter_nonperson_box_frame:
                    if person_bbox[j]['bbox'].shape[0] == 0:
                        self.non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self.valid_nums += 1

                # 
                
                gt_annotation_frame = [{'person_bbox': person_bbox[j]['bbox']}]
                gt_obj_frame=[]
                gt_obj_num=torch.zeros(35)
                # each frames's objects and human
                for k in object_bbox[j]:
                    if k['visible']:
                        assert k['bbox'] != None, 'warning! The object is visible without bbox'
                        k['class'] = self.object_classes.index(k['class'])
                        
                        gt_obj_num[k['class']-2]=1 #add
                        # gt_obj_num[0]=1
                        ####
                        k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
                        k['attention_relationship'] = torch.tensor([self.attention_relationships.index(r) for r in k['attention_relationship']], dtype=torch.long)
                        k['spatial_relationship'] = torch.tensor([self.spatial_relationships.index(r) for r in k['spatial_relationship']], dtype=torch.long)
                        k['contacting_relationship'] = torch.tensor([self.contacting_relationships.index(r) for r in k['contacting_relationship']], dtype=torch.long)
                        gt_annotation_frame.append(k)
                # gt_obj_num = gt_obj_num[2:37]
                gt_obj_frame.append(gt_obj_num)
                gt_obj_frame=torch.stack(gt_obj_frame,dim=0)
                gt_annotation_video.append(gt_annotation_frame)
                gt_obj_video.append(gt_obj_frame)

            # if len(video) > 2:
            if len(video) > 7:
                self.video_list.append(video)
                self.video_size.append(person_bbox[j]['bbox_size'])
                self.gt_annotations.append(gt_annotation_video)
                self.obj_gt.append(gt_obj_video)
            elif len(video) == 1:
                self.one_frame_video += 1
            else:
                self.non_person_video += 1

        print('x'*60)
        if filter_nonperson_box_frame:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} videos are invalid (no person), remove them'.format(self.non_person_video))
            print('{} videos are invalid (only one frame), remove them'.format(self.one_frame_video))
            print('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
        else:
            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            print('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(non_heatmap_nums))
        print('x' * 60)

        # self.labels = []
        # if os.path.exists(self.labels_path):
        #     self.labels = np.load(self.labels_path).astype(np.float64)
        #     self.labels = (self.labels > 0).astype(np.float64)
        # else:
        # print("No preprocessed label file found in {}.".format(self.labels_path))
        # l = len(self.video_list)
        # for i in tqdm(range(l)):
        #     item = self.video_list[i]
        #     # print(i)
        #     categories = self.getCategoryList(item[1])
        #     label = self.getLabelVector(categories)
        #     self.labels.append(label)

        # self.save_datalabels(labels_path)
        
        # import ipdb; ipdb.set_trace()

    def __getitem__(self, index):
        frame_names = self.video_list[index][:8]
        obj = self.obj_gt[index][:8]
        processed_ims = []
        obj = torch.stack(obj,dim=0)
        obj = obj.squeeze(1)
        processed_ims = []
        input1 = []
        input2 = []
        new_tensor_list = []
        im_scales = []
        # for idx, name in enumerate(frame_names):
        #     im = imageio.imread(os.path.join(self.frames_path, name)) # channel h,w,3
        #     im = im[:, :, ::-1] # rgb -> bgr
        #     # im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 360, 640) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        #     im, im_scale = prep_im_for_blob(im, [[[123.675, 116.28, 103.53]]], 448, 448)
        #     im_scales.append(im_scale)
        #     processed_ims.append(im)
        # blob = im_list_to_blob(processed_ims)
        # input = torch.from_numpy(blob)
        # input = input.permute(0, 3, 1, 2)
        for i in range(8):
            path = self.video_list[index][i]
            inputa = Image.open(os.path.join(self.frames_path, path)).convert("RGB")
            input1.append(inputa)
        for a in range(8):
            b = input1[a]
            inputb = self.input_transform(b)
            input2.append(inputb)
        for c  in input2:
            new_tensor = c.unsqueeze(0)
            new_tensor_list.append(new_tensor)
        input = torch.cat(new_tensor_list,dim=0)
        # input = self.input_transform(input)
        # for c in range(8):
        #     d = input2[c]
        #     input = self.input_transform(d)
        #     input3.append(input)
        # return input, self.labels[index]
        return input, obj, index
   
    # def getCategoryList(self, item):
    #     categories = set()
    #     for t in item:
    #         categories.add(t['category_id'])
    #     return list(categories)
    # def getCategoryList(self, items):
    #     categories = set()
    #     for i, item in enumerate(items):
    #         category_id = i+1
    #         categories.add(category_id)
    #     return list(categories)



    # def getLabelVector(self, categories):
    #     label = np.zeros(37)
    #     # label_num = len(categories)
    #     for c in categories:
    #         index = self.category_map[str(c)]-1
    #         label[index] = 1.0 # / label_num
    #     return label

    def __len__(self):
        return len(self.video_list)

    def save_datalabels(self, outpath):
        """
            Save datalabels to disk.
            For faster loading next time.
        """
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        labels = np.array(self.obj_gt)
        np.save(outpath, labels)


