import torchvision.transforms as transforms
import sys
sys.path.append("/home/alvin/.local/lib/python3.8/site-packages")
# from dataset.cocodataset import CoCoDataset
from dataloader.action_genome import AG
from dataloader.cutout import SLCutoutPIL
from randaugment import RandAugment
import os.path as osp




def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size_h, args.img_size_w)),
                                            RandAugment(),
                                               transforms.ToTensor(),
                                               normalize]
    try:
        # for q2l_infer scripts
        if args.cutout:
            print("Using Cutout!!!")
            train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    except Exception as e:
        Warning(e)
    train_data_transform = transforms.Compose(train_data_transform_list)

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size_h, args.img_size_w)),
                                            transforms.ToTensor(),
                                            normalize])
    
    # if args.dataname == 'coco' or args.dataname == 'coco14':
    if args.dataname == 'agtraindata':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = AG(
            mode="train",datasize=args.datasize,data_path=args.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if args.mode == 'predcls' else True,

            image_dir=osp.join(dataset_dir, 'frames'),
            # anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            anno_path=None,
            input_transform=train_data_transform,
            labels_path='/mnt/cephfs/home/alvin/jiashuning/query2labels/dataloader/train_ag.npy',
        )
        val_dataset = AG(
            mode="test", datasize=args.datasize, data_path=args.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if args.mode == 'predcls' else True,

            image_dir=osp.join(dataset_dir, 'frames'),
            # anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            # anno_path=osp.join('/mnt/cephfs/dataset/coco2014/annotations/instances_val2014.json'),
            anno_path=None,
            input_transform=test_data_transform,
            labels_path='/mnt/cephfs/home/alvin/jiashuning/query2labels/dataloader/test_ag.npy',
        )    
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
   
    return train_dataset, val_dataset

# class AG(list):
#       def __len__(train_dataset):
#         return len(train_dataset)


# class AG(list):
#       def __len__(train_dataset):
#         return len(train_dataset)







"""------------------------------------yuandaima----------------------------------------"""
    # if args.dataname == 'coco' or args.dataname == 'coco14':
    #     # ! config your data path here.
    #     dataset_dir = args.dataset_dir
    #     train_dataset = CoCoDataset(
    #         image_dir=osp.join(dataset_dir, 'train2014'),
    #         # anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
    #         anno_path=osp.join('/mnt/cephfs/dataset/coco2014/annotations/instances_train2014.json'),
    #         input_transform=train_data_transform,
    #         labels_path='/mnt/cephfs/home/alvin/jiashuning/query2labels/data/coco/train_label_vectors_coco14.npy',
    #     )
    #     val_dataset = CoCoDataset(
    #         image_dir=osp.join(dataset_dir, 'val2014'),
    #         # anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
    #         anno_path=osp.join('/mnt/cephfs/dataset/coco2014/annotations/instances_val2014.json'),
    #         input_transform=test_data_transform,
    #         labels_path='/mnt/cephfs/home/alvin/jiashuning/query2labels/data/coco/val_label_vectors_coco14.npy',
    #     )    

    # else:
    #     raise NotImplementedError("Unknown dataname %s" % args.dataname)

    # print("len(train_dataset):", len(train_dataset)) 
    # print("len(val_dataset):", len(val_dataset))
    # return train_dataset, val_dataset
