import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset
from utils.cutout import SLCutoutPIL
from randaugment import RandAugment
import os.path as osp
import json
import random

def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
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
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )    
    elif args.dataname == "fashion_attr":
        from dataset.fashion_attr import FashionAttributeMultiLabelDataset
        train_label_file, val_label_file = args.label_file.split(",")
        train_dataset_dir, val_dataset_dir = args.dataset_dir.split(",")
        train_dataset = FashionAttributeMultiLabelDataset(
            train_label_file, args.label_type, train_dataset_dir, train_data_transform
        )

        val_dataset = FashionAttributeMultiLabelDataset(
            val_label_file, args.label_type, val_dataset_dir, test_data_transform
        )
    elif args.dataname == "tagger":
        from dataset import tagger
        label_dict = json.load(open(args.label_file, encoding='utf-8'))

        tag_set = set()
        for file_name in label_dict:
            tags = [v for v in label_dict[file_name]["tags"].split(", ") if v in tagger.val_to_key_map]
            tag_set.update(tags)
        tag_list = list(tag_set)
        tag_list.sort()
        tag_to_cls_idx_map = {tag_list[i]: i for i in range(len(tag_list))}
        print("tag_to_cls_idx_map", tag_to_cls_idx_map)


        # label_dict format
        # {"file_name1": {"tags": "tag1, tag2, tag3"}, ...}
        # split label_dict into train and val using shuffle, val has 10% of the data
        label_items = list(label_dict.items())
        random.shuffle(label_items)
        num_train = int(len(label_items) * 0.9)
        train_label_dict = dict(label_items[:num_train])
        val_label_dict = dict(label_items[num_train:])

        while True:
            tag_set = set()
            for file_name in train_label_dict:
                tags = [v for v in train_label_dict[file_name]["tags"].split(", ") if v in tagger.val_to_key_map]
                tag_set.update(tags)
            if len(tags) != args.num_class:
                random.shuffle(label_items)
                num_train = int(len(label_items) * 0.9)
                train_label_dict = dict(label_items[:num_train])
                val_label_dict = dict(label_items[num_train:])
                continue

            tag_set = set()
            for file_name in val_label_dict:
                tags = [v for v in val_label_dict[file_name]["tags"].split(", ") if v in tagger.val_to_key_map]
                tag_set.update(tags)
            if len(tags) != args.num_class:
                random.shuffle(label_items)
                num_train = int(len(label_items) * 0.9)
                train_label_dict = dict(label_items[:num_train])
                val_label_dict = dict(label_items[num_train:])
                continue
            break

        print("make train dataset")
        train_dataset = tagger.Tagger(args.dataset_dir, train_label_dict, train_data_transform, tag_to_cls_idx_map)
        print("make val dataset")
        val_dataset = tagger.Tagger(args.dataset_dir, val_label_dict, test_data_transform, tag_to_cls_idx_map)
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
