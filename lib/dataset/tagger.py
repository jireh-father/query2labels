import torch.utils.data as data
import json
import os
import numpy as np
import random


class Tagger(data.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, image_dir, label_data, transform=None):
        file_name_to_tags = {}
        tag_set = set()
        self.transform = transform
        for file_name in label_data:
            tags = ", ".split(label_data[file_name]["tags"])
            file_name_to_tags[file_name] = tags
            tag_set.update(tags)

        # sort tag_set
        tag_list = list(tag_set)
        tag_list.sort()
        tag_to_cls_idx_map = {tag_list[i]: i for i in range(len(tag_list))}
        self.tag_to_cls_idx_map = tag_to_cls_idx_map
        self.image_dir = image_dir
        self.num_classes = len(tag_list)
        images = []
        labels = []
        num_no_images = 0
        for file_name in file_name_to_tags:
            image_path = os.path.join(image_dir, file_name)
            if not os.path.isfile(image_path):
                num_no_images += 1
                continue
            images.append(f"{file_name}.jpg")
            tags = file_name_to_tags[file_name]
            label = [0] * len(tag_list)
            for tag in tags:
                label[tag_to_cls_idx_map[tag]] = 1
            labels.append(label)

        print(f"num of classes: {len(tag_list)}")
        print(f"num_no_images: {num_no_images}")
        print(f"num_images: {len(images)}")
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        while True:
            try:
                file_name = self.images[index]
                target = self.labels[index]
                path = os.path.join(self.image_dir, file_name)
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample, np.array(target, dtype=np.float32)
            except Exception as e:
                # traceback.print_exc()
                print(str(e), file_name)
                index = random.randint(0, len(self) - 1)

    def __len__(self):
        return len(self.images)
