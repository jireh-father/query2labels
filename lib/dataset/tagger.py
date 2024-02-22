import torch.utils.data as data
import traceback
import json
import os
import numpy as np
import random
from PIL import Image

val_to_key_map = {
    "bob hair": "hair_length",
    "short hair": "hair_length",
    "medium hair": "hair_length",
    "long hair": "hair_length",
    "no-curl": "curl_type",
    "c-curl perm": "curl_type",
    "s-curl perm": "curl_type",
    "s3-curl perm": "curl_type",
    "inner c-curl perm": "curl_type",
    "outer c-curl perm": "curl_type",
    "cs-curl perm": "curl_type",
    "ss-curl perm": "curl_type",
    "c-shaped curl perm": "curl_type",
    "s-shaped curl perm": "curl_type",
    "s3-shaped curl perm": "curl_type",
    "inner c-shaped curl perm": "curl_type",
    "outer c-shaped curl perm": "curl_type",
    "cs-shaped curl perm": "curl_type",
    "ss-shaped curl perm": "curl_type",
    "thin curl": "curl_width",
    "thick curl": "curl_width",
    "no-layered hair": "cut",
    "layered hair": "cut",
    "hush cut": "hair_style_name",
    "tassel cut": "hair_style_name",
    "hug perm": "hair_style_name",
    "build perm": "hair_style_name",
    "slick cut": "hair_style_name",
    "short cut": "hair_style_name",
    "layered cut": "hair_style_name",
    "full bangs": "bangs",
    "side bangs": "bangs",
    "see-through bangs": "bangs",
    "choppy bangs": "bangs",
    "faceline bangs": "bangs",
    "thin hair": "hair_thickness",
    "thick hair": "hair_thickness",
}


class Tagger(data.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, image_dir, label_data, transform, tag_to_cls_idx_map):
        file_name_to_tags = {}
        self.transform = transform
        for file_name in label_data:
            tags = [v for v in label_data[file_name]["tags"].split(", ") if v in val_to_key_map]
            file_name_to_tags[f"{file_name}.jpg"] = tags

        self.tag_to_cls_idx_map = tag_to_cls_idx_map
        self.image_dir = image_dir

        images = []
        labels = []
        num_no_images = 0
        for file_name in file_name_to_tags:
            image_path = os.path.join(image_dir, file_name)
            if not os.path.isfile(image_path):
                num_no_images += 1
                continue
            images.append(file_name)
            tags = file_name_to_tags[file_name]
            label = [0] * len(self.tag_to_cls_idx_map)
            for tag in tags:
                label[tag_to_cls_idx_map[tag]] = 1
            labels.append(label)

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
                sample = Image.open(path).convert("RGB")
                sample = np.array(sample)
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample, np.array(target, dtype=np.float32)
            except Exception as e:
                traceback.print_exc()
                index = random.randint(0, len(self) - 1)

    def __len__(self):
        return len(self.images)
