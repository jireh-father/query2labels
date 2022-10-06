from torchvision.datasets.folder import ImageFolder
import json
import os
import numpy as np
import random


class FashionAttributeMultiLabelDataset(ImageFolder):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""

    def __init__(self, multi_label_json, label_type, root, transform=None):
        super(FashionAttributeMultiLabelDataset, self).__init__(root, transform=transform)
        multi_label_dict = json.load(open(multi_label_json, encoding='utf-8'))
        label_dict = multi_label_dict[label_type]

        class_dict = {}
        for k in label_dict:
            vals = label_dict[k]
            for v in vals:
                class_dict[v] = True

        classes = list(class_dict.keys())
        classes.sort()
        self.classes = classes
        class_map = {c: i for i, c in enumerate(classes)}
        self.samples = []
        for k in label_dict:
            vals = label_dict[k]
            labels = []
            for v in vals:
                labels.append(class_map[v])
            self.samples.append((k, labels))

        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        while True:
            try:
                path, multi_labels = self.samples[index]
                target = [0] * len(self.classes)
                for l in multi_labels:
                    target[l] = 1
                path = os.path.join(self.root, path)
                sample = np.array(self.loader(path))
                if self.transform is not None:
                    sample = self.transform(image=sample)['image']
                return sample, np.array(target, dtype=np.float32)
            except Exception as e:
                # traceback.print_exc()
                print(str(e), path)
                index = random.randint(0, len(self) - 1)