import os

import numpy as np

import torchvision


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, transform=None,
                 target_transform=None, transforms=None):
        if os.path.basename(annFile) == annFile:
            annFile = os.path.join(root, annFile)

        super(CocoDetection, self).__init__(root, annFile)

        self._post_transform = transform
        self._post_target_transform = target_transform

        self.classes = ["BACKGROUND"] + [
            v["name"] for v in self.coco.cats.values()]
        self.class_names = self.classes

    def __getitem__(self, index):
        image, target = super(CocoDetection, self).__getitem__(index)

        anns = target

        bboxes = []
        labels = []

        cat_ids = self.coco.getCatIds()
        for ann in anns:
            labels.append(cat_ids.index(ann["category_id"]) + 1)

            x1, y1, w, h = ann["bbox"]
            x2 = x1 + w
            y2 = y1 + h

            bboxes.append([x1, y1, x2, y2])

        image = np.array(image)
        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self._post_transform:
            image, bboxes, labels = self._post_transform(image, bboxes, labels)
        if self._post_target_transform:
            bboxes, labels = self._post_target_transform(bboxes, labels)

        return image, bboxes, labels
