from typing import List, Dict, Optional, Callable, Any

import torch
import imageio as io

import cv2
import torch.utils.data

from diploma.data.augmentation import Augmentor
from diploma.data.hierarchy import Hierarchy

import torch

from pathlib import Path


class HierarchyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples_dir: str,
        samples_list: str,
        augmentation: bool,
        input_size: int,
        categories: List[str],
        hierarchy: Hierarchy,
    ):
        self.samples_dir = samples_dir
        self.samples_list = samples_list
        self.augmentation = augmentation
        self.input_size = input_size
        self.categories = categories
        self.hierarchy = hierarchy
        self.samples = sorted([f"{samples_dir}/{sample}" for sample in samples_list])

        if self.augmentation:
            self.aug = Augmentor()

    def __len__(self):
        return len(self.samples)

    def hierarchy_hot(self, sample):
        i_list = [
            self.hierarchy.categories().index(c)
            for c in self.hierarchy.parents(Path(sample).parent.name)
        ]
        v = torch.zeros(size=(len(self.hierarchy.categories()),)).long()
        v[i_list] = 1
        return v

    def __getitem__(self, idx):
        category = Path(self.samples[idx]).parent.name
        image = io.imread(self.samples[idx])
        label = self.hierarchy_hot(self.samples[idx]).float()

        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)

        if self.augmentation:
            image = self.aug.augmentate_sample(image)

        image = torch.from_numpy(image / 255.0).to(torch.float32).permute(2, 0, 1)

        return {
            "sample": self.samples[idx],
            "category": category,
            "inputs": image,
            "target": label,
        }


if __name__ == "__main__":
    import json

    dataset = HierarchyDataset(
        "/media/ap/Transcend/Projects/diploma/_local/diploma/assets/data/tmp/openimages/samples",
        sorted(
            [
                "/".join(p.parts[-2:])
                for p in Path(
                    "/media/ap/Transcend/Projects/diploma/_local/diploma/assets/data/tmp/openimages/samples"
                ).rglob("*.jpg")
            ]
        ),
        True,
        [224, 224],
        json.loads(
            Path(
                "/media/ap/Transcend/Projects/diploma/_local/diploma/assets/data/tmp/openimages/categories.json"
            ).read_text()
        ),
        Hierarchy.from_json(
            "/media/ap/Transcend/Projects/diploma/_local/diploma/assets/data/tmp/openimages/hierarchy.json"
        ),
    )
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample)
        break
