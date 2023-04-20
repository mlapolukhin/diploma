from typing import List, Dict, Optional, Callable, Any

import torch
import torch.nn.functional as F
import imageio as io

import cv2
import torch.utils.data

from diploma.data.augmentation import Augmentor

import torch
import torch.nn.functional as F

from pathlib import Path


class OneHotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples_dir: str,
        samples_list: str,
        augmentation: bool,
        input_size: int,
        categories: List[str],
    ):
        self.samples_dir = samples_dir
        self.samples_list = samples_list
        self.augmentation = augmentation
        self.input_size = input_size
        self.categories = categories
        self.samples = sorted([f"{samples_dir}/{sample}" for sample in samples_list])

        if self.augmentation:
            self.aug = Augmentor()

    def __len__(self):
        return len(self.samples)

    def one_hot(self, sample):
        i = self.categories.index(Path(sample).parent.name)
        return F.one_hot(torch.Tensor([i]).long(), num_classes=len(self.categories))[0]

    def __getitem__(self, idx):
        category = Path(self.samples[idx]).parent.name
        image = io.imread(self.samples[idx])
        label = self.one_hot(self.samples[idx]).float()

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

    dataset = OneHotDataset(
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
    )
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample)
