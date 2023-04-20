import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from diploma.data.hierarchy import Hierarchy


class MobileNetV3(nn.Module):
    def __init__(self, num_classes: int, hierarchy_json: str):
        super().__init__()
        self.num_classes = num_classes
        self.hierarchy = Hierarchy.from_json(hierarchy_json)

        self.int_to_hierarchy_category = dict(enumerate(self.hierarchy.categories()))
        self.hierarchy_category_to_int = dict(
            map(lambda x: (x[1], x[0]), enumerate(self.hierarchy.categories()))
        )

        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True),
        )
        self.model.requires_grad_(False)
        self.model.classifier.requires_grad_(True)

        self.activation = torch.nn.Sigmoid()

    def forward(self, x, return_category=True):
        output = dict()

        output["logits"] = self.activation(self.model(self.transform(x["inputs"])))
        if return_category:
            output["category"] = self.logits_to_category(output["logits"])

        return output

    def logits_to_category(self, logits):
        category = []

        for sample_i in range(len(logits)):
            hierarchy = self.hierarchy.dct.copy()

            probs = {
                class_str: logits[sample_i, class_i]
                for class_i, class_str in self.int_to_hierarchy_category.items()
            }

            key = "Entity"
            while True:
                hierarchy = hierarchy[key]
                if hierarchy is None:
                    break
                else:
                    key = max(list(hierarchy.keys()), key=lambda _key: probs[_key])

            category.append(key)

        return category


if __name__ == "__main__":
    model = MobileNetV3(
        num_classes=17,
        hierarchy_json="/media/ap/Transcend/Projects/diploma/_local/diploma/assets/data/tmp/openimages/hierarchy.json",
    )
    print(model)
    x = torch.randn((1, 3, 224, 224))
    o = model(x)
    print(o)
