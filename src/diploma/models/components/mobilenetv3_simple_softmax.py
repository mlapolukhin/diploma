import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class MobileNetV3(nn.Module):
    def __init__(self, num_classes: int, categories_json: str):
        super().__init__()
        self.num_classes = num_classes
        self.categories = json.loads(Path(categories_json).read_text())

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

        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, x, return_category=True):
        output = dict()

        output["logits"] = self.activation(self.model(self.transform(x["inputs"])))
        if return_category:
            output["category"] = self.logits_to_category(output["logits"])

        return output

    def logits_to_category(self, logits):
        return [self.categories[y.item()] for y in logits.argmax(1)]


if __name__ == "__main__":
    model = MobileNetV3(
        num_classes=17,
        categories_json="/media/ap/Transcend/Projects/diploma/_local/diploma/assets/data/tmp/openimages/categories.json",
    )
    print(model)
    x = torch.randn((1, 3, 224, 224))
    o = model(x)
    print(o)
