import torch
import torch.nn as nn
import torch.nn.functional as F

from diploma.data.hierarchy import Hierarchy


class BCECriterion(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.loss = nn.BCELoss(*args, **kwargs)

    def forward(self, output, batch, *args, **kwargs):
        bce = self.loss(output["logits"], batch["target"])
        return {
            "total_loss": bce,
            "losses": {
                "bce": bce,
            },
        }


class HierarchyCriterion(nn.Module):
    def __init__(
        self,
        hierarchy_json: str,
        L_bce_coef: float,
        L_sim_coef: float,
        L_cat_coef: float,
        L_seq_coef: float,
        L_lvl_coef: float,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hierarchy = Hierarchy.from_json(hierarchy_json)
        self.L_bce_coef = L_bce_coef
        self.L_sim_coef = L_sim_coef
        self.L_cat_coef = L_cat_coef
        self.L_seq_coef = L_seq_coef
        self.L_lvl_coef = L_lvl_coef

    def forward(self, output, batch, *args, **kwargs):
        logits = output["logits"]
        target = batch["target"]
        category = batch["category"]

        L_list = []

        for sample_i in range(len(logits)):
            order_c = self.hierarchy.parents(category[sample_i])
            order_i = [self.hierarchy.categories().index(c) for c in order_c]

            L_bce = F.binary_cross_entropy(
                logits[sample_i, ...], target[sample_i, ...], reduction="mean"
            )

            L_sim = torch.var(logits[sample_i, order_i])

            L_cat = F.binary_cross_entropy(
                logits[sample_i, order_i[0]], target[sample_i, order_i[0]]
            )

            L_seq = 1
            for i in order_i[::-1]:
                L_seq = (
                    1 - F.l1_loss(logits[sample_i, i], target[sample_i, i])
                ) * L_seq
                L_seq = L_seq**2
            L_seq = (1 - L_seq) * F.binary_cross_entropy(
                logits[sample_i, ...], target[sample_i, ...], reduction="mean"
            )

            L_lvl_list = []
            for parent, children_s in self.hierarchy._children.items():
                if len(children_s) == 0:
                    continue
                children_i = [self.hierarchy.categories().index(s) for s in children_s]
                L_lvl = F.binary_cross_entropy(
                    logits[sample_i, children_i],
                    target[sample_i, children_i],
                    reduction="mean",
                )
                L_lvl_list.append(L_lvl)
            L_lvl = torch.stack(L_lvl_list).mean()

            L = (
                L_bce * self.L_bce_coef
                + L_sim * self.L_sim_coef
                + L_cat * self.L_cat_coef
                + L_seq * self.L_seq_coef
                + L_lvl * self.L_lvl_coef
            )
            L_list.append(L)

        return {
            "total_loss": torch.stack(L_list).mean(),
            "losses": {
                "L_bce": L_bce,
                "L_sim": L_sim,
                "L_cat": L_cat,
                "L_seq": L_seq,
                "L_lvl": L_lvl,
            },
        }


if __name__ == "__main__":
    loss = HierarchyCriterion(
        "/media/ap/Transcend/Projects/diploma/_local/diploma/assets/data/tmp/openimages/hierarchy.json",
        L_bce_coef=0,
        L_sim_coef=0,
        L_cat_coef=0,
        L_seq_coef=0,
        L_lvl_coef=1,
    )

    print(loss.hierarchy.categories())

    output = {
        "logits": torch.Tensor(
            [
                [
                    1.0,
                    0.0,
                    0.9,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            ]
        ),  # Tortoise
    }
    batch = {
        "target": torch.Tensor(
            [
                [
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            ]
        ),  # Tortoise
        "category": ["Tortoise"],
    }
    l = loss(output, batch)
    print(
        "*" * 10,
        "loss",
        l,
        "*" * 10,
    )

    output = {
        "logits": torch.Tensor(
            [
                [
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.9,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            ]
        ),  # Tortoise
    }
    batch = {
        "target": torch.Tensor(
            [
                [
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            ]
        ),  # Tortoise
        "category": ["Tortoise"],
    }
    l = loss(output, batch)
    print(
        "*" * 10,
        "loss",
        l,
        "*" * 10,
    )

    output = {
        "logits": torch.Tensor(
            [
                [
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.9,
                ]
            ]
        ),  # Tortoise
    }
    batch = {
        "target": torch.Tensor(
            [
                [
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            ]
        ),  # Tortoise
        "category": ["Tortoise"],
    }
    l = loss(output, batch)
    print(
        "*" * 10,
        "loss",
        l,
        "*" * 10,
    )

    while True:
        output = {
            "logits": torch.rand((1, 17)),  # Tortoise
        }
        batch = {
            "target": torch.randint(0, 2, (1, 17)).float(),  # Tortoise
            "category": ["Tortoise"],
        }
        l = loss(output, batch)
        print(
            "*" * 10,
            "loss",
            l,
            "*" * 10,
        )
