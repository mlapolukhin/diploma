from collections import defaultdict
import json
from pathlib import Path


class Hierarchy:
    def __init__(self, dct):
        self.dct = dct
        self._children = self._init_children_dict(self.dct)
        self._parent = self._init_parent_dict(self.dct)
        self._categories = self._init_categories(self.dct)

    def _init_children_dict(self, hierarchy):
        result = {}
        for key, value in hierarchy.items():
            if value is None:
                result[key] = []
            else:
                result[key] = list(value.keys())
                sub_dict = self._init_children_dict(value)
                result.update(sub_dict)
        return result

    def _init_parent_dict(self, hierarchy, parent=None):
        result = {}
        for category, subcategories in hierarchy.items():
            result[category] = parent
            if subcategories:
                sub_result = self._init_parent_dict(subcategories, parent=category)
                result.update(sub_result)
        return result

    def _init_categories(self, hierarchy):
        def flatten_helper(subtree, level):
            result = []
            for category, subcategories in sorted(subtree.items()):
                if subcategories:
                    subcategories_flat = flatten_helper(subcategories, level + 1)
                    result.extend(subcategories_flat)
                result.append((category, level))
            return result

        flat_categories = flatten_helper(hierarchy, 0)
        flat_categories_sorted = sorted(flat_categories, key=lambda x: (x[1], x[0]))
        return [category for category, _ in flat_categories_sorted]

    def children(self, x):
        return self._children.get(x, None)

    def parent(self, x):
        return self._parent.get(x, None)

    def parents(self, x):
        parents = [x]
        parent = self.parent(parents[-1])
        while parent is not None:
            parents.append(parent)
            parent = self.parent(parents[-1])
        return parents

    def categories(self):
        return self._categories

    @staticmethod
    def from_json(hierarchy_json: str):
        return Hierarchy(json.loads(Path(hierarchy_json).read_text()))


if __name__ == "__main__":
    from pathlib import Path
    import json

    hierarchy_json = "/media/ap/Transcend/Projects/diploma/assets/data/tmp/openimages/hierarchy.json"
    hierarchy = Hierarchy.from_json(hierarchy_json)

    print(hierarchy.parents("Chicken"))
    print(hierarchy.children("Bird"))
    print(hierarchy.children("Entity"))
    print(hierarchy.categories())
    print(hierarchy._children)
