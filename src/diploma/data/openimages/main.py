import urllib.request
from pathlib import Path
import pandas as pd
from tqdm import tqdm


ROOT = Path("/media/ap/Transcend/Projects/diploma/_local/diploma/assets/data")

# RAW CONFIG FILES
class_descriptions = ROOT / "raw/openimages/oidv6-class-descriptions.csv"
hierarchy = ROOT / "raw/openimages/bbox_labels_600_hierarchy.json"
trainable_list = ROOT / "raw/openimages/oidv6-classes-trainable.txt"
train_bboxes = ROOT / "raw/openimages/oidv6-train-annotations-bbox.csv"
train_images_list = (
    ROOT / "raw/openimages/oidv6-train-annotations-human-imagelabels.csv"
)
test_bboxes = ROOT / "raw/openimages/test-annotations-bbox.csv"
test_images_list = ROOT / "raw/openimages/test-annotations-human-imagelabels.csv"
val_bboxes = ROOT / "raw/openimages/validation-annotations-bbox.csv"
val_images_list = ROOT / "raw/openimages/validation-annotations-human-imagelabels.csv"
# TEMPORARY RESULTS
tmp_dir = ROOT / "tmp/openimages/"
samples_dir = ROOT / "tmp/openimages/samples/"
images_dir = ROOT / "tmp/openimages/images/"
hierarchy_json = ROOT / "tmp/openimages/hierarchy.json"
categories_json = ROOT / "tmp/openimages/categories.json"
downloaded_images = ROOT / "tmp/openimages/downloads/images.txt"
# FINAL DATASET
final_dir = ROOT / "final/openimages/"


def download_configs():
    urls = [
        "https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json",
        "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv",
        "https://storage.googleapis.com/openimages/v6/oidv6-classes-trainable.txt",
        "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv",
        "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv",
        "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv",
    ]

    for url in tqdm(urls):
        file_name = url.split("/")[-1]
        save_path = ROOT / "raw" / "openimages" / file_name
        save_path.parent.mkdir(exist_ok=True, parents=True)
        urllib.request.urlretrieve(url, save_path)
        print(f"{file_name} has been downloaded to {save_path}")


categories = [
    "Rocket",
    "Airplane",
    "Helicopter",
    "Bottle",
    "Camera",
    "Chicken",
    "Eagle",
    "Parrot",
    "Cat",
    "Dog",
    "Tortoise",
]


def count_boxes():
    category_to_code = dict(
        pd.read_csv(class_descriptions)[["DisplayName", "LabelName"]].values.tolist()
    )
    code_to_category = dict(
        pd.read_csv(class_descriptions)[["LabelName", "DisplayName"]].values.tolist()
    )

    columns = [
        "ImageID",
        "LabelName",
        "Confidence",
        "XMin",
        "XMax",
        "YMin",
        "YMax",
        "IsOccluded",
        "IsTruncated",
        "IsGroupOf",
        "IsDepiction",
        "IsInside",
    ]

    codes = [category_to_code[category] for category in categories]
    query = (
        "(LabelName in @codes) &"
        "(Confidence == 1) &"
        "(IsOccluded == 0) &"
        "(IsTruncated == 0) &"
        "(IsGroupOf == 0) &"
        "(IsDepiction == 0) &"
        "(IsInside == 0)"
    )
    df_boxes_train = pd.read_csv(train_bboxes)[columns].query(query)
    df_boxes_train["Split"] = "train"
    df_boxes_val = pd.read_csv(val_bboxes)[columns].query(query)
    df_boxes_val["Split"] = "validation"
    df_boxes_test = pd.read_csv(test_bboxes)[columns].query(query)
    df_boxes_test["Split"] = "test"

    df_boxes = pd.concat([df_boxes_train, df_boxes_val, df_boxes_test])
    df_boxes["DisplayName"] = df_boxes["LabelName"].apply(
        lambda label_name: code_to_category[label_name]
    )
    df_boxes["SampleID"] = ["%09d" % i for i in range(df_boxes.shape[0])]

    tmp_dir.mkdir(exist_ok=True, parents=True)
    df_boxes.to_csv(tmp_dir / "df_boxes.csv", index=False)

    df_count = df_boxes.value_counts("DisplayName").to_frame()
    # categories with zero examples
    for category in set(categories).difference(set(df_count.index.values)):
        df_count.loc[category] = 0

    tmp_dir.mkdir(exist_ok=True, parents=True)
    df_count.to_html(tmp_dir / "count.html")


categories = [
    "Rocket",
    "Airplane",
    "Helicopter",
    "Bottle",
    "Camera",
    "Chicken",
    "Eagle",
    "Parrot",
    "Cat",
    "Dog",
    "Tortoise",
]
samples_per_category = 400


def select_samples():
    df_boxes = pd.read_csv(tmp_dir / "df_boxes.csv").query("DisplayName in @categories")
    df_boxes = pd.concat(
        [
            df_boxes.query("DisplayName == @category").sample(
                samples_per_category, replace=False
            )
            for category in categories
        ]
    )
    df_boxes.to_csv(tmp_dir / "df_boxes_samples.csv", index=False)


num_processes = 8


def download_images():
    df_boxes = pd.read_csv(tmp_dir / "df_boxes_samples.csv")

    from diploma.data.openimages.downloader import download_images

    image_list = (
        df_boxes[["Split", "ImageID"]].drop_duplicates("ImageID").values.tolist()
    )
    download_images(
        image_list=image_list,
        num_processes=num_processes,
        download_folder=images_dir,
    )


from PIL import Image
import numpy as np
from concurrent import futures

num_workers = 8


def make_samples():
    downloaded_images = [p.stem for p in Path(images_dir).rglob("*")]
    df_boxes = pd.read_csv(tmp_dir / "df_boxes_samples.csv").query(
        "ImageID in @downloaded_images"
    )

    def make_sample(row):
        sampleid, imageid, code, category, xmin, xmax, ymin, ymax = row[
            [
                "SampleID",
                "ImageID",
                "LabelName",
                "DisplayName",
                "XMin",
                "XMax",
                "YMin",
                "YMax",
            ]
        ]
        image_path = images_dir / (imageid + ".jpg")
        image = Image.open(image_path)
        W, H = image.size
        xmin = round(W * xmin)
        xmax = round(W * xmax)
        ymin = round(H * ymin)
        ymax = round(H * ymax)
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        bbox_size = max(bbox_width, bbox_height)
        center_x = (xmin + xmax) // 2
        center_y = (ymin + ymax) // 2
        new_xmin = max(center_x - bbox_size // 2, 0)
        new_ymin = max(center_y - bbox_size // 2, 0)
        new_xmax = min(new_xmin + bbox_size, W)
        new_ymax = min(new_ymin + bbox_size, H)
        pad_left = max(0 - new_xmin, 0)
        pad_right = max(new_xmax - W, 0)
        pad_top = max(0 - new_ymin, 0)
        pad_bottom = max(new_ymax - H, 0)
        new_xmin += pad_left
        new_ymin += pad_top
        new_xmax += pad_left
        new_ymax += pad_top
        sample_path = (
            samples_dir
            / category
            / f"{imageid}_{sampleid}_{new_ymin}_{new_ymax}_{new_xmin}_{new_xmax}.jpg"
        )
        if not sample_path.exists():
            sample = Image.fromarray(
                np.asarray(image.convert("RGB"))[new_ymin:new_ymax, new_xmin:new_xmax]
            )
            sample_path.parent.mkdir(exist_ok=True, parents=True)
            sample.save(sample_path, quality=100)

    progress_bar = tqdm(total=df_boxes.shape[0], desc=f"Processing samples", leave=True)
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        all_futures = [
            executor.submit(make_sample, row) for (i, row) in df_boxes.iterrows()
        ]
        for future in futures.as_completed(all_futures):
            future.result()
            progress_bar.update(1)
    progress_bar.close()


split_proportions = {
    "train": 0.7,
    "val": 0.3,
}
import json


def make_splits():
    samples = ["/".join(p.parts[-2:]) for p in Path(samples_dir).rglob("*.jpg")]
    splitnames, splitprobs = map(list, zip(*split_proportions.items()))
    splits = [np.random.choice(splitnames, p=splitprobs) for _ in range(len(samples))]
    splits = {
        group_name: group_values["Sample"].tolist()
        for group_name, group_values in pd.DataFrame(
            data={"Sample": samples, "Split": splits}
        ).groupby("Split")
    }

    (tmp_dir / "splits.json").write_text(json.dumps(splits))


import json

hierarchy = {
    "Entity": {
        "Animal": {
            "Mammal": {"Cat": None, "Dog": None, "Tortoise": None},
            "Bird": {"Chicken": None, "Parrot": None, "Eagle": None},
        },
        "Tool": {"Bottle": None, "Camera": None},
        "Aircraft": {"Helicopter": None, "Airplane": None, "Rocket": None},
    }
}


def make_hierarchy():
    hierarchy_json.write_text(json.dumps(hierarchy, indent=4, sort_keys=True))


def make_categories():
    categories_json.write_text(json.dumps(categories, indent=4, sort_keys=True))


if __name__ == "__main__":
    pass

    download_configs()
    count_boxes()
    select_samples()
    download_images()
    make_samples()
    make_splits()
    make_hierarchy()
    make_categories()
