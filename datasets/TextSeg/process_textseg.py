
import json
import os
import shutil

os.makedirs("train_images", exist_ok=True)
os.makedirs("train_gt", exist_ok=True)
os.makedirs("val_images", exist_ok=True)
os.makedirs("val_gt", exist_ok=True)
os.makedirs("test_images", exist_ok=True)
os.makedirs("test_gt", exist_ok=True)

with open("split.json") as f_json:
    split_info = json.load(f_json)
train_img_list = split_info["train"]
val_img_list = split_info["val"]
test_img_list = split_info["test"]

for im_id in train_img_list:
    shutil.copy(
        os.path.join("image", im_id + ".jpg"),
        os.path.join("train_images", im_id + ".jpg"),
    )
    shutil.copy(
        os.path.join("semantic_label", im_id + "_maskfg.png"),
        os.path.join("train_gt", im_id + ".png"),
    )
for im_id in val_img_list:
    shutil.copy(
        os.path.join("image", im_id + ".jpg"),
        os.path.join("val_images", im_id + ".jpg"),
    )
    shutil.copy(
        os.path.join("semantic_label", im_id + "_maskfg.png"),
        os.path.join("val_gt", im_id + ".png"),
    )
for im_id in test_img_list:
    shutil.copy(
        os.path.join("image", im_id + ".jpg"),
        os.path.join("test_images", im_id + ".jpg"),
    )
    shutil.copy(
        os.path.join("semantic_label", im_id + "_maskfg.png"),
        os.path.join("test_gt", im_id + ".png"),
    )
