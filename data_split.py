import os
import shutil
from shutil import copy
from random import shuffle


def remove_ds_store(list):
    ans = []
    for x in list:
        if x != '.DS_Store':
            ans.append(x)
    return ans


train_fraction = 0.8
valid_fraction = 0.1
test_fraction = 0.1
assert train_fraction + valid_fraction + test_fraction == 1, "incorrect fractions"


CWD = os.getcwd()
print(CWD)

n_classes = 102
folder_name = "multi_model_data"
if os.path.isdir(folder_name):
    print("delete previous multi model data...")
    shutil.rmtree(os.path.join(CWD, folder_name))
os.makedirs(folder_name)
os.chdir(os.path.join(CWD, folder_name))
os.makedirs("train")
os.makedirs("valid")
os.makedirs("test")
for part in ["train", "valid", "test"]:
    os.chdir(os.path.join(CWD, folder_name, part))
    for cls in range(1, n_classes+1):
        os.makedirs(f"{cls}")
os.chdir(CWD)


for i in range(1, n_classes + 1):
    print(f"manage {i} class")
    train_path = os.path.join(CWD, f"flower_data/train/{i}")
    valid_path = os.path.join(CWD, f"flower_data/valid/{i}")
    train_list = [os.path.join(train_path, x) for x in remove_ds_store(os.listdir(train_path))]
    valid_list = [os.path.join(valid_path, x) for x in remove_ds_store(os.listdir(valid_path))]
    overall_list = train_list + valid_list
    shuffle(overall_list)

    train_ind = int(train_fraction * len(overall_list))
    valid_ind = train_ind + int(valid_fraction * len(overall_list))
    train_dirs = overall_list[:train_ind]
    valid_dirs = overall_list[train_ind:valid_ind]
    test_dirs = overall_list[valid_ind:]
    assert set(train_dirs + test_dirs + valid_dirs) == set(overall_list), "incorrect dirs spliting"

    new_train_path = os.path.join(CWD, folder_name, f"train/{i}")
    new_valid_path = os.path.join(CWD, folder_name, f"valid/{i}")
    new_test_path = os.path.join(CWD, folder_name, f"test/{i}")

    for dirs, new_dir in zip([train_dirs, valid_dirs, test_dirs], [new_train_path, new_valid_path, new_test_path]):
        for img_path in dirs:
            copy(img_path, new_dir)

