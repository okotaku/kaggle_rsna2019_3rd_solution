# ===============
# SeResNext
# ===============
import os
import gc
import time
import pandas as pd
import numpy as np
from collections import OrderedDict
from contextlib import contextmanager
from albumentations import *
import torch
from torch.utils.data import DataLoader

import sys

sys.path.append("../src")
from utils import seed_torch, postprocess_multitarget
from model import CnnModel, Efficient
from dataset_concat import RSNADatasetTest
from logger import setup_logger, LOGGER
from trainer import predict


# ===============
# Constants
# ===============
DATA_DIR = "../input/"
IMAGE_PATH = "../input/stage_2_test_images/"
LOGGER_PATH = "log.txt"
TEST_PATH = os.path.join(DATA_DIR, "test_concat_st2.csv")
ID_COLUMNS = "Image"
TARGET_COLUMNS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural",
                  "prepre_any", "prepre_epidural", "prepre_intraparenchymal", "prepre_intraventricular", "prepre_subarachnoid", "prepre_subdural",
                  "postpost_any", "postpost_epidural", "postpost_intraparenchymal", "postpost_intraventricular", "postpost_subarachnoid", "postpost_subdural"]
N_CLASSES = 18

# ===============
# Settings
# ===============
SEED = np.random.randint(100000)
device = "cuda"
img_size = 512
batch_size = 128
epochs = 5
N_TTA = 2
EXP_ID = "exp36_seres"
model_path = "../exp/models/{}_ep{}.pth".format(EXP_ID, epochs)

setup_logger(out_file=LOGGER_PATH)
seed_torch(SEED)
LOGGER.info("seed={}".format(SEED))


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


def main():
    with timer('load data'):
        df = pd.read_csv(TEST_PATH)
        df["post_SOPInstanceUID"] = df["post_SOPInstanceUID"].fillna(df["SOPInstanceUID"])
        df["pre_SOPInstanceUID"] = df["pre_SOPInstanceUID"].fillna(df["SOPInstanceUID"])
        df["postpost_SOPInstanceUID"] = df["postpost_SOPInstanceUID"].fillna(
            df["post_SOPInstanceUID"])
        df["prepre_SOPInstanceUID"] = df["prepre_SOPInstanceUID"].fillna(
            df["pre_SOPInstanceUID"])
        df = df[["Image", "pre_SOPInstanceUID", "post_SOPInstanceUID", "prepre_SOPInstanceUID", "postpost_SOPInstanceUID"]]
        ids = df["Image"].values
        pre_ids = df["prepre_SOPInstanceUID"].values
        pos_ids = df["postpost_SOPInstanceUID"].values
        gc.collect()

    with timer('preprocessing'):
        test_augmentation = Compose([
            CenterCrop(512 - 50, 512 - 50, p=1.0),
            Resize(img_size, img_size, p=1)
        ])

        test_dataset = RSNADatasetTest(df, img_size, IMAGE_PATH, id_colname=ID_COLUMNS,
                                    transforms=test_augmentation, black_crop=False, subdural_window=True,
                                    n_tta=N_TTA)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        del df, test_dataset
        gc.collect()

    with timer('create model'):
        model = CnnModel(num_classes=N_CLASSES, encoder="se_resnext50_32x4d", pretrained="imagenet", pool_type="avg")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model = torch.nn.DataParallel(model)

    with timer('predict'):
        pred = predict(model, test_loader, device, n_tta=N_TTA)
        pred = np.clip(pred, 1e-6, 1-1e-6)

    with timer('sub'):
        sub = pd.DataFrame(pred, columns=TARGET_COLUMNS)
        sub["ID"] = ids
        sub["PRE_ID"] = pre_ids
        sub["POST_ID"] = pos_ids
        sub = postprocess_multitarget(sub)
        LOGGER.info(sub.head())
        sub.to_csv("../output/{}_sub_st2.csv".format(EXP_ID), index=False)


if __name__ == '__main__':
    main()
