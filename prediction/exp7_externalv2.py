# ===============
# SeResNext
# ===============
import os
import gc
import sys
import time
import glob

import pandas as pd
import numpy as np
from contextlib import contextmanager
from albumentations import *
import torch
from torch.utils.data import DataLoader

sys.path.append("../src")
from utils import seed_torch
from model import CnnModel
from datasets import RSNADatasetTest
from logger import setup_logger, LOGGER
from trainer import predict_external


# ===============
# Constants
# ===============
DATA_DIR = "../input/"
IMAGE_PATH = ""
LOGGER_PATH = "log.txt"
ID_COLUMNS = "Image"
TARGET_COLUMNS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
N_CLASSES = 6

# ===============
# Settings
# ===============
SEED = np.random.randint(100000)
device = "cuda"
img_size = 512
batch_size = 128
epochs = 5
N_TTA = 2
EXP_ID = "exp7_seresnext"
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
        path = glob.glob("../input_ext/*/*/*/*.dcm")
        df = pd.DataFrame({
            "Image": path
        })
        df = df[["Image"]]
        ids = df["Image"].values
        gc.collect()

    with timer('preprocessing'):
        test_augmentation = Compose([
            CenterCrop(512 - 50, 512 - 50, p=1.0),
            Resize(img_size, img_size, p=1)
        ])

        test_dataset = RSNADatasetTest(df, img_size, IMAGE_PATH, id_colname=ID_COLUMNS,
                                       transforms=test_augmentation, black_crop=False, subdural_window=True, n_tta=N_TTA,
                                       img_type="", external=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        del df, test_dataset
        gc.collect()

    with timer('create model'):
        model = CnnModel(num_classes=N_CLASSES, encoder="se_resnext50_32x4d", pretrained="imagenet", pool_type="avg")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model = torch.nn.DataParallel(model)

    with timer('predict'):
        pred, is_dicoms = predict_external(model, test_loader, device, n_tta=N_TTA)
        pred = np.clip(pred, 1e-6, 1-1e-6)

    with timer('sub'):
        sub = pd.DataFrame(pred, columns=TARGET_COLUMNS)
        sub["is_dicom"] = is_dicoms.reshape(-1)
        sub["Image"] = ids.reshape(-1)
        LOGGER.info(sub.head())
        sub.to_csv("../input_ext/{}_externalv2.csv".format(EXP_ID), index=False)


if __name__ == '__main__':
    main()
