# ===============
# SeResNext
# ===============
import os
import gc
import sys
import time

import pandas as pd
import numpy as np
from contextlib import contextmanager
from albumentations import *
import torch
from torch.utils.data import DataLoader

sys.path.append("../src")
from utils import seed_torch
from model import CnnModel, Efficient
from dataset_concat import RSNADataset
from logger import setup_logger, LOGGER
from trainer import train_one_epoch


# ===============
# Constants
# ===============
DATA_DIR = "../input/"
IMAGE_PATH = "../input/stage_2_train_images/"
LOGGER_PATH = "log.txt"
TRAIN_PATH = os.path.join(DATA_DIR, "train_concat.csv")
ID_COLUMNS = "Image"
TARGET_COLUMNS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
N_CLASSES = 6

# ===============
# Settings
# ===============
SEED = np.random.randint(100000)
device = "cuda"
img_size = 512
batch_size = 32
epochs = 5
EXP_ID = "exp18_seres"
model_path = None

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
        df = pd.read_csv(TRAIN_PATH)
        df = df[df.Image != "ID_6431af929"].reset_index(drop=True)
        df.loc[df.pre_SOPInstanceUID=="ID_6431af929", "pre1_SOPInstanceUID"] = df.loc[
            df.pre_SOPInstanceUID=="ID_6431af929", "Image"]
        df.loc[df.post_SOPInstanceUID == "ID_6431af929", "post1_SOPInstanceUID"] = df.loc[
            df.post_SOPInstanceUID == "ID_6431af929", "Image"]
        y = df[TARGET_COLUMNS].values
        df = df[["Image", "pre1_SOPInstanceUID", "post1_SOPInstanceUID"]]
        gc.collect()

    with timer('preprocessing'):
        train_augmentation = Compose([
            CenterCrop(512 - 50, 512 - 50, p=1.0),
            HorizontalFlip(p=0.5),
            OneOf([
                ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                GridDistortion(p=0.5),
                OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            ], p=0.5),
            Rotate(limit=30, border_mode=0, p=0.7),
            Resize(img_size, img_size, p=1)
        ])

        train_dataset = RSNADataset(df, y, img_size, IMAGE_PATH, id_colname=ID_COLUMNS,
                                    transforms=train_augmentation, black_crop=False, three_window=True, rescaling=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        del df, train_dataset
        gc.collect()

    with timer('create model'):
        model = CnnModel(num_classes=N_CLASSES, encoder="se_resnext50_32x4d", pretrained="imagenet", pool_type="avg")
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([2, 1, 1, 1, 1, 1]).cuda())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-4)
        model = torch.nn.DataParallel(model)

    with timer('train'):
        for epoch in range(1, epochs + 1):
            if epoch == 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
            seed_torch(SEED + epoch)

            LOGGER.info("Starting {} epoch...".format(epoch))
            tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            LOGGER.info('Mean train loss: {}'.format(round(tr_loss, 5)))

            torch.save(model.module.state_dict(), 'models/{}_ep{}.pth'.format(EXP_ID, epoch))


if __name__ == '__main__':
    main()
