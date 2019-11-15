import os
import gc
import time
import pydicom
import pandas as pd
import numpy as np
from collections import OrderedDict
from contextlib import contextmanager
import torch
from torch.utils.data import DataLoader

import sys

sys.path.append("../src")
from utils import seed_torch
from logger import setup_logger, LOGGER


import os
import cv2
import pydicom
import numpy as np
from torch.utils.data import Dataset

import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler
pydicom.config.image_handlers = [None, gdcm_handler]


class RSNADataset(Dataset):

    def __init__(self,
                 df,
                 img_size,
                 image_path,
                 id_colname="Image",
                 img_type=".dcm",
                 ):
        self.df = df
        self.img_size = img_size
        self.image_path = image_path
        self.id_colname = id_colname
        self.img_type = img_type

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_pre = cur_idx_row[pre_cols].values.reshape(-1)
        img_post = cur_idx_row[post_cols].values.reshape(-1)

        self._create(img_pre, img_id, name="pre")
        self._create(img_post, img_id, name="post")

        return np.array([0])

    def _create(self, img_pathes, img_id, name):
        imgs = []
        break_flag = False
        for i, load_id in enumerate(img_pathes):
            if i == 0 and load_id=="UNK":
                break_flag=True
                load_id = img_id
            if load_id=="UNK" or load_id == skip_id:
                continue
            img_path = os.path.join(self.image_path, load_id + ".dcm")
            dataset = pydicom.read_file(img_path)
            window_center, window_width, intercept, slope = get_windowing(dataset)
            img = np.expand_dims(dataset.pixel_array, axis=2)
            if img.shape[0] != 512:
                img = cv2.resize(img, (512, 512))
            img = rescale_image(img, intercept, slope)
            img = window_image(img, 80, 200)
            imgs.append(img)
            if break_flag:
                break

        imgs = np.mean(imgs, axis=0) / len(imgs)
        output_path = "../input/concat3/{}_{}.jpg".format(img_id, name)
        cv2.imwrite(output_path, imgs)


def window_image(img, window_center, window_width, rescale=True):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
        img = img * 255

    return img


def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def rescale_image(img, intercept, slope):
    img = (img * slope + intercept)

    return img

# ===============
# Constants
# ===============
DATA_DIR = "../input/"
IMAGE_PATH = "../input/stage_2_test_images/"
LOGGER_PATH = "log.txt"
TRAIN_PATH = os.path.join(DATA_DIR, "test_concat5_st2.csv")
ID_COLUMNS = "Image"
TARGET_COLUMNS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
N_CLASSES = 6

# ===============
# Settings
# ===============
SEED = np.random.randint(100000)
device = "cuda"
img_size = 512
upper_bs = 1
batch_size = 32 * 8
epochs = 5
EXP_ID = "exp24_seres"
#model_path = "../exp/models/{}_ep{}.pth".format(EXP_ID, 4)
model_path = None
skip_id = "ID_6431af929"
pre_cols = ["pre_SOPInstanceUID", "prepre_SOPInstanceUID", "pre3_SOPInstanceUID"]
post_cols = ["post_SOPInstanceUID", "postpost_SOPInstanceUID", "post3_SOPInstanceUID"]

setup_logger(out_file=LOGGER_PATH)
seed_torch(SEED)
LOGGER.info("seed={}".format(SEED))


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def main():
    with timer('load data'):
        df = pd.read_csv(TRAIN_PATH)
        df = df[df.Image != skip_id].reset_index(drop=True)
        df = df[["Image"]+pre_cols+post_cols].fillna("UNK")
        gc.collect()

    with timer('make image'):
        train_dataset = RSNADataset(df, img_size, IMAGE_PATH, id_colname=ID_COLUMNS)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        del df, train_dataset
        gc.collect()

        for step, notuse in enumerate(train_loader):
            if step % 100 == 0:
                LOGGER.info("done step {}".format(step))


if __name__ == '__main__':
    main()
