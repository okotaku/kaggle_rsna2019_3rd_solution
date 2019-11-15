import os

import cv2
import torch
import pydicom
import numpy as np
from albumentations import *
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from logger import LOGGER


class RSNADataset(Dataset):

    def __init__(self,
                 df,
                 y,
                 img_size,
                 image_path,
                 crop_rate = 1.0,
                 id_colname="Image",
                 img_type=".dcm",
                 transforms=None,
                 means=[0.485, 0.456, 0.406],
                 stds=[0.229, 0.224, 0.225],
                 black_crop=False,
                 subdural_window=False,
                 three_window=False,
                 rescaling=False,
                 flip_aug=False,
                 user_window=1,
                 pick_type="pre_post"
                 ):
        self.df = df
        self.y = y
        self.img_size = img_size
        self.image_path = image_path
        self.transforms = transforms
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.id_colname = id_colname
        self.img_type = img_type
        self.crop_rate = crop_rate
        self.black_crop = black_crop
        self.subdural_window = subdural_window
        self.three_window = three_window
        self.flip_aug = flip_aug
        self.rescaling = rescaling
        self.user_window = user_window
        self.pick_type = pick_type

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        if self.pick_type == "pre_post":
            img_id_pre = cur_idx_row[["pre_SOPInstanceUID"]].fillna(img_id).values[0]
            img_id_post = cur_idx_row[["post_SOPInstanceUID"]].fillna(img_id).values[0]
        elif self.pick_type == "pre_pre":
            img_id_pre = cur_idx_row[["pre_SOPInstanceUID"]].fillna(img_id).values[0]
            img_id_post = cur_idx_row[["prepre_SOPInstanceUID"]].fillna(img_id_pre).values[0]
        elif self.pick_type == "post_post":
            img_id_pre = cur_idx_row[["post_SOPInstanceUID"]].fillna(img_id).values[0]
            img_id_post = cur_idx_row[["postpost_SOPInstanceUID"]].fillna(img_id_pre).values[0]
        if self.user_window == 1:
            img = self._get_img(img_id, 1)
            img_pre = self._get_img(img_id_pre, 2)
            img_post = self._get_img(img_id_post, 3)
        elif self.user_window == 2:
            img_id_prepre = cur_idx_row[["prepre_SOPInstanceUID"]].fillna(img_id_pre).values[0]
            img_id_postpost = cur_idx_row[["postpost_SOPInstanceUID"]].fillna(img_id_post).values[0]
            img = self._get_img(img_id, 1)
            img_pre = self._get_img(img_id_prepre, 2)
            img_post = self._get_img(img_id_postpost, 3)
        target = self.y[idx]

        img = np.concatenate([img, img_pre, img_post], axis=2)

        if self.flip_aug and np.random.rand() >= 0.5:
            img = img[:, ::-1, :]
            target = np.pad(target, [0, 6], 'constant')
        elif self.flip_aug:
            target = np.pad(target, [6, 0], 'constant')

        if self.transforms is not None:
            augmented = self.transforms(image=img)
            img = augmented['image']

        img = img / 255
        img -= self.means
        img /= self.stds
        img = img.transpose((2, 0, 1))

        return torch.FloatTensor(img), torch.FloatTensor(target)

    def _get_img(self, img_id, n):
        img_path = os.path.join(self.image_path, img_id + self.img_type)
        dataset = pydicom.read_file(img_path)
        image = dataset.pixel_array

        if image.shape[0] != 512:
            image = cv2.resize(image, (512, 512))

        if self.black_crop:
            try:
                mask_img = image > np.mean(image)
                sum_channel = np.sum(mask_img, 2)
                w_cr = np.where(sum_channel.sum(0) != 0)
                h_cr = np.where(sum_channel.sum(1) != 0)
            except:
                print("pass black crop {}".format(img_id))

        if self.subdural_window:
            window_center, window_width, intercept, slope = get_windowing(dataset)
            image = rescale_image(image, intercept, slope)
            image = window_image(image, 80, 200)
        elif self.three_window:
            window_center, window_width, intercept, slope = get_windowing(dataset)
            image = rescale_image(image, intercept, slope)
            if n == 1:
                image = window_image(image, 80, 200, self.rescaling)
                if not self.rescaling:
                    image = (image - (-20)) / 200
            elif n == 2:
                image = window_image(image, 40, 80, self.rescaling)
                if not self.rescaling:
                    image = (image - 0) / 80
            elif n == 3:
                image = window_image(image, 40, 300, self.rescaling)
                if not self.rescaling:
                    image = (image - (-150)) / 380

        if not self.subdural_window and not self.three_window:
            min_ = image.min()
            max_ = image.max()
            image = (image - min_) / (max_ - min_)
            image = image * 255

        if self.black_crop:
            try:
                image = image[np.min(h_cr):np.max(h_cr) + 1, np.min(w_cr):np.max(w_cr) + 1, :]
            except:
                print("pass black crop {}".format(img_id))

        image = np.expand_dims(image, axis=2)

        return image


class RSNADatasetTest(Dataset):

    def __init__(self,
                 df,
                 img_size,
                 image_path,
                 crop_rate = 1.0,
                 id_colname="Image",
                 img_type=".dcm",
                 transforms=None,
                 means=[0.485, 0.456, 0.406],
                 stds=[0.229, 0.224, 0.225],
                 black_crop=False,
                 subdural_window=False,
                 three_window=False,
                 n_tta=1,
                 rescaling=False,
                 user_window=1,
                 pick_type="pre_post"
                 ):
        self.df = df
        self.img_size = img_size
        self.image_path = image_path
        self.transforms = transforms
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.id_colname = id_colname
        self.img_type = img_type
        self.crop_rate = crop_rate
        self.black_crop = black_crop
        self.subdural_window = subdural_window
        self.three_window = three_window
        self.n_tta = n_tta
        self.transforms2 = Compose([
            #CenterCrop(512 - 50, 512 - 50, p=1.0),
            Resize(img_size, img_size, p=1)
        ])
        self.rescaling = rescaling
        self.user_window = user_window
        self.pick_type = pick_type

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        if self.pick_type == "pre_post":
            img_id_pre = cur_idx_row[["pre_SOPInstanceUID"]].fillna(img_id).values[0]
            img_id_post = cur_idx_row[["post_SOPInstanceUID"]].fillna(img_id).values[0]
        elif self.pick_type == "pre_pre":
            img_id_pre = cur_idx_row[["pre_SOPInstanceUID"]].fillna(img_id).values[0]
            img_id_post = cur_idx_row[["prepre_SOPInstanceUID"]].fillna(img_id_pre).values[0]
        elif self.pick_type == "post_post":
            img_id_pre = cur_idx_row[["post_SOPInstanceUID"]].fillna(img_id).values[0]
            img_id_post = cur_idx_row[["postpost_SOPInstanceUID"]].fillna(img_id_pre).values[0]
        if self.user_window == 1:
            img = self._get_img(img_id, 1)
            img_pre = self._get_img(img_id_pre, 2)
            img_post = self._get_img(img_id_post, 3)
        elif self.user_window == 2:
            img_id_prepre = cur_idx_row[["prepre_SOPInstanceUID"]].fillna(img_id_pre).values[0]
            img_id_postpost = cur_idx_row[["postpost_SOPInstanceUID"]].fillna(img_id_post).values[0]
            img = self._get_img(img_id, 1)
            img_pre = self._get_img(img_id_prepre, 2)
            img_post = self._get_img(img_id_postpost, 3)

        img = np.concatenate([img, img_pre, img_post], axis=2)

        if self.transforms is not None:
            augmented = self.transforms2(image=img)
            img_tta = augmented['image']
            augmented = self.transforms(image=img)
            img = augmented['image']

        imgs = []
        img = img / 255
        img -= self.means
        img /= self.stds
        img = img.transpose((2, 0, 1))
        imgs.append(torch.FloatTensor(img))
        if self.n_tta >= 2:
            flip_img = img[:, :, ::-1].copy()
            imgs.append(torch.FloatTensor(flip_img))

        if self.n_tta >= 4:
            img_tta = img_tta / 255
            img_tta -= self.means
            img_tta /= self.stds
            img_tta = img_tta.transpose((2, 0, 1))
            imgs.append(torch.FloatTensor(img_tta))
            flip_img_tta = img_tta[:, :, ::-1].copy()
            imgs.append(torch.FloatTensor(flip_img_tta))

        return imgs

    def _get_img(self, img_id, n):
        img_path = os.path.join(self.image_path, img_id + self.img_type)
        dataset = pydicom.read_file(img_path)
        image = dataset.pixel_array

        if image.shape[0] != 512:
            image = cv2.resize(image, (512, 512))

        if self.black_crop:
            try:
                mask_img = image > np.mean(image)
                sum_channel = np.sum(mask_img, 2)
                w_cr = np.where(sum_channel.sum(0) != 0)
                h_cr = np.where(sum_channel.sum(1) != 0)
            except:
                print("pass black crop {}".format(img_id))

        if self.subdural_window:
            window_center, window_width, intercept, slope = get_windowing(dataset)
            image = rescale_image(image, intercept, slope)
            image = window_image(image, 80, 200)
        elif self.three_window:
            window_center, window_width, intercept, slope = get_windowing(dataset)
            image = rescale_image(image, intercept, slope)
            if n == 1:
                image = window_image(image, 80, 200, self.rescaling)
                if not self.rescaling:
                    image = (image - (-20)) / 200
            elif n == 2:
                image = window_image(image, 40, 80, self.rescaling)
                if not self.rescaling:
                    image = (image - 0) / 80
            elif n == 3:
                image = window_image(image, 40, 300, self.rescaling)
                if not self.rescaling:
                    image = (image - (-150)) / 380

        if self.black_crop:
            try:
                image = image[np.min(h_cr):np.max(h_cr) + 1, np.min(w_cr):np.max(w_cr) + 1, :]
            except:
                print("pass black crop {}".format(img_id))

        if not self.subdural_window and not self.three_window:
            min_ = image.min()
            max_ = image.max()
            image = (image - min_) / (max_ - min_)
            image = image * 255

        image = np.expand_dims(image, axis=2)

        return image


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


class EvenSampler(Sampler):
    def __init__(self, train_df, demand_non_empty_proba):
        assert demand_non_empty_proba > 0, 'frequensy of non-empty images must be greater then zero'
        self.positive_proba = demand_non_empty_proba

        self.train_df = train_df.reset_index(drop=True)

        self.positive_idxs = self.train_df[self.train_df.sum_target != 0].index.values
        self.negative_idxs = self.train_df[self.train_df.sum_target == 0].index.values

        self.n_positive = self.positive_idxs.shape[0]
        self.n_negative = int(self.n_positive * (1 - self.positive_proba) / self.positive_proba)
        LOGGER.info("len data = {}".format(self.n_positive + self.n_negative))

    def __iter__(self):
        negative_sample = np.random.choice(self.negative_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.positive_idxs)))
        return iter(shuffled.tolist())

    def __len__(self):
        return self.n_positive + self.n_negative
