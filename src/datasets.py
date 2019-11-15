import os

import cv2
import torch
import pydicom
import numpy as np
from albumentations import *
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from logger import LOGGER

import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler
pydicom.config.image_handlers = [None, gdcm_handler]


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
                 new_three_window_type=True,
                 rescaling=True,
                 flip_aug=False,
                 external=False
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
        self.rescaling = rescaling
        self.new_three_window_type = new_three_window_type
        self.flip_aug = flip_aug
        self.external = external

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        if self.external:
            external_flag = cur_idx_row["external_flag"]
            if external_flag == 1:
                img_path = img_id
            else:
                img_path = os.path.join(self.image_path, img_id + self.img_type)
        else:
            img_path = os.path.join(self.image_path, img_id + self.img_type)
        dataset = pydicom.read_file(img_path)
        img = np.expand_dims(dataset.pixel_array, axis=2)
        if not self.three_window:
            img = img.repeat(3, axis=2)
        target = self.y[idx]

        if img.shape[0] != 512:
            img = cv2.resize(img, (512, 512))
            print("reshape {}".format(img_id))

        if self.black_crop:
            try:
                mask_img = img > np.mean(img)
                sum_channel = np.sum(mask_img, 2)
                w_cr = np.where(sum_channel.sum(0) != 0)
                h_cr = np.where(sum_channel.sum(1) != 0)
            except:
                print("pass black crop {}".format(img_id))

        if self.three_window:
            window_center, window_width, intercept, slope = get_windowing(dataset)
            img = rescale_image(img, intercept, slope)
            img1 = window_image(img, 40, 80, rescale=self.rescaling)
            img2 = window_image(img, 80, 200, rescale=self.rescaling)
            if self.new_three_window_type:
                img3 = window_image(img, 40, 380, rescale=self.rescaling)
            else:
                img3 = window_image(img, 600, 2800, rescale=self.rescaling)
            if not self.rescaling:
                img1 = (img1 - 0) / 80
                img2 = (img2 - (-20)) / 200
                img3 = (img3 - (-150)) / 380
                img1 = img1 - img1.mean()
                img2 = img2 - img2.mean()
                img3 = img3 - img3.mean()
            if len(img1.shape) != 3:
                img1 = img1[:, :, np.newaxis]
            if len(img2.shape) != 3:
                img2 = img2[:, :, np.newaxis]
            if len(img3.shape) != 3:
                img3 = img3[:, :, np.newaxis]
            img = np.concatenate([img1, img2, img3], axis=2)
        elif self.subdural_window:
            window_center, window_width, intercept, slope = get_windowing(dataset)
            img = rescale_image(img, intercept, slope)
            img = window_image(img, 80, 200)

        if self.black_crop:
            try:
                img = img[np.min(h_cr):np.max(h_cr) + 1, np.min(w_cr):np.max(w_cr) + 1, :]
            except:
                print("pass black crop {}".format(img_id))

        if self.flip_aug and np.random.rand() >= 0.5:
            img = img[:, ::-1, :]
            target = np.pad(target, [0, 6], 'constant')
        elif self.flip_aug:
            target = np.pad(target, [6, 0], 'constant')

        if self.transforms is not None:
            augmented = self.transforms(image=img)
            img = augmented['image']

        if self.rescaling:
            img = img / 255
            img -= self.means
            img /= self.stds
        img = img.transpose((2, 0, 1))

        return torch.FloatTensor(img), torch.FloatTensor(target)


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
                 new_three_window_type=True,
                 n_tta=1,
                 rescaling=True,
                 external=False
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
        self.new_three_window_type = new_three_window_type
        self.external = external

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        img_path = os.path.join(self.image_path, img_id + self.img_type)
        dataset = pydicom.read_file(img_path)
        try:
            img = np.expand_dims(dataset.pixel_array, axis=2)
            is_dicom = 1
        except:
            img = np.zeros((512, 512, 1))
            is_dicom = 0
            #LOGGER.info("bad {}".format(img_path))
        if not self.three_window:
            img = img.repeat(3, axis=2)

        if img.shape[0] != 512:
            img = cv2.resize(img, (512, 512))
            print("reshape {}".format(img_id))


        if self.black_crop:
            try:
                mask_img = img > np.mean(img)
                sum_channel = np.sum(mask_img, 2)
                w_cr = np.where(sum_channel.sum(0) != 0)
                h_cr = np.where(sum_channel.sum(1) != 0)
            except:
                print("pass black crop {}".format(img_id))

        if self.three_window:
            window_center, window_width, intercept, slope = get_windowing(dataset)
            img = rescale_image(img, intercept, slope)
            img1 = window_image(img, 40, 80, rescale=self.rescaling)
            img2 = window_image(img, 80, 200, rescale=self.rescaling)
            if self.new_three_window_type:
                img3 = window_image(img, 40, 380, rescale=self.rescaling)
            else:
                img3 = window_image(img, 600, 2800, rescale=self.rescaling)
            if not self.rescaling:
                img1 = (img1 - 0) / 80
                img2 = (img2 - (-20)) / 200
                img3 = (img3 - (-150)) / 380
                img1 = img1 - img1.mean()
                img2 = img2 - img2.mean()
                img3 = img3 - img3.mean()
            if len(img1.shape) != 3:
                img1 = img1[:, :, np.newaxis]
            if len(img2.shape) != 3:
                img2 = img2[:, :, np.newaxis]
            if len(img3.shape) != 3:
                img3 = img3[:, :, np.newaxis]
            img = np.concatenate([img1, img2, img3], axis=2)
        elif self.subdural_window:
            window_center, window_width, intercept, slope = get_windowing(dataset)
            img = rescale_image(img, intercept, slope)
            img = window_image(img, 80, 200)

        if self.black_crop:
            try:
                img = img[np.min(h_cr):np.max(h_cr) + 1, np.min(w_cr):np.max(w_cr) + 1, :]
            except:
                print("pass black crop {}".format(img_id))

        if self.transforms is not None:
            augmented = self.transforms2(image=img)
            img_tta = augmented['image']
            augmented = self.transforms(image=img)
            img = augmented['image']

        imgs = []
        if self.rescaling:
            img = img / 255
            img -= self.means
            img /= self.stds
        img = img.transpose((2, 0, 1))
        imgs.append(torch.FloatTensor(img))
        if self.n_tta >= 2:
            flip_img = img[:, :, ::-1].copy()
            imgs.append(torch.FloatTensor(flip_img))

        if self.n_tta >= 4:
            if self.rescaling:
                img_tta = img_tta / 255
                img_tta -= self.means
                img_tta /= self.stds
            img_tta = img_tta.transpose((2, 0, 1))
            imgs.append(torch.FloatTensor(img_tta))
            flip_img_tta = img_tta[:, :, ::-1].copy()
            imgs.append(torch.FloatTensor(flip_img_tta))
        if self.external:
            return imgs, is_dicom
        else:
            return imgs


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
