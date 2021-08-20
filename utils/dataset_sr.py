import os
import os.path
import numpy as np
import random
import torch
import cv2
import glob
import torch.utils.data as udata
from os import path
from utils.utils import data_augmentation
from torch.utils.data import DataLoader
from PIL import Image

class DataLoad(udata.Dataset):

    def __init__(self, batch_size, patch_size=128, target_dir='data/trainset', train=True, keep_range=False):
        if train:
            split = 'train'
            split_path = 'train_crop'
            train_input = 'train/degraded'
            train_gt = 'train/gt'
        else:
            split = 'val'
            split_path = split
            train_input = 'val/val_blur_jpeg'
            train_gt = 'val/val_sharp'


        path_degraded = path.join(target_dir, train_input)
        scan_degraded = self.scan_over_dirs_png(path_degraded)
        path_gt = path.join(target_dir, train_gt)
        scan_gt = self.scan_over_dirs_png(path_gt)
        scans = [(b, s) for b, s, in zip(scan_degraded, scan_gt)]

        # if train:
        #     random.shuffle(scans)
        #     scans = scans[0:16000]
        # else: #val
        #     #random.shuffle(scans)
        #     scans = scans[0:2999:20]

        print('train =',train,len(scans))

        #print(scans)
        # Shuffle the dataset
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.scans = scans
        self.train = train
        self.keep_range = keep_range

    def scan_over_dirs_png(self, dir):
        filenames = os.listdir(dir)
        folderlist = []
        fileslist = []
        for names in filenames:
            name = names.split('/')[0]
            folderlist.append(name)
        for names in folderlist:
            files = glob.glob(os.path.join(dir, names, '*.png'))
            files.sort()
            fileslist.extend(files)

        return fileslist

    def __len__(self):
        return len(self.scans) // self.batch_size

    def __getitem__(self, idx):
        image, target = self.scans[idx]
        # print(self.scans[idx])

        degraded = Image.open(image)
        gt = Image.open(target)
        if self.train:
            random_angle = int(random.random() * 5) * 90
            degraded = degraded.rotate(random_angle)
            # degraded = degraded.resize((degraded.size(0) // 2, degraded.size(1) // 2), resample=Image.BICUBIC)
            gt = gt.rotate(random_angle)
            # degraded, gt = self.random_crop_img(degraded, gt)
            # degraded = degraded.resize((self.patch_size // 2, self.patch_size // 2), resample=Image.BICUBIC)
            degraded = np.asarray(degraded, dtype=np.float32)
            gt = np.asarray(gt, dtype=np.float32)

            # if random crop in numpy dimension
            degraded, gt = self.random_crop(degraded, gt)
            degraded = Image.fromarray(np.uint8(degraded))
            degraded = degraded.resize((self.patch_size//2, self.patch_size//2), resample=Image.BICUBIC)
            degraded = np.asarray(degraded, dtype=np.float32)

            # degraded = np.transpose(degraded, (2, 0, 1))
            # gt = np.transpose(gt, (2, 0, 1))
            #
            # degraded = self.Im2Patch(degraded, self.patch_size, self.patch_size)
            # gt = self.Im2Patch(gt, self.patch_size, self.patch_size)

            degraded, gt = self.train_preprocess(degraded, gt)
        else:
            degraded = np.asarray(degraded, dtype=np.float32)
            gt = np.asarray(gt, dtype=np.float32)

        degraded /= 255.0
        gt /= 255.0
        degraded = np.transpose(degraded, (2, 0, 1))
        gt = np.transpose(gt, (2, 0, 1))

        # sample = {'image': image, 'target': target}

        return degraded, gt

    def random_crop(self, degraded, gt):
        h, w, _ = degraded.shape

        py = random.randrange(0, h - self.patch_size + 1)
        px = random.randrange(0, w - self.patch_size + 1)

        crop_degraded = degraded[py:(py + self.patch_size), px:(px + self.patch_size)]
        crop_gt = gt[py:(py + self.patch_size), px:(px + self.patch_size)]

        return crop_degraded, crop_gt

    def random_crop_img(self, degraded, gt):
        h = degraded.size[0] - self.patch_size
        w = degraded.size[1] - self.patch_size

        py = random.randrange(0, h//2 + 1)*2
        px = random.randrange(0, w//2 + 1)*2
        # area = (px, py, px + self.patch_size, py + self.patch_size)
        area = (py, px, py + self.patch_size, px + self.patch_size)

        crop_degraded = degraded.crop(area)
        crop_gt = gt.crop(area)

        return crop_degraded, crop_gt

    # def random_crop(lr_img, hr_img, hr_crop_size):
    #     lr_crop_size = hr_crop_size
    #
    #     lr_w = np.random.randint(lr_img.shape[1] - lr_crop_size + 1)
    #     lr_h = np.random.randint(lr_img.shape[0] - lr_crop_size + 1)
    #
    #     hr_w = lr_w
    #     hr_h = lr_h
    #
    #     lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    #     hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]
    #
    #     return lr_img_cropped, hr_img_cropped

    def Im2Patch(self, img, win, stride=1):
        k = 0
        endc = img.shape[0]
        endw = img.shape[1]
        endh = img.shape[2]
        patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
        TotalPatNum = patch.shape[1] * patch.shape[2]
        Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
        for i in range(win):
            for j in range(win):
                patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
                Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
                k = k + 1
        return Y.reshape([endc, win, win, TotalPatNum])

    def train_preprocess(self, image, target):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            target = (target[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        # do_augment = random.random()
        # if do_augment > 0.5:
        #     image = self.augment_image(image)

        return image, target

    def augment_image(self, image):
            # gamma augmentation
            gamma = random.uniform(0.9, 1.1)
            image_aug = image ** gamma

            # brightness augmentation

            brightness = random.uniform(0.9, 1.1)
            image_aug = image_aug * brightness

            # color augmentation
            colors = np.random.uniform(0.9, 1.1, size=3)
            white = np.ones((image.shape[0], image.shape[1]))
            color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
            image_aug *= color_image
            image_aug = np.clip(image_aug, 0, 1)

            return image_aug

class Dataset(object):
    def __init__(self, train=True, batchSize=2, patchSize=128):
        if train:
            # self.transformed_data = DataLoad(batch_size=batchSize, patch_size=patchSize, train=True,)
            self.transformed_data = DataLoad(1, patch_size=patchSize, train=True, )
            print(len(self.transformed_data))
            self.data = DataLoader(self.transformed_data, batchSize, num_workers=4, shuffle=True)
        else:
            self.transformed_data = DataLoad(1, train=False,)
            self.data = DataLoader(self.transformed_data, 1, num_workers=4, shuffle=False)


    def __len__(self):
        return len(self.transformed_data)