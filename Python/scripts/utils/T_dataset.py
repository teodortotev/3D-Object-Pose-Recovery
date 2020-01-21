import numpy as np
import glob
import os
import torch

from torch.utils.data.dataset import Dataset
from PIL import Image
from utils.visualize_mask import visualize_mask
from utils.make_rgb import make_rgb

# This is a dataset creator script that inherits from PyTorch Dataset class
# It loads a dataset which is previously split into train, test, and val folders
# The labels are segmentation masks in the form of csv files

# ___author___: Teodor Totev
# ___contact__: tedi.totev97@gmail.com


class T_dataset(Dataset):

    def __init__(self, img_dir,  msk_dir, split, img_transforms, msk_transforms):
        super(T_dataset, self).__init__()

        # Check if an existing set is wanted
        assert split in ['train', 'val', 'test'], 'split must be "train","val" or "test"'

        # Get a list of all images and labels
        img_list = sorted(glob.glob(os.path.join(img_dir, split) + '/*'))
        file_names = [os.path.splitext(os.path.basename(x))[0] for x in img_list]
        msk_list = [os.path.join(msk_dir, split, x + "_mask.csv") for x in file_names]

        # Set the labels
        self.images = img_list
        self.masks = msk_list
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.split = split
        self.img_transforms = img_transforms
        self.msk_transforms = msk_transforms
        self.n_data = len(self.images)

    def read_image(self, path):
        image = Image.open(path)
        if image.mode != 'RGB':
            image = make_rgb(image)
        return image

    def read_mask(self, path):
        mask = visualize_mask(path)
        return mask

    def get_train_item(self, index):
        image = self.read_image(self.images[index])
        mask = self.read_mask(self.masks[index])

        # Apply transforms for training data
        if self.img_transforms:
            image = self.img_transforms(image)

        if self.msk_transforms:
            mask = self.msk_transforms(mask)
            mask = np.array(mask)

        return torch.as_tensor(image), torch.as_tensor(mask), torch.as_tensor(index), self.images[index], self.masks[index]

    def get_val_item(self, index):
        image = self.read_image(self.images[index])
        mask = self.read_mask(self.masks[index])

        # Apply transforms for testing data
        if self.img_transforms:
            image = self.img_transforms(image)

        if self.msk_transforms:
            mask = self.msk_transforms(mask)
            mask = np.array(mask)

        return torch.as_tensor(image), torch.as_tensor(mask), torch.as_tensor(index), self.images[index], self.masks[index]

    def __getitem__(self, index):
        if self.split is 'train':
            return self.get_train_item(index)
        else:
            return self.get_val_item(index)

    def __len__(self):
        return self.n_data

if __name__ == '__main__':

    # TODO This needs to be checked if the function is to be used independently

    #img_dir = 'C:/Users/Teo/Documents/Engineering/Year4/4YP/Data/Images/car_pascal/'
    #msk_dir = 'C:/Users/Teo/Documents/Engineering/Year4/4YP/Data/Masks/car_pascal/'
    T_dataset()
