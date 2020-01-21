import os
import glob
import random
import shutil

def data_split():

    # Set data directory
    data_dir = '/home/teo/storage/Data'

    # Get mask list
    msk_list = sorted(glob.glob(data_dir + '/Masks/car_imagenet/*.csv'))

    # Define train and val folders:
    paths = []
    img_trainpath = data_dir + '/Images/car_imagenet/train'
    paths.append(img_trainpath)
    img_valpath = data_dir + '/Images/car_imagenet/val'
    paths.append(img_valpath)
    img_testpath = data_dir + '/Images/car_imagenet/test'
    paths.append(img_testpath)
    msk_trainpath = data_dir + '/Masks/car_imagenet/train'
    paths.append(msk_trainpath)
    msk_valpath = data_dir + '/Masks/car_imagenet/val'
    paths.append(msk_valpath)
    msk_testpath = data_dir + '/Masks/car_imagenet/test'
    paths.append(msk_testpath)

    # Check if folders exist and create them otherwise
    for f in paths:
        if os.path.isdir(f) == 0:
            os.mkdir(f)

    # Randomize data
    random.seed(1)
    num_msk = len(msk_list)
    idx = [i for i in range(num_msk)]
    random.shuffle(idx)

    small = len(idx)

    # Move data to corresponding folders
    for i in range(small):
        name = os.path.basename(msk_list[idx[i]][0:-9])
        msk_dir = data_dir + '/Masks/car_imagenet/' + name + '_mask.csv'
        img_dir = data_dir + '/Images/car_imagenet/' + name + '.JPEG'

        if i < 0.2*small - 1:
            msk_dest = data_dir + '/Masks/car_imagenet/test/' + name + '_mask.csv'
            img_dest = data_dir + '/Images/car_imagenet/test/' + name + '.JPEG'

        elif i < 0.36*small - 1:
            msk_dest = data_dir + '/Masks/car_imagenet/val/' + name + '_mask.csv'
            img_dest = data_dir + '/Images/car_imagenet/val/' + name + '.JPEG'

        else:
            msk_dest = data_dir + '/Masks/car_imagenet/train/' + name + '_mask.csv'
            img_dest = data_dir + '/Images/car_imagenet/train/' + name + '.JPEG'

        # Move the data -> be careful
        shutil.move(img_dir, img_dest)
        shutil.move(msk_dir, msk_dest)

if __name__ == '__main__':
    data_split()