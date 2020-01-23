import os
import argparse
import time
import torchvision.transforms as tf
import torch
from torch.utils.data import DataLoader
from utils.iou import iou
from utils.T_dataset import T_dataset
from PIL import Image
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils


# This is a script that takes a trained model and tests it on a test set

# ___author___: Teodor Totev
# ___contact__: tedi.totev97@gmail.com

def resized_iou(preds, labels, n_classes, sizes):
    ious = []
    for i in range(len(preds)):
        p = tf.ToPILImage()(preds[i].cpu().numpy().astype(np.uint8))
        l = tf.ToPILImage()(labels[i].cpu().numpy().astype(np.uint8))
        p = tf.Resize(size=(sizes[0][i].tolist(), sizes[1][i].tolist()), interpolation=Image.NEAREST)(p)
        l = tf.Resize(size=(sizes[0][i].tolist(), sizes[1][i].tolist()), interpolation=Image.NEAREST)(l)
        p = np.array(p)
        l = np.array(l)
        p = torch.as_tensor(p)
        l = torch.as_tensor(l)
        ious.append(iou(p, l, n_classes))
    
    return sum(ious)/len(preds)



def main():
    since = time.time()  # Record start time

    writer = SummaryWriter(comment='_testing')

    print('-' * 50)
    print('| Loading model file %s...' % final_model_file)

    model = torch.load(final_model_file)  # Get the model
    model.eval()  # Set the model to evaluation mode

    print('| Done!')

    model.cuda(args.device)  # Send the model to appropriate device (GPU or CPU)

    # Set the required transforms (maybe different normalization using per-channel mean and stddev will perform better -> need to be consistent with the normalizations used in trainer.py)
    img_transforms = {
        'test': tf.Compose([
            tf.Resize(size=(args.size, args.size), interpolation=Image.BILINEAR),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    msk_transforms = {
        'test': tf.Compose([
            tf.Resize(size=(args.size, args.size), interpolation=Image.NEAREST)
        ])
    }

    test_dataset = T_dataset(args.image_dir, args.mask_dir, 'test', img_transforms=img_transforms['test'], msk_transforms=msk_transforms['test'])

    # Create a dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    correct_pixels = 0  # Initialize the number of correctly classified images

    # Containers for results
    tot_batches = test_loader.__len__()
    stor_im_n = []
    stor_msk_n = []
    stor_sizes = []
    batch_ious = []
    resized_batch_ious = []
    stor_im = torch.zeros((tot_batches, args.batch_size, 3, args.size, args.size))
    stor_lbl = torch.zeros((tot_batches, args.batch_size, args.size, args.size))
    stor_prd = torch.zeros((tot_batches, args.batch_size, args.size, args.size))

    with torch.no_grad():  # Disable gradient calculations
        for iter, (inputs, labels, indices, im_names, msk_names, sizes) in enumerate(test_loader):
            inputs = inputs.to(args.device, dtype=torch.float)  # Send images to device
            labels = labels.to(args.device)  # Send labels to device
            labels = labels.squeeze(1)

            outputs = model(inputs)  # Feed through the model

            _, preds = torch.max(outputs['out'], 1)  # Get predictions

            correct_pixels += torch.sum(preds == labels.long())  # Count correctly classified pixels

            # Store labels and preds
            stor_sizes.extend(sizes)
            n_batches = preds.shape[0]
            stor_im_n.extend(im_names)
            stor_msk_n.extend(msk_names)
            stor_im[iter, 0:n_batches, :, :, :] = inputs
            stor_lbl[iter, 0:n_batches, :, :] = labels.long()
            stor_prd[iter, 0:n_batches, :, :] = preds.long()

            # Resize images and compute batch IoU
            batch_ious.append(iou(preds.long(), labels.long(), args.num_classes))
            resized_batch_ious.append(resized_iou(preds.long(), labels.long(), args.num_classes, sizes))

            #save_predictions(preds, indices, args.pred_dir, test_dataset)

    # Calculate individual image losses
    #im_iou = np.ones((tot_batches, args.batch_size))
    #for i in range(tot_batches):
    #    for j in range(args.batch_size):
    #        im_iou[i, j] = iou(stor_prd[i, j, :, :], stor_lbl[i, j, :, :], args.num_classes) 

    # # Display bad predictions under certain mIoU threshold
    # count = 0
    # for i in range(im_iou.shape[0]):
    #     for j in range(im_iou.shape[1]):
    #         if im_iou[i, j] < 0.1:
    #             writer.add_scalar('mIOU', im_iou[i, j], i*im_iou.shape[1] + j)

    #             x = vutils.make_grid(stor_im[i, j, :, :], normalize=True, scale_each=True)
    #             y = vutils.make_grid(stor_lbl[i, j, :, :] * 30, normalize=False, scale_each=True)
    #             z = vutils.make_grid(stor_prd[i, j, :, :] * 30, normalize=False, scale_each=True)

    #             writer.add_text('im_name', stor_im_n[i*im_iou.shape[1] + j], i*im_iou.shape[1] + j)
    #             writer.add_text('msk_name', stor_msk_n[i*im_iou.shape[1] + j], i*im_iou.shape[1] + j)

    #             writer.add_image('image', x, i*im_iou.shape[1] + j)
    #             writer.add_image('label', y, i*im_iou.shape[1] + j)
    #             writer.add_image('pred', z, i*im_iou.shape[1] + j)
    #             count += 1

    writer.close()

    mbIoU = sum(batch_ious)/len(batch_ious) # Mean batch IoU
    mrbIoU = sum(resized_batch_ious)/len(resized_batch_ious) # Mean resized batch IoU
    IOU = iou(stor_prd, stor_lbl, args.num_classes) # Overall IoU
    acc = correct_pixels.double() / (len(test_loader.dataset)*args.size*args.size)  # Pixel accuracy incl. background
    IOU = '{:4f}'.format(IOU)
    mbIoU = '{:4f}'.format(mbIoU)
    acc = '{:4f}'.format(acc)
    time_elapsed = time.time() - since  # Time elapsed

    print('-' * 50)
    print('| Mean Batch IoU is is %s' % str(mbIoU))
    print('| Mean Resized Batch IoU is is %s' % str(mrbIoU))
    print('| Overall IoU is is %s' % str(IOU))
    print('| Pixel Accuracy is %s' % str(acc))
    print('| Time taken: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':

    # Parser for input arguments
    parser = argparse.ArgumentParser(description='Test a model with PyTorch')
    parser.add_argument('--image_dir',  '-im',  type=str, default='/home/teo/storage/Data/Images/car_imagenet', help="Specify image directory")
    parser.add_argument('--mask_dir',   '-ma',  type=str, default='/home/teo/storage/Data/Masks/car_imagenet',  help="Specify mask directory")
    parser.add_argument('--model_dir',  '-md',  type=str, default='/home/teo/storage/Data/Models/car_imagenet', help='Specify model directory.')
    parser.add_argument('--pred-dir',   '-pd',  type=str, default='/home/teo/storage/Data/Predictions/car_imagenet', help='Specify prediction directory')
    parser.add_argument('--size',        '-s',  type=int, default='128',   help='Specify image resize.')
    parser.add_argument('--model',      '-m',   type=str, default='final_0.6880962872804797_imagenet_128',  help='Specify the model type to test. Default: ResNet')
    parser.add_argument('--model_name', '-mn',  type=str, default='DeepLab101',  help='Specify the exact model name to test. Default: final_0.931575')
    parser.add_argument('--batch_size', '-b',   type=int, default=32,         help='Specify the batch size. Default: 16')
    parser.add_argument('--device',     '-d',   type=str, default='cuda:1',  help='Specify the device to be used. Default: cuda:0')
    parser.add_argument('--num_classes', '-nc', type=int, default=9)
    parser.add_argument('--num_workers', '-j',  type=int, default=8,         help='Specify the number of processes to load data. Default: 8')
    args = parser.parse_args()

    # Set the model directories
    model_dir = os.path.join(args.model_dir, args.model_name)

    # Set model path
    final_model_file = os.path.join(model_dir, args.model)

    # Assert that it exists
    assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

    print('-' * 50)
    print('| Testing %s %s on %s with PyTorch' % (args.model_name, args.model, args.device))

    # Run the test
    main()