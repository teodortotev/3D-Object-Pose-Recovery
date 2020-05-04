import os
import argparse
import time
import torchvision.transforms as tf
import torch
from torch.utils.data import DataLoader
from utils.iou import iou
from utils.pixacc import pixacc
from utils.T_dataset import T_dataset
from PIL import Image
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torchvision


# This is a script that takes a trained model and tests it on a test set

# ___author___: Teodor Totev
# ___contact__: tedi.totev97@gmail.com

def initialize_model(model_name, num_classes):
    # Load pre-trained model weights and modify the model
    if model_name == 'FCN101':
        model = torchvision.models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=num_classes, aux_loss=None)

    if model_name == 'DeepLab101':
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=num_classes, aux_loss=None)

    pretrained_state_dict = torch.load("/home/teo/storage/Data/pretrained_weight_DeepLab101")
    model.load_state_dict(pretrained_state_dict, strict=False)

    for param in model.parameters():
        param.requires_grad = True

    #param_list = []
    #for param in model.classifier._modules['4'].parameters():
    #    param.requires_grad = True
    #    param_list.append(param)


    return model  #, param_list

def add_colours(mask, w, h):
    pix = np.zeros((3, w, h))
    colors = np.array([[0.954174456379543, 0.590608652919636, 0.281507695118553],
                       [0.0319226295039784, 0.660437966312602, 0.731050829723742],
                       [0.356868986182542, 0.0475546731138661, 0.137762892519516],
                       [0.662653834287215, 0.348784808510059, 0.836722781749718],
                       [0.281501559148491, 0.451340580355743, 0.138601715742360],
                       [0.230383067317464, 0.240904997120111, 0.588209385389494],
                       [0.711128551180325, 0.715045013296177, 0.366156800454938],
                       [0.624572916993309, 0.856182292006288, 0.806759544661106]])
    for label in range(1,9):
        for i in range(w):
            for j in range(h):
                if mask[i,j] == label:
                    pix[:, i, j] = colors[label-1, :]

    return pix

def resized_iou(inputs, preds, labels, n_classes, sizes, writer, names):
    ious = []
    accs = []
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(len(preds)):

        # Resize masks
        p = preds[i].cpu().numpy().astype(np.uint8)
        l = labels[i].cpu().numpy().astype(np.uint8)
        p = tf.ToPILImage()(p)
        l = tf.ToPILImage()(l)
        p = tf.Resize(size=(sizes[1][i].tolist(), sizes[0][i].tolist()), interpolation=Image.NEAREST)(p)
        l = tf.Resize(size=(sizes[1][i].tolist(), sizes[0][i].tolist()), interpolation=Image.NEAREST)(l)
        p = np.array(p)
        l = np.array(l)
        p_color = add_colours(p, sizes[1][i].tolist(), sizes[0][i].tolist())
        p_color = torch.as_tensor(p_color)
        l_color = add_colours(l, sizes[1][i].tolist(), sizes[0][i].tolist())
        l_color = torch.as_tensor(l_color)
        # p = torch.as_tensor(p)
        # l = torch.as_tensor(l)

        # # Calculate ious 
        # ious.append(iou(p, l, n_classes))
        # accs.append(pixacc(p, l))

        # Unnormalize and Resize images
        image = inputs[i].cpu().numpy()
        image[0] = (image[0]*std[0] + mean[0])*255
        image[1] = (image[1]*std[1] + mean[1])*255
        image[2] = (image[2]*std[2] + mean[2])*255
        image = image.astype(np.uint8)
        image = np.transpose(image,(1,2,0))
        image = Image.fromarray(image, mode='RGB')
        image = tf.Resize(size=(sizes[1][i].tolist(), sizes[0][i].tolist()), interpolation=Image.BILINEAR)(image)
        image = np.array(image)
        image = np.transpose(image,(2,0,1))

        name = os.path.basename(names[i])

        # Display images and masks in the batch
        writer.add_image('image_' + name, image, i)
        writer.add_image('pred_' + name, p_color, i)
        writer.add_image('target_' + name, l_color, i)

    return sum(ious)/len(preds), sum(accs)/len(preds)

def main():
    since = time.time()  # Record start time

    writer = SummaryWriter(logdir='/home/teo/storage/Code/name/candelete')
    torch.manual_seed(2)

    print('-' * 50)
    print('| Loading model file %s...' % final_model_file)

    model = initialize_model(args.model_name, args.num_classes)
    model.load_state_dict(torch.load(final_model_file))

    # model = torch.load(final_model_file, map_location='cuda:0')  # Get the model
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

    torch.manual_seed(2)

    # Create a dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
    resized_batch_accs = []
    stor_im = torch.zeros((tot_batches, args.batch_size, 3, args.size, args.size))
    stor_lbl = torch.zeros((tot_batches, args.batch_size, args.size, args.size))
    stor_prd = torch.zeros((tot_batches, args.batch_size, args.size, args.size))

    with torch.no_grad():  # Disable gradient calculations
        for iter, (inputs, labels, indices, im_names, msk_names, sizes) in enumerate(test_loader):
            print(iter)
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

            # Resize images, compute batch IoU and display
            batch_ious.append(iou(preds.long(), labels.long(), args.num_classes))
            rbious, rbaccs = resized_iou(inputs, preds.long(), labels.long(), args.num_classes, sizes, writer, im_names)
            resized_batch_ious.append(rbious)
            resized_batch_accs.append(rbaccs)

            #save_predictions(preds, indices, args.pred_dir, test_dataset)
            exit()

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
    mracc = sum(resized_batch_accs)/len(resized_batch_accs)
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
    print('| Mean Resized Pixel Accuracy is %s' % str(mracc))
    print('| Pixel Accuracy is %s' % str(acc))
    print('| Time taken: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':

    # Parser for input arguments
    parser = argparse.ArgumentParser(description='Test a model with PyTorch')
    parser.add_argument('--image_dir',  '-im',  type=str, default='/home/teo/storage/Data/Images/car_pascal', help="Specify image directory")
    parser.add_argument('--mask_dir',   '-ma',  type=str, default='/home/teo/storage/Data/Masks_old/car_pascal',  help="Specify mask directory")
    parser.add_argument('--model_dir',  '-md',  type=str, default='/home/teo/storage/Data/Models/car_pascal', help='Specify model directory.')
    parser.add_argument('--pred-dir',   '-pd',  type=str, default='/home/teo/storage/Data/Predictions/car_pascal', help='Specify prediction directory')
    parser.add_argument('--size',        '-s',  type=int, default='128',   help='Specify image resize.')
    parser.add_argument('--model',      '-m',   type=str, default='final_0.300615063857375_pascal_128_0.0001',  help='Specify the model type to test. Default: ResNet')
    parser.add_argument('--model_name', '-mn',  type=str, default='DeepLab101',  help='Specify the exact model name to test. Default: final_0.931575')
    parser.add_argument('--batch_size', '-b',   type=int, default=64,         help='Specify the batch size. Default: 16')
    parser.add_argument('--device',     '-d',   type=str, default='cuda:0',  help='Specify the device to be used. Default: cuda:0')
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