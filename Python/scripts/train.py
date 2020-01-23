from __future__ import print_function
from __future__ import division

from utils.T_dataset import T_dataset
from utils.iou import iou
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from tensorboardX import SummaryWriter

import argparse
import os
import torch
import time
import torch.nn as nn
import torchvision
import torchvision.transforms as tf
import torchvision.utils as vutils



# ___author___: Teodor Totev
# ___contact__: tedi.totev97@gmail.com

# Define a logger
writer = SummaryWriter(logdir='/home/teo/storage/Code/name/DLV3_23_01_20_18_53_pascal')


def initialize_model(model_name, num_classes):

    if args.load == 0:

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
    else:
        model = torch.load(os.path.join(args.model_dir, args.model_name) +'/final_0.5670459316490073_comb_128')
        print('Load existing model...')

    return model  #, param_list


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    # Set an optimizer scheduler for decaying learning rate
    lambda1 = lambda epoch: ((1 - epoch/num_epochs)**0.9)*epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)

    since = time.time()  # Start a time recorder

    val_acc_history = []

    best_acc = 0.0
    iteration = {'train': 1, 'val': 1}

    # Perform training and validation
    for epoch in range(num_epochs):
        print('-'*50)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                scheduler.step()
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Containers for results
            tot_batches = dataloaders[phase].__len__()
            stor_lbl = torch.zeros((tot_batches, args.batch_size, args.size, args.size))
            stor_prd = torch.zeros((tot_batches, args.batch_size, args.size, args.size))

            # Iterate over data by getting batches
            for iter, (inputs, labels, indices, im_names, msk_names, sizes) in enumerate(dataloaders[phase]):
                inputs = inputs.to(args.device, dtype=torch.float)
                labels = labels.to(args.device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Get model outputs and calculate loss
                outputs = model(inputs)
                loss = criterion(outputs['out'], labels.long())

                _, preds = torch.max(outputs['out'], 1)

                # Store labels and preds
                n_batches = preds.shape[0]
                stor_lbl[iter, 0:n_batches, :, :] = labels.long()
                stor_prd[iter, 0:n_batches, :, :] = preds.long()

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Batch statistics
                running_loss += loss.item() * inputs.size(0)
                b_corrects = torch.sum(preds == labels.long())
                running_corrects += b_corrects
                b_pixacc = b_corrects.item()/(inputs.size(0)*args.size*args.size)
                b_IOU = iou(preds.long(), labels.long(), args.num_classes)

                # Report statistics if 'train'
                if phase == 'train':
                    writer.add_image('label', labels[0].unsqueeze(0)*30, iteration[phase])
                    writer.add_image('pred', preds[0].unsqueeze(0)*30, iteration[phase])

                    writer.add_scalar(phase + 'batch_loss', loss.item(), iteration[phase])
                    writer.add_scalar(phase + 'batch_pixacc', b_pixacc, iteration[phase])
                    writer.add_scalar(phase + 'batch_IOU', b_IOU, iteration[phase])

                    x = vutils.make_grid(inputs, normalize=True, scale_each=True)
                    y = vutils.make_grid(labels.unsqueeze(1) * 30, normalize=False, scale_each=True)
                    z = vutils.make_grid(preds.unsqueeze(1) * 30, normalize=False, scale_each=True)

                    writer.add_image(phase + 'batch_image', x, iteration[phase])
                    writer.add_image(phase + 'batch_label', y, iteration[phase])
                    writer.add_image(phase + 'batch_preds', z, iteration[phase])

                iteration[phase] += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_pixacc = running_corrects.item() / (len(dataloaders[phase].dataset)*args.size*args.size)

            # Calculate Intersection over Union (IoU)
            IOU = iou(stor_prd, stor_lbl, args.num_classes)

            # Deep copy the model if the best
            if phase == 'val' and IOU > best_acc:
                best_acc = IOU
                torch.save(model, final_model_file + '_' + str(best_acc) + '_pascal_' + str(args.size))
            if phase == 'val':
                val_acc_history.append(IOU)

            print('{} Loss: {:.4f} PixAcc: {:.4f} mIoU {:.4f}'.format(phase, epoch_loss, epoch_pixacc, IOU))
            print('Elapsed Time: {:.0f} minutes {:.0f} seconds '.format((time.time() - since) // 60, (time.time() - since) % 60))

            # Write to TensorboardX
            writer.add_scalar(phase + '_loss', epoch_loss, epoch)
            writer.add_scalar(phase + '_pixacc', epoch_pixacc, epoch)
            writer.add_scalar(phase + '_IOU', IOU, epoch)

            x = vutils.make_grid(inputs, normalize=True, scale_each=True)
            y = vutils.make_grid(labels.unsqueeze(1)*30, normalize=False, scale_each=True)
            z = vutils.make_grid(preds.unsqueeze(1)*30, normalize=False, scale_each=True)

            writer.add_image(phase + '_image', x, epoch)
            writer.add_image(phase + '_label', y, epoch)
            writer.add_image(phase + '_preds', z, epoch)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val mIoU: {:4f}'.format(best_acc))

    writer.close()


def main():

    model = initialize_model(args.model_name, args.num_classes)  # Initialize model

    model.cuda(args.device)  # Send to device

    # Set the optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_start)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.01)

    # Set the required transforms
    img_transforms = {
        'train': tf.Compose([
            tf.Resize(size=(args.size, args.size), interpolation=Image.BILINEAR),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': tf.Compose([
            tf.Resize(size=(args.size, args.size), interpolation=Image.BILINEAR),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    msk_transforms = {
        'train': tf.Compose([
            tf.Resize(size=(args.size, args.size), interpolation=Image.NEAREST)
        ]),
        'val': tf.Compose([
            tf.Resize(size=(args.size, args.size), interpolation=Image.NEAREST)
        ]),
    }

    # Load the data using dataset creator
    train_dataset = T_dataset(args.image_dir, args.mask_dir, 'train', img_transforms=img_transforms['train'], msk_transforms=msk_transforms['train'])
    val_dataset = T_dataset(args.image_dir, args.mask_dir, 'val', img_transforms=img_transforms['val'], msk_transforms=msk_transforms['val'])

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    dataloaders = {"train": train_loader, "val": val_loader}  # Create dataloader dictionary for ease of use

    # Setup the loss fxn to be used
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(model, dataloaders, criterion, optimizer, args.epochs)


if __name__ == '__main__':

    # Define parser for input arguments
    parser = argparse.ArgumentParser(description='Train a segmentation network with PyTorch')
    parser.add_argument('--load',       '-l', type=int, default=0)
    parser.add_argument('--image_dir',  '-im',type=str, default='/home/teo/storage/Data/Images/car_pascal')
    parser.add_argument('--mask_dir',   '-ma',type=str, default='/home/teo/storage/Data/Masks/car_pascal')
    parser.add_argument('--model_dir',  '-mo',type=str, default='/home/teo/storage/Data/Models/car_pascal')
    parser.add_argument('--size',       '-s' ,type=int, default=128)
    parser.add_argument('--num_classes','-nc',type=int, default=9)
    parser.add_argument('--model_name', '-mn',type=str, default='DeepLab101')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--epochs',     '-e', type=int, default=150)
    parser.add_argument('--device',      '-d',type=str, default='cuda:2')
    parser.add_argument('--num_workers', '-j',type=int, default=8)
    parser.add_argument('--lr_start', '-lr',type=int, default=0.0007)
    parser.add_argument('--lr_power', '-lrp',type=float, default=0.9)
    args = parser.parse_args()

    # Specify model directories
    model_dir = os.path.join(args.model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Define model save file
    final_model_file = os.path.join(model_dir, 'final')

    # Print starting information
    print('| Training %s on %s with PyTorch' % (args.model_name, args.device))
    print('| for %d epochs' % args.epochs)
    print('| and the model will be saved in: %s' % model_dir)

    # Run the training
    main()
