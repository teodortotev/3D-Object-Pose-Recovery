import json
import torch
import os
import torchvision
import torchvision.transforms as tf
import numpy as np
from tensorboardX import SummaryWriter
import scipy.io as sio

from PIL import Image
from tqdm import tqdm
from utils.make_rgb import make_rgb
from utils.optim import optim

def find_seg_iou(pred, target, n_classes):
    ious = []
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float(1))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.mean(np.array(ious))

def find_bbox_iou(boxA, boxB):
    # Determine the (x,y) coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA +1)

    # Compute the area of each bbox
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute iou
    iou = interArea/ float(boxAArea + boxBArea - interArea)

    return iou

def import_data():

    # TODO: We need to laod targets too

    img_dir = "/home/teo/storage/Data/Images/car_combined/test/"
    pred_file =  "/home/teo/storage/Data/Models/car_combined/MaskRCNN/inference/pasc3d_test_carparts_cocostyle/bbox.json"
    model_path = "/home/teo/storage/Data/Models/car_combined/FCN101/final_0.6799256751510474_comb_128_0.0001"
    test_file = "/home/teo/storage/Data/Annotations/car_combined/car_parts/test_anno.json"
    anno_dir = "/home/teo/storage/Data/Annotations/car_combined/car_objects/"
    cad_path = "/home/teo/storage/Data/CAD/labelled_cads_centre.mat"

    with open(pred_file) as json_file:
        boxes = json.load(json_file)

    with open(test_file) as json_file:
        data = json.load(json_file)
        img_names = data['images']

    model = torchvision.models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=9, aux_loss=None)
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.to('cuda:0')
    
    return boxes, img_names, model, img_dir, anno_dir, cad_path

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

def resize_pred(pred, size, n_classes):
    pred = pred.cpu().numpy().astype(np.uint8)
    pred = tf.ToPILImage()(pred)
    pred = tf.Resize(size=(size[1], size[0]), interpolation=Image.NEAREST)(pred)
    pred = np.array(pred)
    pred_mask = torch.as_tensor(pred)
    pred = add_colours(pred, size[1], size[0])
    pred = torch.as_tensor(pred)

    return pred, pred_mask

def get_target(record, bbox):
    img_name = record['filename'][0][0][0]
    best_match = 0
    best_iou = 0
    # Go through all object annotations in an image
    for a in range(len(record['objects'][0][0][0])):
        clas = record['objects'][0][0][0][a][0][0]
        if clas == 'car':
            if 'n' in img_name:
                t_bbox = record['objects'][0][0][0][a][1][0]
                x1 = t_bbox[0].tolist()
                y1 = t_bbox[1].tolist()
                x2 = t_bbox[2].tolist()
                y2 = t_bbox[3].tolist()
                t_bbox = [x1, y1, x2, y2]
            else:
                t_bbox = record['objects'][0][0][0][a][8][0][0]
                x1 = t_bbox[0][0][0].tolist()
                y1 = t_bbox[1][0][0].tolist()
                x2 = t_bbox[2][0][0].tolist()
                y2 = t_bbox[3][0][0].tolist()
                t_bbox = [x1, y1, x2, y2]

            iou = find_bbox_iou(bbox, t_bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = a

    path = "/home/teo/storage/Data/Masks/car_combined/single/" + os.path.splitext(img_name)[0] + '_mask_' + str(best_match+1) + '.mat'

    if os.path.exists(path):
        target = sio.loadmat(path)
        target = target['single_mask']
        target = target[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    
    return target

def main():

    writer = SummaryWriter(logdir='/home/teo/storage/Code/name/box_seg')

    boxes, img_names, model, img_dir, anno_dir, cad_path = import_data()
    model.eval()

    # Load cads
    cads = sio.loadmat(cad_path)
    cads = cads['cads']

    # Find CAD means
    CAD_means = np.zeros((10, 3, 9))
    CAD_count = np.zeros((10, 9))

    for m in range(10):
        for i in range(cads[0][m][0].shape[0]):
            CAD_count[m, int(cads[0][m][0][i][3])] += 1
            CAD_means[m, 0, int(cads[0][m][0][i][3])] += cads[0][m][0][i][0]
            CAD_means[m, 1, int(cads[0][m][0][i][3])] += cads[0][m][0][i][1]
            CAD_means[m, 2, int(cads[0][m][0][i][3])] += cads[0][m][0][i][2]

    CAD_means[:,0,:] = CAD_means[:,0,:]/(CAD_count + 0.0001)
    CAD_means[:,1,:] = CAD_means[:,1,:]/(CAD_count + 0.0001)
    CAD_means[:,2,:] = CAD_means[:,2,:]/(CAD_count + 0.0001)

    # Define transforms
    img_transforms = tf.Compose([
                            tf.Resize(size=(128, 128), interpolation=Image.BILINEAR),
                            tf.ToTensor(),
                            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])


    # Generate and display predictions
    for i, box in enumerate(tqdm(boxes)):

        # Load image and make sure it is RGB
        image = Image.open(img_dir + img_names[box['image_id']]['file_name'])       
        if image.mode != 'RGB':
            image = make_rgb(image)

        # Get relevant image snippet
        bbox = box['bbox']
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = x1 + bbox[2]
        y2 = y1 + bbox[3]
        bbox = [x1, y1, x2, y2]
        crop = image.crop((int(x1), int(y1), int(x2), int(y2)))

        # Get label
        label = box['category_id']

        # Apply transforms
        img_in = img_transforms(crop)
        img_in = img_in.to('cuda:0', dtype=torch.float)
        img_in = img_in.unsqueeze(0)

        # Get predictions
        output = model(img_in)
        _, pred = torch.max(output['out'], 1)
        pred = pred[0, :, :].long()                            

        pred, pred_mask = resize_pred(pred, crop.size, 9) 

        crop = np.array(crop)
        crop = np.transpose(crop,(2,0,1))
        crop = torch.as_tensor(crop)

        # Load target
        anno = sio.loadmat(anno_dir + os.path.splitext(img_names[box['image_id']]['file_name'])[0] + '.mat')
        record = anno['record']

        # Get the best fit target
        target = get_target(record, bbox)
        target_mask = torch.as_tensor(target)        
        target = add_colours(target, target.shape[0], target.shape[1])
        target = torch.as_tensor(target)

        # Find segmentation IoU
        seg_iou = find_seg_iou(pred_mask, target_mask, n_classes=9)

        # Write iou to output
        writer.add_scalar('iou', seg_iou, i)

        # Write images to output
        writer.add_image('pred', pred, i)
        writer.add_image('crop', crop, i)
        writer.add_image('target', target, i)

        # Recover pose from target (prove that it works)

        # Count the number of pixels in each label and their mean coordinates
        # DO NOT FORGET that this is cropped image i.e. add the crop positions to the top left corner of crop
        count = np.zeros(9)
        means = np.zeros((2,9))
        for i in range(target_mask.shape[0]):
            for j in range(target_mask.shape[1]):
                count[int(target_mask[i,j])] += 1
                means[0,int(target_mask[i,j])] += i
                means[1,int(target_mask[i,j])] += j

        means[0, :] = means[0, :]/(count + 0.0001) + float(bbox[0])
        means[1, :] = means[1, :]/(count + 0.0001) + float(bbox[1])

        # Get viewpoint estimate
        viewpoint = optim(means[:,1:9], count[1:9], CAD_means[label, :, 1:9], CAD_count[1:9], image.size[::-1])

if __name__ == "__main__":
    main()