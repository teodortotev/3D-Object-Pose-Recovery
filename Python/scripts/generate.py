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

def find_mask_iou(pred, label, n_classes):
    """
    Mask 1 and Mask 2 need to be of the same size
    (input) Tensors
    """
    ious = []
    pred = pred.view(-1)
    label = label.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes):
        pred_inds = pred == cls
        target_inds = label == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float(1))  # If this mask does not exist in both pred and label then it is classified perfectly
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.mean(np.array(ious))

def multi_to_color(mask):
    colors = np.array([[0.954174456379543, 0.590608652919636, 0.281507695118553],
        [0.0319226295039784, 0.660437966312602, 0.731050829723742],
        [0.356868986182542, 0.0475546731138661, 0.137762892519516],
        [0.662653834287215, 0.348784808510059, 0.836722781749718],
        [0.281501559148491, 0.451340580355743, 0.138601715742360],
        [0.230383067317464, 0.240904997120111, 0.588209385389494],
        [0.711128551180325, 0.715045013296177, 0.366156800454938],
        [0.624572916993309, 0.856182292006288, 0.806759544661106],
        [0.424572916993309, 0.556182292006288, 0.306759544661106],
        [0.324572916993309, 0.256182292006288, 0.906759544661106]])

    color_mask = torch.zeros((3, mask.shape[0], mask.shape[1]))
    for cat in range(8):
        index_mask = mask == cat + 1
        color_mask[0,index_mask] = colors[cat][0]
        color_mask[1,index_mask] = colors[cat][1]
        color_mask[2,index_mask] = colors[cat][2]
    return color_mask

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

def get_target(record):
    img_name = record['filename'][0][0][0]
    # Go through all object annotations in an image
    targets = []
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

            if os.path.isfile("/home/teo/storage/Data/Masks/car_combined/single/" + os.path.splitext(img_name)[0] + '_mask_' + str(a + 1) + '.mat'):
                target = {
                    'bbox' : t_bbox,
                    'number' : a + 1,
                }
                targets.append(target)
    
    return targets

def import_data():

    paths = {
        'img_dir' : "/home/teo/storage/Data/Images/car_combined/test/",
        'pred_file' :  "/home/teo/storage/Data/Models/car_combined/MaskRCNN/inference/pasc3d_test_carparts_cocostyle/bbox.json",
        'model_path' : "/home/teo/storage/Data/Models/car_combined/DeepLab101/final_0.6742715127254424_comb_128_0.0001",
        'test_file' : "/home/teo/storage/Data/Annotations/car_combined/car_parts/test_anno.json",
        'anno_dir' : "/home/teo/storage/Data/Annotations/car_combined/car_objects/",
        'cad_path' : "/home/teo/storage/Data/CAD/labelled_cads_centre.mat",
        'single_path' : "/home/teo/storage/Data/Masks/car_combined/single/",
    }

    with open(paths['pred_file']) as json_file:
        pred_boxes = json.load(json_file)

    with open(paths['test_file']) as json_file:
        images = json.load(json_file)['images']

    # Load model
    # model = torchvision.models.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=9, aux_loss=None)
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=9, aux_loss=None)
    model.load_state_dict(torch.load(paths['model_path'], map_location='cuda:0'))
    model.to('cuda:0')
    model.eval()

    # Define transforms
    img_transforms = tf.Compose([
                        tf.Resize(size=(128, 128), interpolation=Image.BILINEAR),
                        tf.ToTensor(),
                        tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

    data = {
        'pred_boxes' : pred_boxes,
        'model' : model,
        'img_info' : images,
        'paths' : paths,
        'transforms' : img_transforms,
    }
    
    return data

def main():

    writer = SummaryWriter(logdir='/home/teo/storage/Code/name/generate')

    data = import_data()

    # Go through all images in the dataset
    count = 0
    img_iou = 0
    for number, image in tqdm(enumerate(data['img_info'])):
        name = image['file_name']
        h = image['height']
        w = image['width']
        idx = image['id']

        # Load image and make sure it is RGB
        img = Image.open(data['paths']['img_dir'] + name)       
        if img.mode != 'RGB':
            img = make_rgb(img)

        # Get targets
        target_anno = sio.loadmat(data['paths']['anno_dir'] + os.path.splitext(name)[0] + '.mat')['record']
        target_boxes = get_target(target_anno)        

        # Get prediction boxes
        indices = [(box['image_id'] == idx) for box in data['pred_boxes']]
        pred_boxes = [i for indx,i in enumerate(data['pred_boxes']) if indices[indx] == True]
        pred_boxes = [[box['bbox'][0], box['bbox'][1], box['bbox'][0] + box['bbox'][2], box['bbox'][1] + box['bbox'][3]] for box in pred_boxes]

        # Go through all target boxes
        seg_box_iou = 0  
        if len(pred_boxes):
            for target_box in target_boxes:
                t_bbox = target_box['bbox']

                # Find the best prediction box
                pred_box_ious = [find_bbox_iou(t_bbox, pred_box) for pred_box in pred_boxes]
                if max(pred_box_ious) > 0.5:
                    best_box_index = pred_box_ious.index(max(pred_box_ious))
                else:
                    best_box_index = None

                # Find IoU between target mask and best prediction mask
                if best_box_index is not None:
                    # Get target mask
                    path = "/home/teo/storage/Data/Masks/car_combined/single/" + os.path.splitext(name)[0] + '_mask_' + str(target_box['number']) + '.mat'
                    t_mask = torch.from_numpy(sio.loadmat(path)['single_mask'])                   
                    x1, y1, x2, y2 = int(t_bbox[0]), int(t_bbox[1]), int(t_bbox[2]), int(t_bbox[3])
                    target_snippet = torch.zeros_like(t_mask)
                    target_snippet[y1:y2, x1:x2] = t_mask[y1:y2, x1:x2]
                    t_color_mask = multi_to_color(target_snippet)
                        
                    # Get prediction
                    pred_box = pred_boxes[best_box_index]
                    px1, py1, px2, py2 = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
                    crop = img.crop((int(px1), int(py1), int(px2), int(py2)))
                    crop = data['transforms'](crop)
                    crop = crop.to('cuda:0', dtype=torch.float)
                    crop = crop.unsqueeze(0)
                    output = data['model'](crop) # Run through the model
                    # Bilinear interpolation of output
                    output = torch.nn.functional.interpolate(output['out'], size=[py2-py1, px2-px1], mode='bilinear')
                    _, pred = torch.max(output.squeeze(0), 0)                
                    pred_snippet = torch.zeros((h, w))
                    pred_snippet[py1:py2, px1:px2] = pred # Get a mask with the size of the image
                    p_color_mask = multi_to_color(pred_snippet)

                    # Compute IoU without background class
                    iou = find_mask_iou(pred_snippet.long(), target_snippet.long(), n_classes=9)

                    writer.add_image("target_" + name, t_color_mask, count)
                    writer.add_image("pred_" + name, p_color_mask, count)
                    count += 1
                else:
                    path = "/home/teo/storage/Data/Masks/car_combined/single/" + os.path.splitext(name)[0] + '_mask_' + str(target_box['number']) + '.mat'
                    t_mask = torch.from_numpy(sio.loadmat(path)['single_mask'])                   
                    x1, y1, x2, y2 = int(t_bbox[0]), int(t_bbox[1]), int(t_bbox[2]), int(t_bbox[3])
                    target_snippet = torch.zeros_like(t_mask)
                    target_snippet[y1:y2, x1:x2] = t_mask[y1:y2, x1:x2]
                    t_color_mask = multi_to_color(target_snippet)
                    writer.add_image("target_" + name, t_color_mask , count)
                    count += 1
                    iou = 0
                
                print(iou)
                seg_box_iou += iou

        seg_box_iou /= len(target_boxes)
        img_iou += seg_box_iou

    img_iou /= len(data['img_info'])
    writer.close()

    print(img_iou)
    print('Done!')

if __name__ == '__main__':
    main()