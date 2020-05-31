import argparse
import json
import glob
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import scipy.io as sio
from scipy import spatial
from skimage import measure
from shapely.geometry import Polygon


# # Transform Pascal3D+ to CoCo format
def create_segmentation_list(mask, num_classes):

    index_list = []
    segmentation_list = []

    # Get indices of different classes
    for c in range(num_classes):
        idx = np.argwhere(mask == c+1)
        index_list.append(idx)
    
    # Generate mask for each class
    for c in range(num_classes):
        single_mask = np.zeros((mask.shape))
        idx = index_list[c]
        for p in idx:
            single_mask[p[0], p[1]] = 1
        contours = measure.find_contours(single_mask, 0.5, positive_orientation='low')
        for contour in contours:
            poly = Polygon(contour)
            segmentation = np.expand_dims(np.array(poly.exterior.coords).ravel().tolist(), axis=0).tolist()
            segment = {
                'class': c+1,
                'segment': segmentation, 
            }
            segmentation_list.append(segment)

    return segmentation_list

def create_keypoint_list(keypoints):
    keypoint_list = []
    num_keypoints = 0
    for keypoint in keypoints:
        if int(keypoint[0][0][1]) != 1:
            keypoint_list.extend([0,0,0])
        else:
            x, y = keypoint[0][0][0][0]
            keypoint_list.extend([x, y, 2])
            num_keypoints += 1
    return keypoint_list, num_keypoints 

def read_mat(mat, img_id, an_id):
    # Read the .mat file
    img_mat = sio.loadmat(mat)
    record = img_mat['record']
    img_name = record['filename'][0][0][0]

    annotation_list = []

    # Go through all object annotations in an image
    for a in range(len(record['objects'][0][0][0])):
        clas = record['objects'][0][0][0][a][0][0]
        if clas == 'car':
            if 'n' in img_name:
                bbox = record['objects'][0][0][0][a][1][0]
                x = bbox[0].tolist()
                y = bbox[1].tolist()
                w = (bbox[2]-bbox[0]).tolist()
                h = (bbox[3]-bbox[1]).tolist()
                category = record['objects'][0][0][0][a][4][0][0].tolist()
                keypoints = record['objects'][0][0][0][a][2][0][0]
            else:
                bbox = record['objects'][0][0][0][a][8][0][0]
                x = bbox[0][0][0].tolist()
                y = bbox[1][0][0].tolist()
                w = (bbox[2][0][0]-bbox[0][0][0]).tolist()
                h = (bbox[3][0][0]-bbox[1][0][0]).tolist()
                category = record['objects'][0][0][0][a][18][0][0].tolist()
                keypoints = record['objects'][0][0][0][a][16][0][0]
            bbox = (x, y, w, h)
            area = int(bbox[2]) * int(bbox[3])

            path = args.single_mask_dir + '/' + os.path.splitext(img_name)[0] + '_mask_' + str(a+1) + '.mat'

            if os.path.exists(path):
                mask = sio.loadmat(path)
                mask = mask['single_mask']
                mask = np.transpose(mask)
                segmentation_list = create_segmentation_list(mask, 8)
                keypoint_list, num_keypoints = create_keypoint_list(keypoints)

                annotation = {
                    'iscrowd': 0,
                    'image_id': img_id,
                    'category_id': category,
                    'id': an_id,
                    'bbox': bbox,
                    'area': area,
                    'segmentation': segmentation_list,
                    'keypoints' : keypoint_list,
                    'num_keypoints' : num_keypoints,
                }

                an_id+=1
                annotation_list.append(annotation)
            else:
                print('The following path does not exist:')
                print(path)

    return annotation_list, an_id

def create_img_desc(img_path, img_id):
    name = os.path.basename(img_path)
    img = Image.open(img_path)
    width = img.size[0]
    height = img.size[1]
    img_desc = {
        "file_name": name,
        "height": height,
        "width": width,
        "id": img_id
    }
    return img_desc

def main():
    
    split = ['train', 'val', 'test']
    
    # Create Coco Style Dataset
    for phase in split:
        images = []
        annotations = []
        an_id = 0

        img_list = sorted(glob.glob(os.path.join(args.image_dir, phase) + '/*'))
        file_names = [os.path.splitext(os.path.basename(x))[0] for x in img_list]
        anno_list = [os.path.join(args.anno_dir, x + ".mat") for x in file_names]

        # for i in tqdm(range(len(img_list))):
        for i in tqdm(range(5,10)):
            img_desc = create_img_desc(img_list[i], i)
            img_annotations, an_id = read_mat(anno_list[i], i, an_id)
            annotations.extend(img_annotations)
            images.append(img_desc)

        info = {
            "description": "Pascal3D+ Car Parts Dataset Object Detection",
            "url": "http://cvgl.stanford.edu/projects/pascal3d.html",
            "version": "1.0",
            "year": 2020,
            "contributor": "Teodor Totev",
            "date_created": "2020/02/18"
        }

        licenses = {}

        keypoint_names = ['left_front_wheel', 'left_back_wheel', 'right_front_wheel', 'right_back_wheel', 'upper_left_windshield', 'upper_right_windshield', 'upper_left_rearwindow', 'upper_right_rearwindow', 'left_front_light', 'right_front_light', 'left_back_trunk', 'right_back_trunk']

        categories = [
            {"supercategory": "CAD1", "id": 1, "name": "CAD1", "keypoints": keypoint_names, "skeleton": []},
            {"supercategory": "CAD2", "id": 2, "name": "CAD2", "keypoints": keypoint_names, "skeleton": []},
            {"supercategory": "CAD3", "id": 3, "name": "CAD3", "keypoints": keypoint_names, "skeleton": []},
            {"supercategory": "CAD4", "id": 4, "name": "CAD4", "keypoints": keypoint_names, "skeleton": []},
            {"supercategory": "CAD5", "id": 5, "name": "CAD5", "keypoints": keypoint_names, "skeleton": []},
            {"supercategory": "CAD6", "id": 6, "name": "CAD6", "keypoints": keypoint_names, "skeleton": []},
            {"supercategory": "CAD7", "id": 7, "name": "CAD7", "keypoints": keypoint_names, "skeleton": []},
            {"supercategory": "CAD8", "id": 8, "name": "CAD8", "keypoints": keypoint_names, "skeleton": []},
            {"supercategory": "CAD9", "id": 9, "name": "CAD9", "keypoints": keypoint_names, "skeleton": []},
            {"supercategory": "CAD10", "id": 10, "name": "CAD10", "keypoints": keypoint_names, "skeleton": []}
        ]

        partcategories = [
            {"supercategory": "FBR", "id": 1, "name": "FBR"},
            {"supercategory": "FBL", "id": 2, "name": "FBL"},
            {"supercategory": "BBR", "id": 3, "name": "BBR"},
            {"supercategory": "BBL", "id": 4, "name": "BBL"},
            {"supercategory": "FTR", "id": 5, "name": "FTR"},
            {"supercategory": "FTL", "id": 6, "name": "FTL"},
            {"supercategory": "BTR", "id": 7, "name": "BTR"},
            {"supercategory": "BTL", "id": 8, "name": "BTL"},
        ]

        annotation_file = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "partcategories": partcategories,
            "images": images,
            "annotations": annotations
        }

        out_file = os.path.join(args.save_dir) + '/' + str(phase) + '_subset_keypoint_anno.json'
        with open(out_file, 'w') as outfile:
            json.dump(annotation_file, outfile, indent=4)

        # out_file = os.path.join(args.save_dir) + '/' + str(phase) + 'keypoint_anno.json'
        # with open(out_file, 'w') as outfile:
        #     json.dump(annotation_file, outfile, indent=4)

if __name__ == '__main__':

    # Define parser for input arguments
    parser = argparse.ArgumentParser(description='Convert Pascal3D to Coco Style')
    parser.add_argument('--image_dir',  '-im',  type=str, default='/home/teo/storage/Data/Images/car_combined')
    parser.add_argument('--anno_dir',   '-ma',  type=str, default='/home/teo/storage/Data/Annotations/car_combined/car_objects')
    parser.add_argument('--save_dir',   '-sd',  type=str, default='/home/teo/storage/Data/Annotations/car_combined/car_parts')
    parser.add_argument('--single_mask_dir',   '-smd',  type=str, default='/home/teo/storage/Data/Masks/car_combined/single')
    args = parser.parse_args()

    # Run the conversion
    main()
