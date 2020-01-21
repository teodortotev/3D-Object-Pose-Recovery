import argparse
import json
import glob
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import scipy.io as sio


# # Transform Pascal3D+ to CoCo format
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
            else:
                bbox = record['objects'][0][0][0][a][8][0][0]
                x = bbox[0][0][0].tolist()
                y = bbox[1][0][0].tolist()
                w = (bbox[2][0][0]-bbox[0][0][0]).tolist()
                h = (bbox[3][0][0]-bbox[1][0][0]).tolist()
                category = record['objects'][0][0][0][a][18][0][0].tolist()
            bbox = (x, y, w, h)
            area = int(bbox[2]) * int(bbox[3])

            annotation = {
                'iscrowd': 0,
                'image_id': img_id,
                'category_id': category,
                'id': an_id,
                'bbox': bbox,
                'area': area
            }

            an_id+=1
            annotation_list.append(annotation)

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

        for i in tqdm(range(len(img_list))):
            img_desc = create_img_desc(img_list[i], i)
            img_annotations, an_id = read_mat(anno_list[i], i, an_id)
            annotations.extend(img_annotations)
            images.append(img_desc)

        info = {
            "description": "Pascal3D+ Car Dataset Object Detections",
            "url": "http://cvgl.stanford.edu/projects/pascal3d.html",
            "version": "1.0",
            "year": 2019,
            "contributor": "Teodor Totev",
            "date_created": "2020/01/20"
        }

        licenses = {}

        categories = [
            {"supercategory": "car_type", "id": 1, "name": "CAD1"},
            {"supercategory": "car_type", "id": 2, "name": "CAD2"},
            {"supercategory": "car_type", "id": 3, "name": "CAD3"},
            {"supercategory": "car_type", "id": 4, "name": "CAD4"},
            {"supercategory": "car_type", "id": 5, "name": "CAD5"},
            {"supercategory": "car_type", "id": 6, "name": "CAD6"},
            {"supercategory": "car_type", "id": 7, "name": "CAD7"},
            {"supercategory": "car_type", "id": 8, "name": "CAD8"},
            {"supercategory": "car_type", "id": 9, "name": "CAD9"},
            {"supercategory": "car_type", "id": 10, "name": "CAD10"}
        ]

        annotation_file = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": images,
            "annotations": annotations
        }

        out_file = os.path.join(args.anno_dir) + '/' + str(phase) + '_test_anno.json'
        with open(out_file, 'w') as outfile:
            json.dump(annotation_file, outfile, indent=4)

if __name__ == '__main__':

    # Define parser for input arguments
    parser = argparse.ArgumentParser(description='Convert Pascal3D to Coco Style')
    parser.add_argument('--image_dir',  '-im',  type=str, default='/home/teo/storage/Data/Images/car_combined')
    parser.add_argument('--anno_dir',   '-ma',  type=str, default='/home/teo/storage/Data/Annotations/car_combined/car_objects')
    args = parser.parse_args()

    # Run the conversion
    main()
