import argparse
import json
import glob
import os
from PIL import Image, ImageDraw
import numpy as np
from tensorboardX import SummaryWriter
from skimage import measure
from shapely.geometry import Polygon
from tqdm import tqdm


# # Transform Pascal3D+ to CoCo format

def create_sub_masks(msk_path, writer):
    mask = np.genfromtxt(msk_path, delimiter=',', dtype=np.uint8)
    mask_image = Image.fromarray(mask)

    #writer.add_image('mask', np.expand_dims(mask_image, axis=0)*30, 1)

    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by 8 colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get pixel value
            pixel = mask_image.getpixel((x,y))

            # If the pixel is not black...
            if pixel != 0:
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask, image_id, category, annotation_id, is_crowd, writer):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    print(len(contours))

    annotation_list = []
    for j in range(len(contours)):
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        contour = contours[j]
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        #poly = poly.simplify(1, preserve_topology=True)
        segmentation = np.expand_dims(np.array(poly.exterior.coords).ravel().tolist(), axis=0).tolist()

        # Visualize segmentations
        image = Image.new("L", (sub_mask.size[0], sub_mask.size[1]))
        draw = ImageDraw.Draw(image)
        draw.polygon((segmentation[0]), fill=200)
        image = np.asarray(image)
        #writer.add_image('sub_seg', np.expand_dims(image, axis=0), annotation_id)

        # Calculate the bounding box and area
        x, y, max_x, max_y = poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = poly.area

        print(type(area))
        print(type(bbox))
        exit()

        annotation = {
            'segmentation': segmentation,
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category,
            'id': annotation_id,
            'bbox': bbox,
            'area': area
        }

        annotation_list.append(annotation)
        annotation_id += 1

    return annotation_list, annotation_id

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

    writer = SummaryWriter(logdir='/home/teo/storage/Code/3D_Object_Pose_Recovery/Python/scripts/runs/name', comment='_test_coco_trans')
    
    split = ['train', 'val', 'test']
    
    # Create Coco Style Dataset
    for phase in split:
        images = []
        annotations = []
        an_id = 1

        img_list = sorted(glob.glob(os.path.join(args.image_dir, phase) + '/*'))
        file_names = [os.path.splitext(os.path.basename(x))[0] for x in img_list]
        msk_list = [os.path.join(args.mask_dir, phase, x + "_mask.csv") for x in file_names]

        for i in tqdm(range(len(img_list))):
            img_desc = create_img_desc(img_list[i], i)
            submasks = create_sub_masks(msk_list[i], writer)
            for color, sub_mask in submasks.items():
                annotation_list, an_id = create_sub_mask_annotation(sub_mask, i, int(color), an_id, 0, writer)
                #writer.add_image('submask', np.expand_dims(np.asarray(sub_mask), axis=0), an_id)
                annotations.extend(annotation_list)
            exit()
            images.append(img_desc)
        
        #writer.close()

        info = {
            "description": "Pascal3D+ Car Dataset",
            "url": "http://cvgl.stanford.edu/projects/pascal3d.html",
            "version": "1.0",
            "year": 2019,
            "contributor": "Teodor Totev",
            "date_created": "2019/12/06"
        }

        licenses = {}

        categories = [
            {"supercategory": "car_part", "id": 1, "name": "FBR"},
            {"supercategory": "car_part", "id": 2, "name": "FBL"},
            {"supercategory": "car_part", "id": 3, "name": "BBR"},
            {"supercategory": "car_part", "id": 4, "name": "BBL"},
            {"supercategory": "car_part", "id": 5, "name": "FTR"},
            {"supercategory": "car_part", "id": 6, "name": "FTL"},
            {"supercategory": "car_part", "id": 7, "name": "BTR"},
            {"supercategory": "car_part", "id": 8, "name": "BTL"},
        ]

        annotation_file = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": images,
            "annotations": annotations
        }

        out_file = os.path.join(args.anno_dir, phase) + '/test_anno.json'
        with open(out_file, 'w') as outfile:
            json.dump(annotation_file, outfile, indent=4)

if __name__ == '__main__':

    # Define parser for input arguments
    parser = argparse.ArgumentParser(description='Convert Pascal3D to Coco Style')
    parser.add_argument('--image_dir',  '-im',  type=str, default='/home/teo/storage/Data/Images/car_combined')
    parser.add_argument('--mask_dir',   '-ma',  type=str, default='/home/teo/storage/Data/Masks/car_combined')
    parser.add_argument('--anno_dir',   '-an',  type=str, default='/home/teo/storage/Data/Annotations/car_combined')
    args = parser.parse_args()

    # Run the conversion
    main()
