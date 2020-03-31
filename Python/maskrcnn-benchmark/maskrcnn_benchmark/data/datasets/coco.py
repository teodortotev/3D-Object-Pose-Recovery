# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.partcategories = {part['id']: part['name'] for part in self.coco.parts.values()}

        self.json_partcategory_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getPartIds())
        }
        self.contiguous_partcategory_id_to_json_id = {
            v: k for k, v in self.json_partcategory_id_to_contiguous_id.items()
        }
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            # Get dictionary of part segmentations for each object
            segmentations = [obj["segmentation"] for obj in anno]

            # Accumulate classes
            partclasses = []
            for i in range(len(segmentations)): 
                partclass = [obj['class'] for obj in segmentations[i]]
                partclass = [self.json_partcategory_id_to_contiguous_id[p] for p in partclass]
                partclass = torch.tensor(partclass)
                partclasses.append(partclass)

            # Accumulate masks
            masks = []
            for i in range(len(segmentations)):
                mask = [obj["segment"] for obj in segmentations[i]]
                mask = SegmentationMask(mask, img.size, mode='poly')
                masks.append(mask)

            # Merge all masks belonging to the same part class
            new_masks = []
            for msk, pcls in zip(masks, partclasses):
                segments = msk.get_mask_tensor()
                new_segments = torch.zeros((len(self.partcategories), segments.size()[1], segments.size(2)), dtype=torch.uint8)
                for partcat in range(len(self.partcategories)):
                    for n_poly in range(pcls.size()[0]):
                        if int(pcls[n_poly]) == (partcat + 1):
                            new_segments[partcat, :, :] = new_segments[partcat, :, :] | segments[n_poly, :, :]
                new_mask = SegmentationMask(new_segments, img.size, mode='mask')
                new_masks.append(new_mask)
            
            new_partclass = [self.json_partcategory_id_to_contiguous_id[p+1] for p in range(len(self.partcategories))]
            new_partclass = torch.tensor(new_partclass)

            new_partclasses = []
            for a in range(len(new_masks)):
                new_partclasses.append(new_partclass)

            target.add_field('partlabels', new_partclasses)
            target.add_field('masks', new_masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
