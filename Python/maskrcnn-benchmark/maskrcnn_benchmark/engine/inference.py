# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import numpy as np
from skimage import draw as Draw
import copy

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

from maskrcnn_benchmark.layers.misc import interpolate

def rectangle_perimeter(x1, y1, width, height, shape=None, clip=False):
    rr, cc = [x1, x1 + width, x1 + width, x1], [y1, y1, y1 + height, y1 + height]

    return Draw.polygon_perimeter(rr, cc, shape=shape, clip=clip)

def compute_on_dataset(model, data_loader, device, bbox_aug, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):

    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

def visualize_output(dataset, predictions, writer):
    count = 0
    for image_id, prediction in enumerate(predictions):
        image = dataset.__getitem__(image_id)[0]
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        image = interpolate(image.unsqueeze(0), size=(image_height, image_width), mode='bilinear')
        image = image.squeeze(0)
        image = image.cpu().numpy()
        means = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        means[0] = 102.9801
        means[1] = 115.9465
        means[2] = 122.7717
        image = image + means
        image = image[[2, 1, 0]].astype(np.uint8)
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")

        # Combine masks using max operation and storing the part class
        for mask_id in range(masks.shape[0]):
            box = prediction.bbox[mask_id]
            mask = masks[mask_id]
            mask_width = int(box[2] - box[0])
            mask_height = int(box[3] - box[1])
            new_mask = torch.zeros((2, mask[0].shape[0], mask[0].shape[1]))
            
            for i in range(mask[0].shape[0]):
                for j in range(mask[0].shape[1]):
                    max_value, index = torch.max(mask[:, i, j], dim=0)
                    new_mask[0, i, j] = max_value
                    new_mask[1, i, j] = index + 1

            # Interpolate values (billinear) and part classes (nearest) according to image size
            threshold = 0.5
            ip_mask = torch.zeros((2, mask_height, mask_width)).unsqueeze(0).unsqueeze(0)
            ip_mask[0,0,0,:,:] = interpolate(new_mask[0, :, :].unsqueeze(0).unsqueeze(0), size=(mask_height, mask_width), mode='bilinear')
            ip_mask[0,0,1,:,:] = interpolate(new_mask[1, :, :].unsqueeze(0).unsqueeze(0), size=(mask_height, mask_width), mode='nearest')
            ip_mask = ip_mask.squeeze(0).squeeze(0)
            ip_mask[0, :, :] = ip_mask[0, :, :] > threshold
            final_mask = ip_mask[0, :, :] * ip_mask[1, :, :]

            # Add bounding box prediction
            image1 = copy.deepcopy(image)
            x1 = np.floor(box[0].cpu().numpy())
            y1 = np.floor(box[1].cpu().numpy())
            x2 = np.floor(box[2].cpu().numpy())
            y2 = np.floor(box[3].cpu().numpy())
            rr, cc = rectangle_perimeter(y1, x1, y2-y1, x2-x1)
            image1[:, rr, cc] = 255

            writer.add_image('image bbox', image1, count)
            writer.add_image('mask', final_mask.unsqueeze(0)*30, count)
            count = count + 1
    
def inference(
        model,
        data_loader,
        writer, 
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    visualize_output(dataset=dataset, predictions=predictions, writer=writer)

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
