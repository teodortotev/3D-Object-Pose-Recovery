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

def enumerate2(xs, start=0, step=1):
    for x in xs:
        yield (start, x)
        start += step

def find_mask_pixacc(pred, label, n_classes):
    pred = pred.view(-1)
    label = label.view(-1)
    for cat in range(1,9):
        corrects += torch.sum(pred == label)
    pixacc = float(corrects) / float(pred.shape[0])

    return pixacc

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
    for cat in range(1,9):
        index_mask = mask == cat
        color_mask[0,index_mask] = colors[cat-1][0]
        color_mask[1,index_mask] = colors[cat-1][1]
        color_mask[2,index_mask] = colors[cat-1][2]
    return color_mask

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

def prob_to_multi(mask, threshold):
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

    # car_mask = (mask >= threshold).type(torch.float32)
    # multi_mask = mask * car_mask
    # values, indices = torch.max(multi_mask, dim=0)
    # values_mask = (values != 0).type(torch.int64)
    # multi_mask = (indices + 1) * values_mask
    values, indices = torch.max(mask, dim=0)

    return indices.type(torch.float)

def binary_to_multi(mask):

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

    multi_mask = torch.zeros((mask.shape[1], mask.shape[2]))
    color_mask = torch.zeros((3, mask.shape[1], mask.shape[2]))
    for cat in range(mask.shape[0]):
        index_mask = mask[cat, :, :] != 0
        multi_mask[index_mask] = cat + 1
        color_mask[0,index_mask] = colors[cat][0]
        color_mask[1,index_mask] = colors[cat][1]
        color_mask[2,index_mask] = colors[cat][2]
    return multi_mask, color_mask

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

def rectangle_perimeter(x1, y1, width, height, shape=None, clip=False):
    rr, cc = [x1, x1 + width, x1 + width, x1], [y1, y1, y1 + height, y1 + height]

    return Draw.polygon_perimeter(rr, cc, shape=shape, clip=clip)

def compute_on_dataset(model, data_loader, device, bbox_aug, writer, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for iteration, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))

            # colors = np.array([[0.954174456379543, 0.590608652919636, 0.281507695118553],
            #         [0.0319226295039784, 0.660437966312602, 0.731050829723742],
            #         [0.356868986182542, 0.0475546731138661, 0.137762892519516],
            #         [0.662653834287215, 0.348784808510059, 0.836722781749718],
            #         [0.281501559148491, 0.451340580355743, 0.138601715742360],
            #         [0.230383067317464, 0.240904997120111, 0.588209385389494],
            #         [0.711128551180325, 0.715045013296177, 0.366156800454938],
            #         [0.624572916993309, 0.856182292006288, 0.806759544661106],
            #         [0.424572916993309, 0.556182292006288, 0.306759544661106],
            #         [0.324572916993309, 0.256182292006288, 0.906759544661106]])

            # # Display targets
            # image = images.tensors[0].cpu().numpy()
            # means = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
            # means[0] = 102.9801
            # means[1] = 115.9465
            # means[2] = 122.7717
            # image = image + means
            # image = image[[2, 1, 0]].astype(np.uint8)
            # image1 = copy.deepcopy(image)
            # image2 = copy.deepcopy(image)

            # for b in range(len(targets[0].bbox)):
            #     box = targets[0].bbox[b]
            #     label = int(targets[0].extra_fields['labels'][b])
            #     x1 = np.around(box[0].cpu().numpy())
            #     y1 = np.around(box[1].cpu().numpy())
            #     x2 = np.around(box[2].cpu().numpy())
            #     y2 = np.around(box[3].cpu().numpy())
            #     rr, cc = rectangle_perimeter(y1, x1, y2-y1, x2-x1)
            #     image1[0, rr, cc] = colors[label-1][0]*255
            #     image1[1, rr, cc] = colors[label-1][1]*255
            #     image1[2, rr, cc] = colors[label-1][2]*255

            # writer.add_image('target', image1, iteration)

            # # Display target masks
            # masks = targets[0].get_field('masks')
            # for m in range(len(masks)):
            #     mask = masks[m].get_mask_tensor()
            #     combined_mask = mask[0, :, :]
            #     color_mask = torch.zeros((3, mask.shape[1], mask.shape[2]))
            #     for cat in range(1,8):
            #         combined_mask = combined_mask + mask[cat, :, :]*(cat+1)
            #     for s1 in range(combined_mask.shape[0]):
            #         for s2 in range(color_mask.shape[1]):
            #             idx = int(combined_mask[s1, s2])
            #             if idx != 0:
            #                 color_mask[0, s1, s2] = colors[idx-1][0]*255
            #                 color_mask[1, s1, s2] = colors[idx-1][1]*255
            #                 color_mask[2, s1, s2] = colors[idx-1][2]*255
                
            #     writer.add_image('mask_' + str(m), color_mask, iteration)


            # combined_mask = masks[0, :, :]
            # for i in range(1,8):
            #     combined_mask = combined_mask | masks[i, :, :]
            # writer.add_image('mask', combined_mask.unsqueeze(0)*255, iteration)
            # writer.add_image('single part 2', masks[1, :, :].unsqueeze(0)*255, iteration)


            # exp_n_boxes = len(targets[0].bbox)
            # order = torch.argsort(output[0].extra_fields['scores'], descending=True)

            # # Display prediction bboxes
            # for b in range(len(output[0].bbox)):
            #     # if b >= len(order):
            #     #     break
            #     box = output[0].bbox[order[b]]
            #     label = int(output[0].extra_fields['labels'][b])
            #     if output[0].extra_fields['scores'][b] < 0.5:
            #         continue
            #     x1 = np.around(box[0].cpu().numpy())
            #     y1 = np.around(box[1].cpu().numpy())
            #     x2 = np.around(box[2].cpu().numpy())
            #     y2 = np.around(box[3].cpu().numpy())
            #     rr, cc = rectangle_perimeter(y1, x1, y2-y1, x2-x1)
            #     image2[0, rr, cc] = colors[label-1][0]*255
            #     image2[1, rr, cc] = colors[label-1][1]*255
            #     image2[2, rr, cc] = colors[label-1][2]*255


            # writer.add_image('pred', image2, iteration)

            # writer.add_image('image', image, iteration)

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

    # count = 0
    # for image_id, prediction in enumerate(predictions):
    #     element = dataset.__getitem__(image_id)
    #     image = element[0]
    #     target_masks = element[1].extra_fields['masks']
    #     img_info = dataset.get_img_info(image_id)
    #     image_width = img_info["width"]
    #     image_height = img_info["height"]
    #     image = interpolate(image.unsqueeze(0), size=(image_height, image_width), mode='bilinear')
    #     image = image.squeeze(0)
    #     image = image.cpu().numpy()
    #     means = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    #     means[0] = 102.9801
    #     means[1] = 115.9465
    #     means[2] = 122.7717
    #     image = image + means
    #     image = image[[2, 1, 0]].astype(np.uint8)
    #     prediction = prediction.resize((image_width, image_height))
    #     masks = prediction.get_field("mask")

    #     # Combine masks using max operation and storing the part class
    #     for mask_id in range(masks.shape[0]):
    #         box = prediction.bbox[mask_id]
    #         label = prediction.extra_fields['labels'][mask_id]
    #         mask = masks[mask_id]
    #         mask_width = int(box[2] - box[0])
    #         mask_height = int(box[3] - box[1])
    #         new_mask = torch.zeros((2, mask[0].shape[0], mask[0].shape[1]))
            
    #         for i in range(mask[0].shape[0]):
    #             for j in range(mask[0].shape[1]):
    #                 max_value, index = torch.max(mask[:, i, j], dim=0)
    #                 new_mask[0, i, j] = max_value
    #                 new_mask[1, i, j] = index + 1

    #         # Interpolate values (billinear) and part classes (nearest) according to image size
    #         threshold = 0.5
    #         color_mask = torch.zeros((3, mask_height, mask_width), dtype=torch.float64)
    #         ip_mask = torch.zeros((2, mask_height, mask_width)).unsqueeze(0).unsqueeze(0)
    #         ip_mask[0,0,0,:,:] = interpolate(new_mask[0, :, :].unsqueeze(0).unsqueeze(0), size=(mask_height, mask_width), mode='bilinear')
    #         ip_mask[0,0,1,:,:] = interpolate(new_mask[1, :, :].unsqueeze(0).unsqueeze(0), size=(mask_height, mask_width), mode='nearest')
    #         ip_mask = ip_mask.squeeze(0).squeeze(0)
    #         ip_mask[0, :, :] = ip_mask[0, :, :] > threshold
    #         final_mask = ip_mask[0, :, :] * ip_mask[1, :, :]

    #         for s1 in range(final_mask.shape[0]):
    #             for s2 in range(final_mask.shape[1]):
    #                 cat = int(final_mask[s1, s2])
    #                 if cat != 0:
    #                     color_mask[0, s1, s2] = colors[cat-1][0]
    #                     color_mask[1, s1, s2] = colors[cat-1][1]
    #                     color_mask[2, s1, s2] = colors[cat-1][2] 


    #         # Add bounding box prediction
    #         image1 = copy.deepcopy(image)
    #         x1 = np.floor(box[0].cpu().numpy())
    #         y1 = np.floor(box[1].cpu().numpy())
    #         x2 = np.floor(box[2].cpu().numpy())
    #         y2 = np.floor(box[3].cpu().numpy())
    #         rr, cc = rectangle_perimeter(y1, x1, y2-y1, x2-x1)
    #         image1[0, rr, cc] = colors[label-1][0]*255
    #         image1[1, rr, cc] = colors[label-1][1]*255
    #         image1[2, rr, cc] = colors[label-1][2]*255

    #         writer.add_image('pred_bbox' + str(count), image1, count)
    #         # writer.add_image('pred_mask' + str(count), final_mask.unsqueeze(0)*30, count)
    #         writer.add_image('pred_mask' + str(count), color_mask, count)
    #         count = count + 1

    # test_image = torch.zeros((3,200,200), dtype=torch.float64)
    # for a in range(10):
    #     test_image[0, :, :] = colors[a][0]
    #     test_image[1, :, :] = colors[a][1]
    #     test_image[2, :, :] = colors[a][2]
    #     writer.add_image('test_IMAGE', test_image, )
    # writer.close()

def evaluate_results(dataset, predictions, writer):
    img_iou = 0
    count = 0
    for idx, pred in tqdm(enumerate(predictions)):
        # Get image information
        info = dataset.get_img_info(idx)
        img_width = info["width"]
        img_height = info["height"]

        # Get ground truth data
        gt = dataset.__getitem__(idx)
        gt = gt[1].resize((img_width, img_height))
        gt_masks = [segment.get_mask_tensor() for segment in gt.extra_fields["masks"]]
        gt_labels = gt.extra_fields['labels']        
        
        # For each ground truth box and mask 
        seg_box_iou = 0
        if len(pred.bbox):
            for target_box, target_mask, target_label in zip(gt.bbox, gt_masks, gt_labels):

                # Resize prediction bbox
                pred = pred.resize((img_width, img_height))

                # Find best fit bbox prediction if exists
                pred_box_ious = [find_bbox_iou(target_box.tolist(), pred_box.tolist()) for pred_box in pred.bbox]
                if max(pred_box_ious) > 0.5:
                    best_box_index = pred_box_ious.index(max(pred_box_ious))
                else:
                    best_box_index = None

                # Find IoU between target mask and best prediction mask
                if best_box_index is not None:
                    # Get target mask
                    x1, y1, x2, y2 = int(np.around(target_box[0])), int(np.around(target_box[1])), int(np.around(target_box[2])), int(np.around(target_box[3]))
                    target_snippet = torch.zeros_like(target_mask)
                    target_snippet[:, y1:y2, x1:x2] = target_mask[:, y1:y2, x1:x2]
                    target_snippet, t_color_mask = binary_to_multi(target_snippet)
                    
                    # Get predicted mask
                    pred_box = pred.bbox[best_box_index]
                    px1, py1, px2, py2 = int(np.around(pred_box[0])), int(np.around(pred_box[1])), int(np.around(pred_box[2])), int(np.around(pred_box[3]))
                    pred_label = pred.extra_fields['labels'][best_box_index]
                    pred_mask = pred.extra_fields['mask'][best_box_index] # Get mask corresponding to best box

                    pred_mask = torch.nn.functional.interpolate(pred_mask.unsqueeze(0), size=[py2-py1, px2-px1], mode='bilinear') # interpolate to bbox dimensions
                    interp_mask = prob_to_multi(pred_mask.squeeze(0), threshold=0.5)
                    
                    pred_snippet = torch.zeros((img_height, img_width))
                    pred_snippet[py1:py2, px1:px2] = interp_mask # Get a mask with the size of the image
                    p_color_mask = multi_to_color(pred_snippet)

                    # Compute IoU without background class
                    iou = find_mask_iou(pred_snippet.long(), target_snippet.long(), n_classes=9)

                    writer.add_image("target_" + info['file_name'], t_color_mask, count)
                    writer.add_image("pred_" + info['file_name'], p_color_mask, count)
                    count += 1
                else:
                    # Get target mask
                    x1, y1, x2, y2 = int(np.around(target_box[0])), int(np.around(target_box[1])), int(np.around(target_box[2])), int(np.around(target_box[3]))
                    target_snippet = torch.zeros_like(target_mask)
                    target_snippet[:, y1:y2, x1:x2] = target_mask[:, y1:y2, x1:x2]
                    target_snippet, t_color_mask = binary_to_multi(target_snippet)
                    writer.add_image("target_" + info["file_name"], t_color_mask , count)
                    count += 1
                    iou = 0

                # print(iou)
                seg_box_iou += iou

        seg_box_iou /= len(gt.bbox)
        img_iou += seg_box_iou

    img_iou /= len(predictions)
    writer.close()

    return img_iou

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
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, writer, inference_timer)
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

    ## Uncomment for data visualisations
    # visualize_output(dataset=dataset, predictions=predictions, writer=writer)

    # Compute APs
    # mIoU = evaluate_results(dataset=dataset, predictions=predictions, writer=writer)
    # print(mIoU)
    # exit()

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