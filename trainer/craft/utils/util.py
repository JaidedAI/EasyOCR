from collections import OrderedDict
import os

import cv2
import numpy as np

from data import imgproc
from utils import craft_utils


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def saveInput(
    imagename, vis_dir, image, region_scores, affinity_scores, confidence_mask
):
    image = np.uint8(image.copy())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes, polys = craft_utils.getDetBoxes(
        region_scores, affinity_scores, 0.85, 0.2, 0.5, False
    )

    if image.shape[0] / region_scores.shape[0] >= 2:
        boxes = np.array(boxes, np.int32) * 2
    else:
        boxes = np.array(boxes, np.int32)

    if len(boxes) > 0:
        np.clip(boxes[:, :, 0], 0, image.shape[1])
        np.clip(boxes[:, :, 1], 0, image.shape[0])
        for box in boxes:
            cv2.polylines(image, [np.reshape(box, (-1, 1, 2))], True, (0, 0, 255))
    target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores)
    target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores)
    confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)

    # overlay
    height, width, channel = image.shape
    overlay_region = cv2.resize(target_gaussian_heatmap_color, (width, height))
    overlay_aff = cv2.resize(target_gaussian_affinity_heatmap_color, (width, height))
    confidence_mask_gray = cv2.resize(
        confidence_mask_gray, (width, height), interpolation=cv2.INTER_NEAREST
    )
    overlay_region = cv2.addWeighted(image, 0.4, overlay_region, 0.6, 5)
    overlay_aff = cv2.addWeighted(image, 0.4, overlay_aff, 0.7, 6)

    gt_scores = np.concatenate([overlay_region, overlay_aff], axis=1)

    output = np.concatenate([gt_scores, confidence_mask_gray], axis=1)

    output = np.hstack([image, output])

    # synthtext
    if type(imagename) is not str:
        imagename = imagename[0].split("/")[-1][:-4]

    outpath = vis_dir + f"/{imagename}_input.jpg"
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
    cv2.imwrite(outpath, output)
    # print(f'Logging train input into {outpath}')


def saveImage(
    imagename,
    vis_dir,
    image,
    bboxes,
    affi_bboxes,
    region_scores,
    affinity_scores,
    confidence_mask,
):
    output_image = np.uint8(image.copy())
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    if len(bboxes) > 0:
        for i in range(len(bboxes)):
            _bboxes = np.int32(bboxes[i])
            for j in range(_bboxes.shape[0]):
                cv2.polylines(
                    output_image,
                    [np.reshape(_bboxes[j], (-1, 1, 2))],
                    True,
                    (0, 0, 255),
                )

        for i in range(len(affi_bboxes)):
            cv2.polylines(
                output_image,
                [np.reshape(affi_bboxes[i].astype(np.int32), (-1, 1, 2))],
                True,
                (255, 0, 0),
            )

    target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores)
    target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores)
    confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)

    # overlay
    height, width, channel = image.shape
    overlay_region = cv2.resize(target_gaussian_heatmap_color, (width, height))
    overlay_aff = cv2.resize(target_gaussian_affinity_heatmap_color, (width, height))

    overlay_region = cv2.addWeighted(image.copy(), 0.4, overlay_region, 0.6, 5)
    overlay_aff = cv2.addWeighted(image.copy(), 0.4, overlay_aff, 0.6, 5)

    heat_map = np.concatenate([overlay_region, overlay_aff], axis=1)

    # synthtext
    if type(imagename) is not str:
        imagename = imagename[0].split("/")[-1][:-4]

    output = np.concatenate([output_image, heat_map, confidence_mask_gray], axis=1)
    outpath = vis_dir + f"/{imagename}.jpg"
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    cv2.imwrite(outpath, output)
    # print(f'Logging original image into {outpath}')


def save_parser(args):

    """ final options """
    with open(f"{args.results_dir}/opt.txt", "a", encoding="utf-8") as opt_file:
        opt_log = "------------ Options -------------\n"
        arg = vars(args)
        for k, v in arg.items():
            opt_log += f"{str(k)}: {str(v)}\n"
        opt_log += "---------------------------------------\n"
        print(opt_log)
        opt_file.write(opt_log)
