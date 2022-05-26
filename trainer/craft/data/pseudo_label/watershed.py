import cv2
import numpy as np
from skimage.segmentation import watershed


def segment_region_score(watershed_param, region_score, word_image, pseudo_vis_opt):
    region_score = np.float32(region_score) / 255
    fore = np.uint8(region_score > 0.75)
    back = np.uint8(region_score < 0.05)
    unknown = 1 - (fore + back)
    ret, markers = cv2.connectedComponents(fore)
    markers += 1
    markers[unknown == 1] = 0

    labels = watershed(-region_score, markers)
    boxes = []
    for label in range(2, ret + 1):
        y, x = np.where(labels == label)
        x_max = x.max()
        y_max = y.max()
        x_min = x.min()
        y_min = y.min()
        box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        box = np.array(box)
        box *= 2
        boxes.append(box)
    return np.array(boxes, dtype=np.float32)


def exec_watershed_by_version(
    watershed_param, region_score, word_image, pseudo_vis_opt
):

    func_name_map_dict = {
        "skimage": segment_region_score,
    }

    try:
        return func_name_map_dict[watershed_param.version](
            watershed_param, region_score, word_image, pseudo_vis_opt
        )
    except:
        print(
            f"Watershed version {watershed_param.version} does not exist in func_name_map_dict."
        )
