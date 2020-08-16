from collections import OrderedDict
from typing import Dict

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable

from .craft import CRAFT
from .craft_utils import adjustResultCoordinates, getDetBoxes
from .imgproc import normalizeMeanVariance, resize_aspect_ratio


def copyStateDict(state_dict: OrderedDict) -> OrderedDict:
    """[summary] # TODO

    Args:
        state_dict (OrderedDict): [description]

    Returns:
        OrderedDict: [description]
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def test_net(
    canvas_size: int,
    mag_ratio: int,
    net,
    image: np.ndarray,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
    poly: bool,
    device: str,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """[summary] # TODO

    Args:
        canvas_size (int): [description]
        mag_ratio (int): [description]
        net ([type]): [description]
        image (np.ndarray): [description]
        text_threshold (float): [description]
        link_threshold (float): [description]
        low_text (float): [description]
        poly (bool): [description]
        device (str): [description]

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: [description]
    """

    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # Post-processing
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    return boxes, polys


def get_detector(trained_model: str, device: str = "cpu") -> CRAFT:
    """[summary] # TODO

    Args:
        trained_model (str): [description]
        device (str, optional): [description]. Defaults to "cpu".

    Returns:
        CRAFT: [description]
    """
    net = CRAFT()

    if device == "cpu":
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = False

    net.eval()
    return net


def get_textbox(
    detector: CRAFT,
    image: np.ndarray,
    canvas_size: int,
    mag_ratio: int,
    text_threshold: float,
    link_threshold: float,
    low_text: float,
    poly: bool,
    device: str,
) -> List[np.ndarray]:
    """[summary] # TODO

    Args:
        detector (CRAFT): [description]
        image (np.ndarray): [description]
        canvas_size (int): [description]
        mag_ratio (int): [description]
        text_threshold (float): [description]
        link_threshold (float): [description]
        low_text (float): [description]
        poly (bool): [description]
        device (str): [description]

    Returns:
        List[np.ndarray]: [description]
    """
    result = []
    bboxes, polys = test_net(
        canvas_size, mag_ratio, detector, image, text_threshold, link_threshold, low_text, poly, device
    )

    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        result.append(poly)

    return result
