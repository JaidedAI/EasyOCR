import os
import re
import itertools

import cv2
import time
import numpy as np
import torch
from torch.autograd import Variable

from utils.craft_utils import getDetBoxes, adjustResultCoordinates
from data import imgproc
from data.dataset import SynthTextDataSet
import math
import xml.etree.ElementTree as elemTree


#-------------------------------------------------------------------------------------------------------------------#
def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc
    yoff = yp - yc

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    # pRes = (xc + pResx, yc + pResy)
    return int(xc + pResx), int(yc + pResy)

def addRotatedShape(cx, cy, w, h, angle):
    p0x, p0y = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
    p1x, p1y = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
    p2x, p2y = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
    p3x, p3y = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)

    points = [[p0x, p0y], [p1x, p1y], [p2x, p2y], [p3x, p3y]]

    return points

def xml_parsing(xml):
    tree = elemTree.parse(xml)

    annotations = []  # Initialize the list to store labels
    iter_element = tree.iter(tag="object")

    for element in iter_element:
        annotation = {}  # Initialize the dict to store labels

        annotation['name'] = element.find("name").text  # Save the name tag value

        box_coords = element.iter(tag="robndbox")

        for box_coord in box_coords:
            cx = float(box_coord.find("cx").text)
            cy = float(box_coord.find("cy").text)
            w = float(box_coord.find("w").text)
            h = float(box_coord.find("h").text)
            angle = float(box_coord.find("angle").text)

            convertcoodi = addRotatedShape(cx, cy, w, h, angle)

            annotation['box_coodi'] = convertcoodi
            annotations.append(annotation)

        box_coords = element.iter(tag="bndbox")

        for box_coord in box_coords:
            xmin = int(box_coord.find("xmin").text)
            ymin = int(box_coord.find("ymin").text)
            xmax = int(box_coord.find("xmax").text)
            ymax = int(box_coord.find("ymax").text)
            # annotation['bndbox'] = [xmin,ymin,xmax,ymax]

            annotation['box_coodi'] = [[xmin, ymin], [xmax, ymin], [xmax, ymax],
                                       [xmin, ymax]]
            annotations.append(annotation)




    bounds = []
    for i in range(len(annotations)):
        box_info_dict = {"points": None, "text": None, "ignore": None}

        box_info_dict["points"] = np.array(annotations[i]['box_coodi'])
        if annotations[i]['name'] == "dnc":
            box_info_dict["text"] = "###"
            box_info_dict["ignore"] = True
        else:
            box_info_dict["text"] = annotations[i]['name']
            box_info_dict["ignore"] = False

        bounds.append(box_info_dict)



    return bounds

#-------------------------------------------------------------------------------------------------------------------#

def load_prescription_gt(dataFolder):


    total_img_path = []
    total_imgs_bboxes = []
    for (root, directories, files) in os.walk(dataFolder):
        for file in files:
            if '.jpg' in file:
                img_path = os.path.join(root, file)
                total_img_path.append(img_path)
            if '.xml' in file:
                gt_path = os.path.join(root, file)
                total_imgs_bboxes.append(gt_path)


    total_imgs_parsing_bboxes = []
    for img_path, bbox in zip(sorted(total_img_path), sorted(total_imgs_bboxes)):
        # check file

        assert img_path.split(".jpg")[0] == bbox.split(".xml")[0]

        result_label = xml_parsing(bbox)
        total_imgs_parsing_bboxes.append(result_label)


    return total_imgs_parsing_bboxes, sorted(total_img_path)


# NOTE
def load_prescription_cleval_gt(dataFolder):


    total_img_path = []
    total_gt_path = []
    for (root, directories, files) in os.walk(dataFolder):
        for file in files:
            if '.jpg' in file:
                img_path = os.path.join(root, file)
                total_img_path.append(img_path)
            if '_cl.txt' in file:
                gt_path = os.path.join(root, file)
                total_gt_path.append(gt_path)


    total_imgs_parsing_bboxes = []
    for img_path, gt_path in zip(sorted(total_img_path), sorted(total_gt_path)):
        # check file

        assert img_path.split(".jpg")[0] == gt_path.split('_label_cl.txt')[0]

        lines = open(gt_path, encoding="utf-8").readlines()
        word_bboxes = []

        for line in lines:
            box_info_dict = {"points": None, "text": None, "ignore": None}
            box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")

            box_points = [int(box_info[i]) for i in range(8)]
            box_info_dict["points"] = np.array(box_points)

            word_bboxes.append(box_info_dict)
        total_imgs_parsing_bboxes.append(word_bboxes)

    return total_imgs_parsing_bboxes, sorted(total_img_path)


def load_synthtext_gt(data_folder):

    synth_dataset = SynthTextDataSet(
        output_size=768, data_dir=data_folder, saved_gt_dir=data_folder, logging=False
    )
    img_names, img_bbox, img_words = synth_dataset.load_data(bbox="word")

    total_img_path = []
    total_imgs_bboxes = []
    for index in range(len(img_bbox[:100])):
        img_path = os.path.join(data_folder, img_names[index][0])
        total_img_path.append(img_path)
        try:
            wordbox = img_bbox[index].transpose((2, 1, 0))
        except:
            wordbox = np.expand_dims(img_bbox[index], axis=0)
            wordbox = wordbox.transpose((0, 2, 1))

        words = [re.split(" \n|\n |\n| ", t.strip()) for t in img_words[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

        if len(words) != len(wordbox):
            import ipdb

            ipdb.set_trace()

        single_img_bboxes = []
        for j in range(len(words)):
            box_info_dict = {"points": None, "text": None, "ignore": None}
            box_info_dict["points"] = wordbox[j]
            box_info_dict["text"] = words[j]
            box_info_dict["ignore"] = False
            single_img_bboxes.append(box_info_dict)

        total_imgs_bboxes.append(single_img_bboxes)

    return total_imgs_bboxes, total_img_path


def load_icdar2015_gt(dataFolder, isTraing=False):
    if isTraing:
        img_folderName = "ch4_training_images"
        gt_folderName = "ch4_training_localization_transcription_gt"
    else:
        img_folderName = "ch4_test_images"
        gt_folderName = "ch4_test_localization_transcription_gt"

    gt_folder_path = os.listdir(os.path.join(dataFolder, gt_folderName))
    total_imgs_bboxes = []
    total_img_path = []
    for gt_path in gt_folder_path:
        gt_path = os.path.join(os.path.join(dataFolder, gt_folderName), gt_path)
        img_path = (
            gt_path.replace(gt_folderName, img_folderName)
            .replace(".txt", ".jpg")
            .replace("gt_", "")
        )
        image = cv2.imread(img_path)
        lines = open(gt_path, encoding="utf-8").readlines()
        single_img_bboxes = []
        for line in lines:
            box_info_dict = {"points": None, "text": None, "ignore": None}

            box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")
            box_points = [int(box_info[j]) for j in range(8)]
            word = box_info[8:]
            word = ",".join(word)
            box_points = np.array(box_points, np.int32).reshape(4, 2)
            cv2.polylines(
                image, [np.array(box_points).astype(np.int)], True, (0, 0, 255), 1
            )
            box_info_dict["points"] = box_points
            box_info_dict["text"] = word
            if word == "###":
                box_info_dict["ignore"] = True
            else:
                box_info_dict["ignore"] = False

            single_img_bboxes.append(box_info_dict)
        total_imgs_bboxes.append(single_img_bboxes)
        total_img_path.append(img_path)
    return total_imgs_bboxes, total_img_path


def load_icdar2013_gt(dataFolder, isTraing=False):

    # choise test dataset
    if isTraing:
        img_folderName = "Challenge2_Test_Task12_Images"
        gt_folderName = "Challenge2_Test_Task1_GT"
    else:
        img_folderName = "Challenge2_Test_Task12_Images"
        gt_folderName = "Challenge2_Test_Task1_GT"

    gt_folder_path = os.listdir(os.path.join(dataFolder, gt_folderName))

    total_imgs_bboxes = []
    total_img_path = []
    for gt_path in gt_folder_path:
        gt_path = os.path.join(os.path.join(dataFolder, gt_folderName), gt_path)
        img_path = (
            gt_path.replace(gt_folderName, img_folderName)
            .replace(".txt", ".jpg")
            .replace("gt_", "")
        )
        image = cv2.imread(img_path)
        lines = open(gt_path, encoding="utf-8").readlines()
        single_img_bboxes = []
        for line in lines:
            box_info_dict = {"points": None, "text": None, "ignore": None}

            box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")
            box = [int(box_info[j]) for j in range(4)]
            word = box_info[4:]
            word = ",".join(word)
            box = [
                [box[0], box[1]],
                [box[2], box[1]],
                [box[2], box[3]],
                [box[0], box[3]],
            ]

            box_info_dict["points"] = box
            box_info_dict["text"] = word
            if word == "###":
                box_info_dict["ignore"] = True
            else:
                box_info_dict["ignore"] = False

            single_img_bboxes.append(box_info_dict)

        total_imgs_bboxes.append(single_img_bboxes)
        total_img_path.append(img_path)

    return total_imgs_bboxes, total_img_path


def test_net(
    net,
    image,
    text_threshold,
    link_threshold,
    low_text,
    cuda,
    poly,
    canvas_size=1280,
    mag_ratio=1.5,
):
    # resize

    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy().astype(np.float32)
    score_link = y[0, :, :, 1].cpu().data.numpy().astype(np.float32)

    # NOTE
    score_text = score_text[: size_heatmap[0], : size_heatmap[1]]
    score_link = score_link[: size_heatmap[0], : size_heatmap[1]]

    # Post-processing
    boxes, polys = getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    # render results (optional)
    score_text = score_text.copy()
    render_score_text = imgproc.cvt2HeatmapImg(score_text)
    render_score_link = imgproc.cvt2HeatmapImg(score_link)
    render_img = [render_score_text, render_score_link]
    # ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, render_img
