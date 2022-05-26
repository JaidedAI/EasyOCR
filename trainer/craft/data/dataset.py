import os
import re
import itertools
import random

import numpy as np
import scipy.io as scio
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data import imgproc
from data.gaussian import GaussianBuilder
from data.imgaug import (
    rescale,
    random_resize_crop_synth,
    random_resize_crop,
    random_horizontal_flip,
    random_rotate,
    random_scale,
    random_crop,
)
from data.pseudo_label.make_charbox import PseudoCharBoxBuilder
from utils.util import saveInput, saveImage


class CraftBaseDataset(Dataset):
    def __init__(
        self,
        output_size,
        data_dir,
        saved_gt_dir,
        mean,
        variance,
        gauss_init_size,
        gauss_sigma,
        enlarge_region,
        enlarge_affinity,
        aug,
        vis_test_dir,
        vis_opt,
        sample,
    ):
        self.output_size = output_size
        self.data_dir = data_dir
        self.saved_gt_dir = saved_gt_dir
        self.mean, self.variance = mean, variance
        self.gaussian_builder = GaussianBuilder(
            gauss_init_size, gauss_sigma, enlarge_region, enlarge_affinity
        )
        self.aug = aug
        self.vis_test_dir = vis_test_dir
        self.vis_opt = vis_opt
        self.sample = sample
        if self.sample != -1:
            random.seed(0)
            self.idx = random.sample(range(0, len(self.img_names)), self.sample)

        self.pre_crop_area = []

    def augment_image(
        self, image, region_score, affinity_score, confidence_mask, word_level_char_bbox
    ):
        augment_targets = [image, region_score, affinity_score, confidence_mask]

        if self.aug.random_scale.option:
            augment_targets, word_level_char_bbox = random_scale(
                augment_targets, word_level_char_bbox, self.aug.random_scale.range
            )

        if self.aug.random_rotate.option:
            augment_targets = random_rotate(
                augment_targets, self.aug.random_rotate.max_angle
            )

        if self.aug.random_crop.option:
            if self.aug.random_crop.version == "random_crop_with_bbox":
                augment_targets = random_crop_with_bbox(
                    augment_targets, word_level_char_bbox, self.output_size
                )
            elif self.aug.random_crop.version == "random_resize_crop_synth":
                augment_targets = random_resize_crop_synth(
                    augment_targets, self.output_size
                )
            elif self.aug.random_crop.version == "random_resize_crop":

                if len(self.pre_crop_area) > 0:
                    pre_crop_area = self.pre_crop_area
                else:
                    pre_crop_area = None

                augment_targets = random_resize_crop(
                    augment_targets,
                    self.aug.random_crop.scale,
                    self.aug.random_crop.ratio,
                    self.output_size,
                    self.aug.random_crop.rnd_threshold,
                    pre_crop_area,
                )

            elif self.aug.random_crop.version == "random_crop":
                augment_targets = random_crop(augment_targets, self.output_size,)

            else:
                assert "Undefined RandomCrop version"

        if self.aug.random_horizontal_flip.option:
            augment_targets = random_horizontal_flip(augment_targets)

        if self.aug.random_colorjitter.option:
            image, region_score, affinity_score, confidence_mask = augment_targets
            image = Image.fromarray(image)
            image = transforms.ColorJitter(
                brightness=self.aug.random_colorjitter.brightness,
                contrast=self.aug.random_colorjitter.contrast,
                saturation=self.aug.random_colorjitter.saturation,
                hue=self.aug.random_colorjitter.hue,
            )(image)
        else:
            image, region_score, affinity_score, confidence_mask = augment_targets

        return np.array(image), region_score, affinity_score, confidence_mask

    def resize_to_half(self, ground_truth, interpolation):
        return cv2.resize(
            ground_truth,
            (self.output_size // 2, self.output_size // 2),
            interpolation=interpolation,
        )

    def __len__(self):
        if self.sample != -1:
            return len(self.idx)
        else:
            return len(self.img_names)

    def __getitem__(self, index):
        if self.sample != -1:
            index = self.idx[index]
        if self.saved_gt_dir is None:
            (
                image,
                region_score,
                affinity_score,
                confidence_mask,
                word_level_char_bbox,
                all_affinity_bbox,
                words,
            ) = self.make_gt_score(index)
        else:
            (
                image,
                region_score,
                affinity_score,
                confidence_mask,
                word_level_char_bbox,
                words,
            ) = self.load_saved_gt_score(index)
            all_affinity_bbox = []

        if self.vis_opt:
            saveImage(
                self.img_names[index],
                self.vis_test_dir,
                image.copy(),
                word_level_char_bbox.copy(),
                all_affinity_bbox.copy(),
                region_score.copy(),
                affinity_score.copy(),
                confidence_mask.copy(),
            )

        image, region_score, affinity_score, confidence_mask = self.augment_image(
            image, region_score, affinity_score, confidence_mask, word_level_char_bbox
        )

        if self.vis_opt:
            saveInput(
                self.img_names[index],
                self.vis_test_dir,
                image,
                region_score,
                affinity_score,
                confidence_mask,
            )

        region_score = self.resize_to_half(region_score, interpolation=cv2.INTER_CUBIC)
        affinity_score = self.resize_to_half(
            affinity_score, interpolation=cv2.INTER_CUBIC
        )
        confidence_mask = self.resize_to_half(
            confidence_mask, interpolation=cv2.INTER_NEAREST
        )

        image = imgproc.normalizeMeanVariance(
            np.array(image), mean=self.mean, variance=self.variance
        )
        image = image.transpose(2, 0, 1)

        return image, region_score, affinity_score, confidence_mask


class SynthTextDataSet(CraftBaseDataset):
    def __init__(
        self,
        output_size,
        data_dir,
        saved_gt_dir,
        mean,
        variance,
        gauss_init_size,
        gauss_sigma,
        enlarge_region,
        enlarge_affinity,
        aug,
        vis_test_dir,
        vis_opt,
        sample,
    ):
        super().__init__(
            output_size,
            data_dir,
            saved_gt_dir,
            mean,
            variance,
            gauss_init_size,
            gauss_sigma,
            enlarge_region,
            enlarge_affinity,
            aug,
            vis_test_dir,
            vis_opt,
            sample,
        )
        self.img_names, self.char_bbox, self.img_words = self.load_data()
        self.vis_index = list(range(1000))

    def load_data(self, bbox="char"):

        gt = scio.loadmat(os.path.join(self.data_dir, "gt.mat"))
        img_names = gt["imnames"][0]
        img_words = gt["txt"][0]

        if bbox == "char":
            img_bbox = gt["charBB"][0]
        else:
            img_bbox = gt["wordBB"][0]  # word bbox needed for test

        return img_names, img_bbox, img_words

    def dilate_img_to_output_size(self, image, char_bbox):
        h, w, _ = image.shape
        if min(h, w) <= self.output_size:
            scale = float(self.output_size) / min(h, w)
        else:
            scale = 1.0
        image = cv2.resize(
            image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        char_bbox *= scale
        return image, char_bbox

    def make_gt_score(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_char_bbox = self.char_bbox[index].transpose(
            (2, 1, 0)
        )  # shape : (Number of characters in image, 4, 2)

        img_h, img_w, _ = image.shape

        confidence_mask = np.ones((img_h, img_w), dtype=np.float32)

        words = [
            re.split(" \n|\n |\n| ", word.strip()) for word in self.img_words[index]
        ]
        words = list(itertools.chain(*words))
        words = [word for word in words if len(word) > 0]

        word_level_char_bbox = []
        char_idx = 0

        for i in range(len(words)):
            length_of_word = len(words[i])
            word_bbox = all_char_bbox[char_idx : char_idx + length_of_word]
            assert len(word_bbox) == length_of_word
            char_idx += length_of_word
            word_bbox = np.array(word_bbox)
            word_level_char_bbox.append(word_bbox)

        region_score = self.gaussian_builder.generate_region(
            img_h,
            img_w,
            word_level_char_bbox,
            horizontal_text_bools=[True for _ in range(len(words))],
        )
        affinity_score, all_affinity_bbox = self.gaussian_builder.generate_affinity(
            img_h,
            img_w,
            word_level_char_bbox,
            horizontal_text_bools=[True for _ in range(len(words))],
        )

        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            all_affinity_bbox,
            words,
        )


class CustomDataset(CraftBaseDataset):
    def __init__(
        self,
        output_size,
        data_dir,
        saved_gt_dir,
        mean,
        variance,
        gauss_init_size,
        gauss_sigma,
        enlarge_region,
        enlarge_affinity,
        aug,
        vis_test_dir,
        vis_opt,
        sample,
        watershed_param,
        pseudo_vis_opt,
        do_not_care_label,
    ):
        super().__init__(
            output_size,
            data_dir,
            saved_gt_dir,
            mean,
            variance,
            gauss_init_size,
            gauss_sigma,
            enlarge_region,
            enlarge_affinity,
            aug,
            vis_test_dir,
            vis_opt,
            sample,
        )
        self.pseudo_vis_opt = pseudo_vis_opt
        self.do_not_care_label = do_not_care_label
        self.pseudo_charbox_builder = PseudoCharBoxBuilder(
            watershed_param, vis_test_dir, pseudo_vis_opt, self.gaussian_builder
        )
        self.vis_index = list(range(1000))
        self.img_dir = os.path.join(data_dir, "ch4_training_images")
        self.img_gt_box_dir = os.path.join(
            data_dir, "ch4_training_localization_transcription_gt"
        )
        self.img_names = os.listdir(self.img_dir)

    def update_model(self, net):
        self.net = net

    def update_device(self, gpu):
        self.gpu = gpu

    def load_img_gt_box(self, img_gt_box_path):
        lines = open(img_gt_box_path, encoding="utf-8").readlines()
        word_bboxes = []
        words = []
        for line in lines:
            box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")
            box_points = [int(box_info[i]) for i in range(8)]
            box_points = np.array(box_points, np.float32).reshape(4, 2)
            word = box_info[8:]
            word = ",".join(word)
            if word in self.do_not_care_label:
                words.append(self.do_not_care_label[0])
                word_bboxes.append(box_points)
                continue
            word_bboxes.append(box_points)
            words.append(word)
        return np.array(word_bboxes), words

    def load_data(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gt_box_path = os.path.join(
            self.img_gt_box_dir, "gt_%s.txt" % os.path.splitext(img_name)[0]
        )
        word_bboxes, words = self.load_img_gt_box(
            img_gt_box_path
        )  # shape : (Number of word bbox, 4, 2)
        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        word_level_char_bbox = []
        do_care_words = []
        horizontal_text_bools = []

        if len(word_bboxes) == 0:
            return (
                image,
                word_level_char_bbox,
                do_care_words,
                confidence_mask,
                horizontal_text_bools,
            )
        _word_bboxes = word_bboxes.copy()
        for i in range(len(word_bboxes)):
            if words[i] in self.do_not_care_label:
                cv2.fillPoly(confidence_mask, [np.int32(_word_bboxes[i])], 0)
                continue

            (
                pseudo_char_bbox,
                confidence,
                horizontal_text_bool,
            ) = self.pseudo_charbox_builder.build_char_box(
                self.net, self.gpu, image, word_bboxes[i], words[i], img_name=img_name
            )

            cv2.fillPoly(confidence_mask, [np.int32(_word_bboxes[i])], confidence)
            do_care_words.append(words[i])
            word_level_char_bbox.append(pseudo_char_bbox)
            horizontal_text_bools.append(horizontal_text_bool)

        return (
            image,
            word_level_char_bbox,
            do_care_words,
            confidence_mask,
            horizontal_text_bools,
        )

    def make_gt_score(self, index):
        """
        Make region, affinity scores using pseudo character-level GT bounding box
        word_level_char_bbox's shape : [word_num, [char_num_in_one_word, 4, 2]]
        :rtype region_score: np.float32
        :rtype affinity_score: np.float32
        :rtype confidence_mask: np.float32
        :rtype word_level_char_bbox: np.float32
        :rtype words: list
        """
        (
            image,
            word_level_char_bbox,
            words,
            confidence_mask,
            horizontal_text_bools,
        ) = self.load_data(index)
        img_h, img_w, _ = image.shape

        if len(word_level_char_bbox) == 0:
            region_score = np.zeros((img_h, img_w), dtype=np.float32)
            affinity_score = np.zeros((img_h, img_w), dtype=np.float32)
            all_affinity_bbox = []
        else:
            region_score = self.gaussian_builder.generate_region(
                img_h, img_w, word_level_char_bbox, horizontal_text_bools
            )
            affinity_score, all_affinity_bbox = self.gaussian_builder.generate_affinity(
                img_h, img_w, word_level_char_bbox, horizontal_text_bools
            )

        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            all_affinity_bbox,
            words,
        )

    def load_saved_gt_score(self, index):
        """
        Load pre-saved official CRAFT model's region, affinity scores to train
        word_level_char_bbox's shape : [word_num, [char_num_in_one_word, 4, 2]]
        :rtype region_score: np.float32
        :rtype affinity_score: np.float32
        :rtype confidence_mask: np.float32
        :rtype word_level_char_bbox: np.float32
        :rtype words: list
        """
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gt_box_path = os.path.join(
            self.img_gt_box_dir, "gt_%s.txt" % os.path.splitext(img_name)[0]
        )
        word_bboxes, words = self.load_img_gt_box(img_gt_box_path)
        image, word_bboxes = rescale(image, word_bboxes)
        img_h, img_w, _ = image.shape

        query_idx = int(self.img_names[index].split(".")[0].split("_")[1])

        saved_region_scores_path = os.path.join(
            self.saved_gt_dir, f"res_img_{query_idx}_region.jpg"
        )
        saved_affi_scores_path = os.path.join(
            self.saved_gt_dir, f"res_img_{query_idx}_affi.jpg"
        )
        saved_cf_mask_path = os.path.join(
            self.saved_gt_dir, f"res_img_{query_idx}_cf_mask_thresh_0.6.jpg"
        )
        region_score = cv2.imread(saved_region_scores_path, cv2.IMREAD_GRAYSCALE)
        affinity_score = cv2.imread(saved_affi_scores_path, cv2.IMREAD_GRAYSCALE)
        confidence_mask = cv2.imread(saved_cf_mask_path, cv2.IMREAD_GRAYSCALE)

        region_score = cv2.resize(region_score, (img_w, img_h))
        affinity_score = cv2.resize(affinity_score, (img_w, img_h))
        confidence_mask = cv2.resize(
            confidence_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST
        )

        region_score = region_score.astype(np.float32) / 255
        affinity_score = affinity_score.astype(np.float32) / 255
        confidence_mask = confidence_mask.astype(np.float32) / 255

        # NOTE : Even though word_level_char_bbox is not necessary, align bbox format with make_gt_score()
        word_level_char_bbox = []

        for i in range(len(word_bboxes)):
            word_level_char_bbox.append(np.expand_dims(word_bboxes[i], 0))

        return (
            image,
            region_score,
            affinity_score,
            confidence_mask,
            word_level_char_bbox,
            words,
        )
