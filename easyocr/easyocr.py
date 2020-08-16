# -*- coding: utf-8 -*-

import os
import sys
from logging import getLogger
from typing import Any, List

import cv2
import numpy as np
import torch
from bidi.algorithm import get_display

from .detection import get_detector, get_textbox
from .imgproc import loadImage
from .recognition import get_recognizer, get_text
from .settings import DETECTOR_FILENAME, MODULE_PATH, all_lang_list
from .utils import calculate_md5, download_and_unzip, get_image_list, get_paragraph, group_text_box

if sys.version_info[0] == 2:
    from io import open
    from six.moves.urllib.request import urlretrieve
    from pathlib2 import Path
else:
    from urllib.request import urlretrieve
    from pathlib import Path

LOGGER = getLogger(__name__)


class Reader(object):
    def __init__(
        self, lang_list: List[str], gpu: bool = True, model_storage_directory: str = None, download_enabled: bool = True
    ):
        """Create an EasyOCR Reader.

        Args:
            lang_list (List[str]): Language codes (ISO 639) for languages to be recognized during analysis.
            gpu (bool, optional): Enable GPU support. Defaults to True.
            model_storage_directory (str, optional): Path to directory for model data. If not specified,
                models will be read from a directory as defined by the environment variable
                EASYOCR_MODULE_PATH (preferred), MODULE_PATH (if defined), or ~/.EasyOCR/. Defaults to None.
            download_enabled (bool, optional): Enabled downloading of model data via HTTP. Defaults to True.
        """

        self.download_enabled = download_enabled
        self._set_model_dir(model_storage_directory)
        self._set_device(gpu)
        self._set_model_lang(lang_list)
        self._set_character_choices()
        self._set_lang_char(lang_list)  # self.lang_list doesn't seem to be used
        self._download_models()

        self.detector = get_detector(detector_path, self.device)
        self.recognizer, self.converter = get_recognizer(
            input_channel,
            output_channel,
            hidden_size,
            self.character,
            separator_list,
            dict_list,
            model_path,
            device=self.device,
        )

    def readtext(
        self,
        image: Any,
        decoder: str = "greedy",
        beamWidth: int = 5,
        batch_size: int = 1,
        workers: int = 0,
        allowlist: List[str] = None,
        blocklist: List[str] = None,
        detail: int = 1,
        paragraph: bool = False,
        contrast_ths: float = 0.1,
        adjust_contrast: float = 0.5,
        filter_ths: float = 0.003,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        slope_ths: float = 0.1,
        ycenter_ths: float = 0.5,
        height_ths: float = 0.5,
        width_ths: float = 0.5,
        add_margin: float = 0.1,
    ) -> List:  # TODO: ghandic - unsure on output shape
        """[summary] # TODO

        Args:
            image (Any): [description]
            decoder (str, optional): [description]. Defaults to "greedy".
            beamWidth (int, optional): [description]. Defaults to 5.
            batch_size (int, optional): [description]. Defaults to 1.
            workers (int, optional): [description]. Defaults to 0.
            allowlist (List[str], optional): [description]. Defaults to None.
            blocklist (List[str], optional): [description]. Defaults to None.
            detail (int, optional): [description]. Defaults to 1.
            paragraph (bool, optional): [description]. Defaults to False.
            contrast_ths (float, optional): [description]. Defaults to 0.1.
            adjust_contrast (float, optional): [description]. Defaults to 0.5.
            filter_ths (float, optional): [description]. Defaults to 0.003.
            text_threshold (float, optional): [description]. Defaults to 0.7.
            low_text (float, optional): [description]. Defaults to 0.4.
            link_threshold (float, optional): [description]. Defaults to 0.4.
            canvas_size (int, optional): [description]. Defaults to 2560.
            mag_ratio (float, optional): [description]. Defaults to 1.0.
            slope_ths (float, optional): [description]. Defaults to 0.1.
            ycenter_ths (float, optional): [description]. Defaults to 0.5.
            height_ths (float, optional): [description]. Defaults to 0.5.
            width_ths (float, optional): [description]. Defaults to 0.5.
            add_margin (float, optional): [description]. Defaults to 0.1.

        Returns:
            List: [description]
        """
        img, img_cv_grey = self._load_image(image)

        text_box = get_textbox(
            self.detector, img, canvas_size, mag_ratio, text_threshold, link_threshold, low_text, False, self.device
        )
        horizontal_list, free_list = group_text_box(text_box, slope_ths, ycenter_ths, height_ths, width_ths, add_margin)

        # should add filter to screen small box out

        image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height=imgH)

        if allowlist:
            ignore_char = "".join(set(self.character) - set(allowlist))
        elif blocklist:
            ignore_char = "".join(set(blocklist))
        else:
            ignore_char = "".join(set(self.character) - set(self.lang_char))

        if self.model_lang in ["chinese_tra", "chinese_sim", "japanese", "korean"]:
            decoder = "greedy"

        result = get_text(
            self.character,
            imgH,
            int(max_width),
            self.recognizer,
            self.converter,
            image_list,
            ignore_char,
            decoder,
            beamWidth,
            batch_size,
            contrast_ths,
            adjust_contrast,
            filter_ths,
            workers,
            self.device,
        )

        if self.model_lang == "arabic":
            direction_mode = "rtl"
            result = [list(item) for item in result]
            for item in result:
                item[1] = get_display(item[1])
        else:
            direction_mode = "ltr"

        if paragraph:
            result = get_paragraph(result, mode=direction_mode)

        if detail == 0:
            return [item[1] for item in result]
        else:
            return result

    def _load_image(self, image: Any) -> Tuple[np.ndarray, np.ndarray]:
        if type(image) == str:
            if image.startswith("http://") or image.startswith("https://"):
                tmp, _ = urlretrieve(image)
                img_cv_grey = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
                os.remove(tmp)
            else:
                img_cv_grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                image = os.path.expanduser(image)
            img = loadImage(image)  # can accept URL
        elif type(image) == bytes:
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif type(image) == np.ndarray:
            if len(image.shape) == 2:  # grayscale
                img_cv_grey = image
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3:  # BGRscale
                img = image
                img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError("Could not load image")

        return img, img_cv_grey

    def _download_models(self):
        model_path = os.path.join(self.model_storage_directory, model_file)
        corrupt_msg = "MD5 hash mismatch, possible file corruption"
        detector_path = os.path.join(self.model_storage_directory, DETECTOR_FILENAME)
        if os.path.isfile(detector_path) == False:
            if not self.download_enabled:
                raise FileNotFoundError("Missing %s and downloads disabled" % detector_path)
            LOGGER.warning(
                "Downloading detection model, please wait. "
                "This may take several minutes depending upon your network connection."
            )
            download_and_unzip(model_url["detector"][0], DETECTOR_FILENAME, self.model_storage_directory)
            assert calculate_md5(detector_path) == model_url["detector"][1], corrupt_msg
            LOGGER.info("Download complete")
        elif calculate_md5(detector_path) != model_url["detector"][1]:
            if not self.download_enabled:
                raise FileNotFoundError("MD5 mismatch for %s and downloads disabled" % detector_path)
            LOGGER.warning(corrupt_msg)
            os.remove(detector_path)
            LOGGER.warning(
                "Re-downloading the detection model, please wait. "
                "This may take several minutes depending upon your network connection."
            )
            download_and_unzip(model_url["detector"][0], DETECTOR_FILENAME, self.model_storage_directory)
            assert calculate_md5(detector_path) == model_url["detector"][1], corrupt_msg
        # check model file
        if os.path.isfile(model_path) == False:
            if not self.download_enabled:
                raise FileNotFoundError("Missing %s and downloads disabled" % model_path)
            LOGGER.warning(
                "Downloading recognition model, please wait. "
                "This may take several minutes depending upon your network connection."
            )
            download_and_unzip(model_url[model_file][0], model_file, self.model_storage_directory)
            assert calculate_md5(model_path) == model_url[model_file][1], corrupt_msg
            LOGGER.info("Download complete.")
        elif calculate_md5(model_path) != model_url[model_file][1]:
            if not self.download_enabled:
                raise FileNotFoundError("MD5 mismatch for %s and downloads disabled" % model_path)
            LOGGER.warning(corrupt_msg)
            os.remove(model_path)
            LOGGER.warning(
                "Re-downloading the recognition model, please wait. "
                "This may take several minutes depending upon your network connection."
            )
            download_and_unzip(model_url[model_file][0], model_file, self.model_storage_directory)
            assert calculate_md5(model_path) == model_url[model_file][1], corrupt_msg
            LOGGER.info("Download complete")

    def _set_lang_char(self, lang_list: List[str]):
        dict_list = {}
        for lang in lang_list:
            dict_list[lang] = os.path.join(BASE_PATH, "dict", lang + ".txt")

        self.lang_char = []
        for lang in lang_list:
            char_file = os.path.join(BASE_PATH, "character", lang + "_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                char_list = input_file.read().splitlines()
            self.lang_char += char_list
        self.lang_char = set(self.lang_char).union(set(number + symbol))
        self.lang_char = "".join(self.lang_char)

    def _set_model_lang(self, lang_list: List[str]):

        # check available languages
        unknown_lang = set(lang_list) - set(all_lang_list)
        if unknown_lang != set():
            raise ValueError(unknown_lang, "is not supported")

        # choose model
        if "th" in lang_list:
            self.model_lang = "thai"
            if set(lang_list) - set(["th", "en"]) != set():
                raise ValueError('Thai is only compatible with English, try lang_list=["th","en"]')
        elif "ch_tra" in lang_list:
            self.model_lang = "chinese_tra"
            if set(lang_list) - set(["ch_tra", "en"]) != set():
                raise ValueError('Chinese is only compatible with English, try lang_list=["ch_tra","en"]')
        elif "ch_sim" in lang_list:
            self.model_lang = "chinese_sim"
            if set(lang_list) - set(["ch_sim", "en"]) != set():
                raise ValueError('Chinese is only compatible with English, try lang_list=["ch_sim","en"]')
        elif "ja" in lang_list:
            self.model_lang = "japanese"
            if set(lang_list) - set(["ja", "en"]) != set():
                raise ValueError('Japanese is only compatible with English, try lang_list=["ja","en"]')
        elif "ko" in lang_list:
            self.model_lang = "korean"
            if set(lang_list) - set(["ko", "en"]) != set():
                raise ValueError('Korean is only compatible with English, try lang_list=["ko","en"]')
        elif "ta" in lang_list:
            self.model_lang = "tamil"
            if set(lang_list) - set(["ta", "en"]) != set():
                raise ValueError('Tamil is only compatible with English, try lang_list=["ta","en"]')
        elif set(lang_list) & set(arabic_lang_list):
            self.model_lang = "arabic"
            if set(lang_list) - set(arabic_lang_list + ["en"]) != set():
                raise ValueError('Arabic is only compatible with English, try lang_list=["ar","fa","ur","ug","en"]')
        elif set(lang_list) & set(devanagari_lang_list):
            self.model_lang = "devanagari"
            if set(lang_list) - set(devanagari_lang_list + ["en"]) != set():
                raise ValueError('Devanagari is only compatible with English, try lang_list=["hi","mr","ne","en"]')
        elif set(lang_list) & set(cyrillic_lang_list):
            self.model_lang = "cyrillic"
            if set(lang_list) - set(cyrillic_lang_list + ["en"]) != set():
                raise ValueError(
                    'Cyrillic is only compatible with English, try lang_list=["ru","rs_cyrillic","be","bg","uk","mn","en"]'
                )
        else:
            self.model_lang = "latin"

    def _set_character_choices(self):

        separator_list = {}
        if self.model_lang == "latin":
            all_char = (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                + "ÀÁÂÃÄÅÆÇÈÉÊËÍÎÑÒÓÔÕÖØÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿąęĮįıŁłŒœŠšųŽž"
            )
            self.character = number + symbol + all_char
            model_file = "latin.pth"
        elif self.model_lang == "arabic":
            ar_number = "٠١٢٣٤٥٦٧٨٩"
            ar_symbol = "«»؟،؛"
            ar_char = "ءآأؤإئااًبةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٰٓٔٱٹپچڈڑژکڭگںھۀہۂۃۆۇۈۋیېےۓە"
            self.character = number + symbol + en_char + ar_number + ar_symbol + ar_char
            model_file = "arabic.pth"
        elif self.model_lang == "cyrillic":
            cyrillic_char = (
                "ЁЂЄІЇЈЉЊЋЎЏАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёђєіїјљњћўџҐґҮүө"
            )
            self.character = number + symbol + en_char + cyrillic_char
            model_file = "cyrillic.pth"
        elif self.model_lang == "devanagari":
            devanagari_char = (
                ".ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰"
            )
            self.character = number + symbol + en_char + devanagari_char
            model_file = "devanagari.pth"
        elif self.model_lang == "chinese_tra":
            char_file = os.path.join(BASE_PATH, "character", "ch_tra_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                ch_tra_list = input_file.read().splitlines()
                ch_tra_char = "".join(ch_tra_list)
            self.character = number + symbol + en_char + ch_tra_char
            model_file = "chinese.pth"
        elif self.model_lang == "chinese_sim":
            char_file = os.path.join(BASE_PATH, "character", "ch_sim_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                ch_sim_list = input_file.read().splitlines()
                ch_sim_char = "".join(ch_sim_list)
            self.character = number + symbol + en_char + ch_sim_char
            model_file = "chinese_sim.pth"
        elif self.model_lang == "japanese":
            char_file = os.path.join(BASE_PATH, "character", "ja_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                ja_list = input_file.read().splitlines()
                ja_char = "".join(ja_list)
            self.character = number + symbol + en_char + ja_char
            model_file = "japanese.pth"
        elif self.model_lang == "korean":
            char_file = os.path.join(BASE_PATH, "character", "ko_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                ko_list = input_file.read().splitlines()
                ko_char = "".join(ko_list)
            self.character = number + symbol + en_char + ko_char
            model_file = "korean.pth"
        elif self.model_lang == "tamil":
            char_file = os.path.join(BASE_PATH, "character", "ta_char.txt")
            with open(char_file, "r", encoding="utf-8-sig") as input_file:
                ta_list = input_file.read().splitlines()
                ta_char = "".join(ta_list)
            self.character = number + symbol + en_char + ta_char
            model_file = "tamil.pth"
        elif self.model_lang == "thai":
            separator_list = {"th": ["\xa2", "\xa3"], "en": ["\xa4", "\xa5"]}
            separator_char = []
            for lang, sep in separator_list.items():
                separator_char += sep

            special_c0 = "ุู"
            special_c1 = "ิีืึ" + "ั"
            special_c2 = "่้๊๋"
            special_c3 = "็์"
            special_c = special_c0 + special_c1 + special_c2 + special_c3 + "ำ"
            th_char = "กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤ" + "เแโใไะา" + special_c + "ํฺ" + "ฯๆ"
            th_number = "0123456789๑๒๓๔๕๖๗๘๙"
            self.character = "".join(separator_char) + symbol + en_char + th_char + th_number
            model_file = "thai.pth"
        else:
            LOGGER.error("invalid language")

    def _set_model_dir(self, dir: str):
        self.model_storage_directory = MODULE_PATH + "/model"
        if dir:
            self.model_storage_directory = dir
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)

    def _set_device(self, gpu: bool):
        if gpu is False:
            self.device = "cpu"
            LOGGER.warning("Using CPU. Note: This module is much faster with a GPU.")
        elif not torch.cuda.is_available():
            self.device = "cpu"
            LOGGER.warning("CUDA not available - defaulting to CPU. Note: This module is much faster with a GPU.")
        elif gpu is True:
            self.device = "cuda"
        else:
            self.device = gpu

