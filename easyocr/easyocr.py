# -*- coding: utf-8 -*-

from .detection import get_detector, get_textbox
from .imgproc import loadImage
from .recognition import get_recognizer, get_text
from .utils import group_text_box, get_image_list, calculate_md5, get_paragraph
import numpy as np
import cv2
import torch
import os
import sys
from logging import getLogger

if sys.version_info[0] == 2:
    from io import open
    from six.moves.urllib.request import urlretrieve
    from pathlib2 import Path
else:
    from urllib.request import urlretrieve
    from pathlib import Path

LOGGER = getLogger(__name__)

BASE_PATH = os.path.dirname(__file__)
MODULE_PATH = os.environ.get("EASYOCR_MODULE_PATH") or \
              os.environ.get("MODULE_PATH") or \
              os.path.expanduser("~/.EasyOCR/")

# detector parameters
DETECTOR_FILENAME = 'craft_mlt_25k.pth'

# recognizer parameters
latin_lang_list = ['af','az','bs','cs','cy','da','de','en','es','et','fr','ga',\
                   'hr','hu','id','is','it','ku','la','lt','lv','mi','ms','mt',\
                   'nl','no','oc','pl','pt','ro','rs_latin','sk','sl','sq',\
                   'sv','sw','tl','tr','uz','vi']
devanagari_lang_list = ['hi','mr','ne']
all_lang_list = latin_lang_list + devanagari_lang_list + ['th','ch_sim','ch_tra','ja','ko']
imgH = 64
input_channel = 1
output_channel = 512
hidden_size = 512

number = '0123456789'
symbol  = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
en_char = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

model_url = {
    'detector': ('https://www.jaided.ai/read_download/craft_mlt_25k.pth', '2f8227d2def4037cdb3b34389dcf9ec1'),
    'latin.pth': ('https://www.jaided.ai/read_download/latin.pth', 'fb91b9abf65aeeac95a172291b4a6176'),
    'chinese.pth': ('https://www.jaided.ai/read_download/chinese.pth', 'dfba8e364cd98ed4fed7ad54d71e3965'),
    'chinese_sim.pth': ('https://www.jaided.ai/read_download/chinese_sim.pth', '0e19a9d5902572e5237b04ee29bdb636'),
    'japanese.pth': ('https://www.jaided.ai/read_download/japanese.pth', '6d891a4aad9cb7f492809515e4e9fd2e'),
    'korean.pth': ('https://www.jaided.ai/read_download/korean.pth', '45b3300e0f04ce4d03dda9913b20c336'),
    'thai.pth': ('https://www.jaided.ai/read_download/thai.pth', '40a06b563a2b3d7897e2d19df20dc709'),
    'devanagari.pth': ('https://www.jaided.ai/read_download/devanagari.pth', 'db6b1f074fae3070f561675db908ac08'),
}

class Reader(object):

    def __init__(self, lang_list, gpu=True, model_storage_directory=None, download_enabled=True):
        """Create an EasyOCR Reader.

        Parameters:
            lang_list (list): Language codes (ISO 639) for languages to be recognized during analysis.

            gpu (bool): Enable GPU support (default)

            model_storage_directory (string): Path to directory for model data. If not specified,
            models will be read from a directory as defined by the environment variable
            EASYOCR_MODULE_PATH (preferred), MODULE_PATH (if defined), or ~/.EasyOCR/.

            download_enabled (bool): Enabled downloading of model data via HTTP (default).
        """
        self.download_enabled = download_enabled

        self.model_storage_directory = MODULE_PATH + '/model'
        if model_storage_directory:
            self.model_storage_directory = model_storage_directory
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)

        if gpu is False:
            self.device = 'cpu'
            LOGGER.warning('Using CPU. Note: This module is much faster with a GPU.')
        elif not torch.cuda.is_available():
            self.device = 'cpu'
            LOGGER.warning('CUDA not available - defaulting to CPU. Note: This module is much faster with a GPU.')
        elif gpu is True:
            self.device = 'cuda'
        else:
            self.device = gpu

        # check available languages
        unknown_lang = set(lang_list) - set(all_lang_list)
        if unknown_lang != set():
            raise ValueError(unknown_lang, 'is not supported')

        # choose model
        if 'th' in lang_list:
            self.model_lang = 'thai'
            if set(lang_list) - set(['th','en']) != set():
                raise ValueError('Thai is only compatible with English, try lang_list=["th","en"]')
        elif 'ch_tra' in lang_list:
            self.model_lang = 'chinese_tra'
            if set(lang_list) - set(['ch_tra','en']) != set():
                raise ValueError('Chinese is only compatible with English, try lang_list=["ch_tra","en"]')
        elif 'ch_sim' in lang_list:
            self.model_lang = 'chinese_sim'
            if set(lang_list) - set(['ch_sim','en']) != set():
                raise ValueError('Chinese is only compatible with English, try lang_list=["ch_sim","en"]')
        elif 'ja' in lang_list:
            self.model_lang = 'japanese'
            if set(lang_list) - set(['ja','en']) != set():
                raise ValueError('Japanese is only compatible with English, try lang_list=["ja","en"]')
        elif 'ko' in lang_list:
            self.model_lang = 'korean'
            if set(lang_list) - set(['ko','en']) != set():
                raise ValueError('Korean is only compatible with English, try lang_list=["ko","en"]')
        elif set(lang_list) & set(devanagari_lang_list):
            self.model_lang = 'devanagari'
            if set(lang_list) - set(devanagari_lang_list+['en']) != set():
                raise ValueError('Devanagari is only compatible with English, try lang_list=["hi","mr","ne","en"]')
        else: self.model_lang = 'latin'

        separator_list = {}
        if self.model_lang == 'latin':
            all_char = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'+\
            'ÀÁÂÃÄÅÆÇÈÉÊËÍÎÑÒÓÔÕÖØÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿąęĮįıŁłŒœŠšųŽž'
            self.character = number+ symbol + all_char
            model_file = 'latin.pth'

        elif self.model_lang == 'devanagari':
            devanagari_char = '.ँंःअअंअःआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळवशषसह़ािीुूृॅेैॉोौ्ॐ॒क़ख़ग़ज़ड़ढ़फ़ॠ।०१२३४५६७८९॰'
            self.character = number+ symbol + en_char + devanagari_char
            model_file = 'devanagari.pth'

        elif  self.model_lang == 'chinese_tra':
            char_file = os.path.join(BASE_PATH, 'character', "ch_tra_char.txt")
            with open(char_file, "r", encoding = "utf-8-sig") as input_file:
                ch_tra_list =  input_file.read().splitlines()
                ch_tra_char = ''.join(ch_tra_list)
            self.character = number + symbol + en_char + ch_tra_char
            model_file = 'chinese.pth'

        elif  self.model_lang == 'chinese_sim':
            char_file = os.path.join(BASE_PATH, 'character', "ch_sim_char.txt")
            with open(char_file, "r", encoding = "utf-8-sig") as input_file:
                ch_sim_list =  input_file.read().splitlines()
                ch_sim_char = ''.join(ch_sim_list)
            self.character = number + symbol + en_char + ch_sim_char
            model_file = 'chinese_sim.pth'

        elif  self.model_lang == 'japanese':
            char_file = os.path.join(BASE_PATH, 'character', "ja_char.txt")
            with open(char_file, "r", encoding = "utf-8-sig") as input_file:
                ja_list =  input_file.read().splitlines()
                ja_char = ''.join(ja_list)
            self.character = number + symbol + en_char + ja_char
            model_file = 'japanese.pth'

        elif  self.model_lang == 'korean':
            char_file = os.path.join(BASE_PATH, 'character', "ko_char.txt")
            with open(char_file, "r", encoding = "utf-8-sig") as input_file:
                ko_list =  input_file.read().splitlines()
                ko_char = ''.join(ko_list)
            self.character = number + symbol + en_char + ko_char
            model_file = 'korean.pth'

        elif self.model_lang == 'thai':
            separator_list = {
                'th': ['\xa2', '\xa3'],
                'en': ['\xa4', '\xa5']
            }
            separator_char = []
            for lang, sep in separator_list.items():
                separator_char += sep

            special_c0 = 'ุู'
            special_c1 = 'ิีืึ'+ 'ั'
            special_c2 = '่้๊๋'
            special_c3 = '็์'
            special_c = special_c0+special_c1+special_c2+special_c3 + 'ำ'
            th_char = 'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮฤ' +'เแโใไะา'+ special_c +  'ํฺ'+'ฯๆ'
            th_number = '0123456789๑๒๓๔๕๖๗๘๙'
            self.character = ''.join(separator_char) + symbol + en_char + th_char + th_number
            model_file = 'thai.pth'
        else:
            LOGGER.error('invalid language')

        dict_list = {}
        for lang in lang_list:
            dict_list[lang] = os.path.join(BASE_PATH, 'dict', lang + ".txt")

        self.lang_char = []
        for lang in lang_list:
            char_file = os.path.join(BASE_PATH, 'character', lang + "_char.txt")
            with open(char_file, "r", encoding = "utf-8-sig") as input_file:
                char_list =  input_file.read().splitlines()
            self.lang_char += char_list
        self.lang_char = set(self.lang_char).union(set(number+symbol))
        self.lang_char = ''.join(self.lang_char)

        model_path = os.path.join(self.model_storage_directory, model_file)
        corrupt_msg = 'MD5 hash mismatch, possible file corruption'
        detector_path = os.path.join(self.model_storage_directory, DETECTOR_FILENAME)
        if os.path.isfile(detector_path) == False:
            if not self.download_enabled:
                raise FileNotFoundError("Missing %s and downloads disabled" % detector_path)
            LOGGER.warning('Downloading detection model, please wait. '
                           'This may take several minutes depending upon your network connection.')
            urlretrieve(model_url['detector'][0] , detector_path)
            assert calculate_md5(detector_path) == model_url['detector'][1], corrupt_msg
            LOGGER.info('Download complete')
        elif calculate_md5(detector_path) != model_url['detector'][1]:
            if not self.download_enabled:
                raise FileNotFoundError("MD5 mismatch for %s and downloads disabled" % detector_path)
            LOGGER.warning(corrupt_msg)
            os.remove(detector_path)
            LOGGER.warning('Re-downloading the detection model, please wait. '
                           'This may take several minutes depending upon your network connection.')
            urlretrieve(model_url['detector'][0], detector_path)
            assert calculate_md5(detector_path) == model_url['detector'][1], corrupt_msg
        # check model file
        if os.path.isfile(model_path) == False:
            if not self.download_enabled:
                raise FileNotFoundError("Missing %s and downloads disabled" % model_path)
            LOGGER.warning('Downloading recognition model, please wait. '
                           'This may take several minutes depending upon your network connection.')
            urlretrieve(model_url[model_file][0], model_path)
            assert calculate_md5(model_path) == model_url[model_file][1], corrupt_msg
            LOGGER.info('Download complete.')
        elif calculate_md5(model_path) != model_url[model_file][1]:
            if not self.download_enabled:
                raise FileNotFoundError("MD5 mismatch for %s and downloads disabled" % model_path)
            LOGGER.warning(corrupt_msg)
            os.remove(model_path)
            LOGGER.warning('Re-downloading the recognition model, please wait. '
                           'This may take several minutes depending upon your network connection.')
            urlretrieve(model_url[model_file][0], model_path)
            assert calculate_md5(model_path) == model_url[model_file][1], corrupt_msg
            LOGGER.info('Download complete')

        self.detector = get_detector(detector_path, self.device)
        self.recognizer, self.converter = get_recognizer(input_channel, output_channel,\
                                                         hidden_size, self.character, separator_list,\
                                                         dict_list, model_path, device = self.device)

    def readtext(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                 workers = 0, allowlist = None, blocklist = None, detail = 1,\
                 paragraph = False,\
                 contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, add_margin = 0.1):
        '''
        Parameters:
        file: file path or numpy-array or a byte stream object
        '''

        if type(image) == str:
            img = loadImage(image)  # can accept URL
            if image.startswith('http://') or image.startswith('https://'):
                tmp, _ = urlretrieve(image)
                img_cv_grey = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
                os.remove(tmp)
            else:
                img_cv_grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif type(image) == bytes:
            nparr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        elif type(image) == np.ndarray:
            img = image
            img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        text_box = get_textbox(self.detector, img, canvas_size, mag_ratio, text_threshold,\
                               link_threshold, low_text, False, self.device)
        horizontal_list, free_list = group_text_box(text_box, slope_ths, ycenter_ths, height_ths, width_ths, add_margin)

        # should add filter to screen small box out

        image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height = imgH)

        if allowlist:
            ignore_char = ''.join(set(self.character)-set(allowlist))
        elif blocklist:
            ignore_char = ''.join(set(blocklist))
        else:
            ignore_char = ''.join(set(self.character)-set(self.lang_char))

        if self.model_lang in ['chinese_tra','chinese_sim', 'japanese', 'korean']: decoder = 'greedy'
        result = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,\
                      ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                      workers, self.device)

        if paragraph:
            result = get_paragraph(result)

        if detail == 0:
            return [item[1] for item in result]
        else:
            return result
