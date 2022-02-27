# -*- coding: utf-8 -*-

from .detection import get_detector, get_textbox
from .recognition import get_recognizer, get_text
from .utils import group_text_box, get_image_list, calculate_md5, get_paragraph,\
                   download_and_unzip, printProgressBar, diff, reformat_input,\
                   make_rotated_img_list, set_result_with_confidence,\
                   reformat_input_batched
from .config import *
from bidi.algorithm import get_display
import numpy as np
import cv2
import torch
import os
import sys
from PIL import Image
from logging import getLogger
import yaml

if sys.version_info[0] == 2:
    from io import open
    from six.moves.urllib.request import urlretrieve
    from pathlib2 import Path
else:
    from urllib.request import urlretrieve
    from pathlib import Path

LOGGER = getLogger(__name__)

class Reader(object):

    def __init__(self, lang_list, gpu=True, model_storage_directory=None,
                 user_network_directory=None, recog_network = 'standard',
                 download_enabled=True, detector=True, recognizer=True,
                 verbose=True, quantize=True, cudnn_benchmark=False):
        """Create an EasyOCR Reader

        Parameters:
            lang_list (list): Language codes (ISO 639) for languages to be recognized during analysis.

            gpu (bool): Enable GPU support (default)

            model_storage_directory (string): Path to directory for model data. If not specified,
            models will be read from a directory as defined by the environment variable
            EASYOCR_MODULE_PATH (preferred), MODULE_PATH (if defined), or ~/.EasyOCR/.

            user_network_directory (string): Path to directory for custom network architecture.
            If not specified, it is as defined by the environment variable
            EASYOCR_MODULE_PATH (preferred), MODULE_PATH (if defined), or ~/.EasyOCR/.

            download_enabled (bool): Enabled downloading of model data via HTTP (default).
        """
        self.download_enabled = download_enabled

        self.model_storage_directory = MODULE_PATH + '/model'
        if model_storage_directory:
            self.model_storage_directory = model_storage_directory
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)

        self.user_network_directory = MODULE_PATH + '/user_network'
        if user_network_directory:
            self.user_network_directory = user_network_directory
        Path(self.user_network_directory).mkdir(parents=True, exist_ok=True)
        sys.path.append(self.user_network_directory)

        if gpu is False:
            self.device = 'cpu'
            if verbose:
                LOGGER.warning('Using CPU. Note: This module is much faster with a GPU.')
        elif not torch.cuda.is_available():
            self.device = 'cpu'
            if verbose:
                LOGGER.warning('CUDA not available - defaulting to CPU. Note: This module is much faster with a GPU.')
        elif gpu is True:
            self.device = 'cuda'
        else:
            self.device = gpu
        self.recognition_models = recognition_models

        # check and download detection model
        detector_model = 'craft'
        corrupt_msg = 'MD5 hash mismatch, possible file corruption'
        detector_path = os.path.join(self.model_storage_directory, detection_models[detector_model]['filename'])
        if detector:
            if os.path.isfile(detector_path) == False:
                if not self.download_enabled:
                    raise FileNotFoundError("Missing %s and downloads disabled" % detector_path)
                LOGGER.warning('Downloading detection model, please wait. '
                               'This may take several minutes depending upon your network connection.')
                download_and_unzip(detection_models[detector_model]['url'], detection_models[detector_model]['filename'], self.model_storage_directory, verbose)
                assert calculate_md5(detector_path) == detection_models[detector_model]['md5sum'], corrupt_msg
                LOGGER.info('Download complete')
            elif calculate_md5(detector_path) != detection_models[detector_model]['md5sum']:
                if not self.download_enabled:
                    raise FileNotFoundError("MD5 mismatch for %s and downloads disabled" % detector_path)
                LOGGER.warning(corrupt_msg)
                os.remove(detector_path)
                LOGGER.warning('Re-downloading the detection model, please wait. '
                               'This may take several minutes depending upon your network connection.')
                download_and_unzip(detection_models[detector_model]['url'], detection_models[detector_model]['filename'], self.model_storage_directory, verbose)
                assert calculate_md5(detector_path) == detection_models[detector_model]['md5sum'], corrupt_msg

        # recognition model
        separator_list = {}

        if recog_network in ['standard'] + [model for model in recognition_models['gen1']] + [model for model in recognition_models['gen2']]:
            if recog_network in [model for model in recognition_models['gen1']]:
                model = recognition_models['gen1'][recog_network]
                recog_network = 'generation1'
                self.model_lang = model['model_script']
            elif recog_network in [model for model in recognition_models['gen2']]:
                model = recognition_models['gen2'][recog_network]
                recog_network = 'generation2'
                self.model_lang = model['model_script']
            else: # auto-detect
                unknown_lang = set(lang_list) - set(all_lang_list)
                if unknown_lang != set():
                    raise ValueError(unknown_lang, 'is not supported')
                # choose recognition model
                if lang_list == ['en']:
                    self.setModelLanguage('english', lang_list, ['en'], '["en"]')
                    model = recognition_models['gen2']['english_g2']
                    recog_network = 'generation2'
                elif 'th' in lang_list:
                    self.setModelLanguage('thai', lang_list, ['th','en'], '["th","en"]')
                    model = recognition_models['gen1']['thai_g1']
                    recog_network = 'generation1'
                elif 'ch_tra' in lang_list:
                    self.setModelLanguage('chinese_tra', lang_list, ['ch_tra','en'], '["ch_tra","en"]')
                    model = recognition_models['gen1']['zh_tra_g1']
                    recog_network = 'generation1'
                elif 'ch_sim' in lang_list:
                    self.setModelLanguage('chinese_sim', lang_list, ['ch_sim','en'], '["ch_sim","en"]')
                    model = recognition_models['gen2']['zh_sim_g2']
                    recog_network = 'generation2'
                elif 'ja' in lang_list:
                    self.setModelLanguage('japanese', lang_list, ['ja','en'], '["ja","en"]')
                    model = recognition_models['gen2']['japanese_g2']
                    recog_network = 'generation2'
                elif 'ko' in lang_list:
                    self.setModelLanguage('korean', lang_list, ['ko','en'], '["ko","en"]')
                    model = recognition_models['gen2']['korean_g2']
                    recog_network = 'generation2'
                elif 'ta' in lang_list:
                    self.setModelLanguage('tamil', lang_list, ['ta','en'], '["ta","en"]')
                    model = recognition_models['gen1']['tamil_g1']
                    recog_network = 'generation1'
                elif 'te' in lang_list:
                    self.setModelLanguage('telugu', lang_list, ['te','en'], '["te","en"]')
                    model = recognition_models['gen2']['telugu_g2']
                    recog_network = 'generation2'
                elif 'kn' in lang_list:
                    self.setModelLanguage('kannada', lang_list, ['kn','en'], '["kn","en"]')
                    model = recognition_models['gen2']['kannada_g2']
                    recog_network = 'generation2'
                elif set(lang_list) & set(bengali_lang_list):
                    self.setModelLanguage('bengali', lang_list, bengali_lang_list+['en'], '["bn","as","en"]')
                    model = recognition_models['gen1']['bengali_g1']
                    recog_network = 'generation1'
                elif set(lang_list) & set(arabic_lang_list):
                    self.setModelLanguage('arabic', lang_list, arabic_lang_list+['en'], '["ar","fa","ur","ug","en"]')
                    model = recognition_models['gen1']['arabic_g1']
                    recog_network = 'generation1'
                elif set(lang_list) & set(devanagari_lang_list):
                    self.setModelLanguage('devanagari', lang_list, devanagari_lang_list+['en'], '["hi","mr","ne","en"]')
                    model = recognition_models['gen1']['devanagari_g1']
                    recog_network = 'generation1'
                elif set(lang_list) & set(cyrillic_lang_list):
                    self.setModelLanguage('cyrillic', lang_list, cyrillic_lang_list+['en'],
                                          '["ru","rs_cyrillic","be","bg","uk","mn","en"]')
                    model = recognition_models['gen1']['cyrillic_g1']
                    recog_network = 'generation1'
                else:
                    self.model_lang = 'latin'
                    model = recognition_models['gen2']['latin_g2']
                    recog_network = 'generation2'
            self.character = model['characters']

            model_path = os.path.join(self.model_storage_directory, model['filename'])
            # check recognition model file
            if recognizer:
                if os.path.isfile(model_path) == False:
                    if not self.download_enabled:
                        raise FileNotFoundError("Missing %s and downloads disabled" % model_path)
                    LOGGER.warning('Downloading recognition model, please wait. '
                                   'This may take several minutes depending upon your network connection.')
                    download_and_unzip(model['url'], model['filename'], self.model_storage_directory, verbose)
                    assert calculate_md5(model_path) == model['md5sum'], corrupt_msg
                    LOGGER.info('Download complete.')
                elif calculate_md5(model_path) != model['md5sum']:
                    if not self.download_enabled:
                        raise FileNotFoundError("MD5 mismatch for %s and downloads disabled" % model_path)
                    LOGGER.warning(corrupt_msg)
                    os.remove(model_path)
                    LOGGER.warning('Re-downloading the recognition model, please wait. '
                                   'This may take several minutes depending upon your network connection.')
                    download_and_unzip(model['url'], model['filename'], self.model_storage_directory, verbose)
                    assert calculate_md5(model_path) == model['md5sum'], corrupt_msg
                    LOGGER.info('Download complete')
            self.setLanguageList(lang_list, model)

        else: # user-defined model
            with open(os.path.join(self.user_network_directory, recog_network+ '.yaml'), encoding='utf8') as file:
                recog_config = yaml.load(file, Loader=yaml.FullLoader)
            
            global imgH # if custom model, save this variable. (from *.yaml)
            if recog_config['imgH']:
                imgH = recog_config['imgH']
                
            available_lang = recog_config['lang_list']
            self.setModelLanguage(recog_network, lang_list, available_lang, available_lang)
            #char_file = os.path.join(self.user_network_directory, recog_network+ '.txt')
            self.character = recog_config['character_list']
            model_file = recog_network+ '.pth'
            model_path = os.path.join(self.model_storage_directory, model_file)
            self.setLanguageList(lang_list, None)

        dict_list = {}
        for lang in lang_list:
            dict_list[lang] = os.path.join(BASE_PATH, 'dict', lang + ".txt")

        if detector:
            self.detector = get_detector(detector_path, self.device, quantize, cudnn_benchmark=cudnn_benchmark)
        if recognizer:
            if recog_network == 'generation1':
                network_params = {
                    'input_channel': 1,
                    'output_channel': 512,
                    'hidden_size': 512
                    }
            elif recog_network == 'generation2':
                network_params = {
                    'input_channel': 1,
                    'output_channel': 256,
                    'hidden_size': 256
                    }
            else:
                network_params = recog_config['network_params']
            self.recognizer, self.converter = get_recognizer(recog_network, network_params,\
                                                         self.character, separator_list,\
                                                         dict_list, model_path, device = self.device, quantize=quantize)

    def setModelLanguage(self, language, lang_list, list_lang, list_lang_string):
        self.model_lang = language
        if set(lang_list) - set(list_lang) != set():
            if language == 'ch_tra' or language == 'ch_sim':
                language = 'chinese'
            raise ValueError(language.capitalize() + ' is only compatible with English, try lang_list=' + list_lang_string)

    def getChar(self, fileName):
        char_file = os.path.join(BASE_PATH, 'character', fileName)
        with open(char_file, "r", encoding="utf-8-sig") as input_file:
            list = input_file.read().splitlines()
            char = ''.join(list)
        return char

    def setLanguageList(self, lang_list, model):
        self.lang_char = []
        for lang in lang_list:
            char_file = os.path.join(BASE_PATH, 'character', lang + "_char.txt")
            with open(char_file, "r", encoding = "utf-8-sig") as input_file:
                char_list =  input_file.read().splitlines()
            self.lang_char += char_list
        if model:
            symbol = model['symbols']
        else:
            symbol = '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
        self.lang_char = set(self.lang_char).union(set(symbol))
        self.lang_char = ''.join(self.lang_char)

    def detect(self, img, min_size = 20, text_threshold = 0.7, low_text = 0.4,\
               link_threshold = 0.4,canvas_size = 2560, mag_ratio = 1.,\
               slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
               width_ths = 0.5, add_margin = 0.1, reformat=True, optimal_num_chars=None):

        if reformat:
            img, img_cv_grey = reformat_input(img)

        text_box_list = get_textbox(self.detector, img, canvas_size, mag_ratio,
                                    text_threshold, link_threshold, low_text,
                                    False, self.device, optimal_num_chars)

        horizontal_list_agg, free_list_agg = [], []
        for text_box in text_box_list:
            horizontal_list, free_list = group_text_box(text_box, slope_ths,
                                                        ycenter_ths, height_ths,
                                                        width_ths, add_margin,
                                                        (optimal_num_chars is None))
            if min_size:
                horizontal_list = [i for i in horizontal_list if max(
                    i[1] - i[0], i[3] - i[2]) > min_size]
                free_list = [i for i in free_list if max(
                    diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)

        return horizontal_list_agg, free_list_agg

    def recognize(self, img_cv_grey, horizontal_list=None, free_list=None,\
                  decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                  workers = 0, allowlist = None, blocklist = None, detail = 1,\
                  rotation_info = None,paragraph = False,\
                  contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                  y_ths = 0.5, x_ths = 1.0, reformat=True, output_format='standard'):

        if reformat:
            img, img_cv_grey = reformat_input(img_cv_grey)

        if allowlist:
            ignore_char = ''.join(set(self.character)-set(allowlist))
        elif blocklist:
            ignore_char = ''.join(set(blocklist))
        else:
            ignore_char = ''.join(set(self.character)-set(self.lang_char))

        if self.model_lang in ['chinese_tra','chinese_sim']: decoder = 'greedy'

        if (horizontal_list==None) and (free_list==None):
            y_max, x_max = img_cv_grey.shape
            horizontal_list = [[0, x_max, 0, y_max]]
            free_list = []

        # without gpu/parallelization, it is faster to process image one by one
        if ((batch_size == 1) or (self.device == 'cpu')) and not rotation_info:
            result = []
            for bbox in horizontal_list:
                h_list = [bbox]
                f_list = []
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,\
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                              workers, self.device)
                result += result0
            for bbox in free_list:
                h_list = []
                f_list = [bbox]
                image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = imgH)
                result0 = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,\
                              ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                              workers, self.device)
                result += result0
        # default mode will try to process multiple boxes at the same time
        else:
            image_list, max_width = get_image_list(horizontal_list, free_list, img_cv_grey, model_height = imgH)
            image_len = len(image_list)
            if rotation_info and image_list:
                image_list = make_rotated_img_list(rotation_info, image_list)
                max_width = max(max_width, imgH)

            result = get_text(self.character, imgH, int(max_width), self.recognizer, self.converter, image_list,\
                          ignore_char, decoder, beamWidth, batch_size, contrast_ths, adjust_contrast, filter_ths,\
                          workers, self.device)

            if rotation_info and (horizontal_list+free_list):
                # Reshape result to be a list of lists, each row being for 
                # one of the rotations (first row being no rotation)
                result = set_result_with_confidence(
                    [result[image_len*i:image_len*(i+1)] for i in range(len(rotation_info) + 1)])

        if self.model_lang == 'arabic':
            direction_mode = 'rtl'
            result = [list(item) for item in result]
            for item in result:
                item[1] = get_display(item[1])
        else:
            direction_mode = 'ltr'

        if paragraph:
            result = get_paragraph(result, x_ths=x_ths, y_ths=y_ths, mode = direction_mode)

        if detail == 0:
            return [item[1] for item in result]
        elif output_format == 'dict':
            return [ {'boxes':item[0],'text':item[1],'confident':item[2]} for item in result]
        else:
            return result

    def readtext(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                 workers = 0, allowlist = None, blocklist = None, detail = 1,\
                 rotation_info = None, paragraph = False, min_size = 20,\
                 contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detect(img, min_size, text_threshold,\
                                                 low_text, link_threshold,\
                                                 canvas_size, mag_ratio,\
                                                 slope_ths, ycenter_ths,\
                                                 height_ths,width_ths,\
                                                 add_margin, False)
        # get the 1st result from hor & free list as self.detect returns a list of depth 3
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        result = self.recognize(img_cv_grey, horizontal_list, free_list,\
                                decoder, beamWidth, batch_size,\
                                workers, allowlist, blocklist, detail, rotation_info,\
                                paragraph, contrast_ths, adjust_contrast,\
                                filter_ths, y_ths, x_ths, False, output_format)

        return result
    
    def readtextlang(self, image, decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                 workers = 0, allowlist = None, blocklist = None, detail = 1,\
                 rotation_info = None, paragraph = False, min_size = 20,\
                 contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detect(img, min_size, text_threshold,\
                                                 low_text, link_threshold,\
                                                 canvas_size, mag_ratio,\
                                                 slope_ths, ycenter_ths,\
                                                 height_ths,width_ths,\
                                                 add_margin, False)
        # get the 1st result from hor & free list as self.detect returns a list of depth 3
        horizontal_list, free_list = horizontal_list[0], free_list[0]
        result = self.recognize(img_cv_grey, horizontal_list, free_list,\
                                decoder, beamWidth, batch_size,\
                                workers, allowlist, blocklist, detail, rotation_info,\
                                paragraph, contrast_ths, adjust_contrast,\
                                filter_ths, y_ths, x_ths, False, output_format)
       
        char = []
        directory = 'characters/'
        for i in range(len(result)):
            char.append(result[i][1])
        
        def search(arr,x):
            g = False
            for i in range(len(arr)):
                if arr[i]==x:
                    g = True
                    return 1
            if g == False:
                return -1
        def tupleadd(i):
            a = result[i]
            b = a + (filename[0:2],)
            return b
        
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                with open ('characters/'+ filename,'rt',encoding="utf8") as myfile:  
                    chartrs = str(myfile.read().splitlines()).replace('\n','') 
                    for i in range(len(char)):
                        res = search(chartrs,char[i])
                        if res != -1:
                            if filename[0:2]=="en" or filename[0:2]=="ch":
                                print(tupleadd(i))

    def readtext_batched(self, image, n_width=None, n_height=None,\
                         decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                         workers = 0, allowlist = None, blocklist = None, detail = 1,\
                         rotation_info = None, paragraph = False, min_size = 20,\
                         contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                         text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                         canvas_size = 2560, mag_ratio = 1.,\
                         slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                         width_ths = 0.5, y_ths = 0.5, x_ths = 1.0, add_margin = 0.1, output_format='standard'):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        When sending a list of images, they all must of the same size,
        the following parameters will automatically resize if they are not None
        n_width: int, new width
        n_height: int, new height
        '''
        img, img_cv_grey = reformat_input_batched(image, n_width, n_height)

        horizontal_list_agg, free_list_agg = self.detect(img, min_size, text_threshold,\
                                                         low_text, link_threshold,\
                                                         canvas_size, mag_ratio,\
                                                         slope_ths, ycenter_ths,\
                                                         height_ths, width_ths,\
                                                         add_margin, False)
        result_agg = []
        # put img_cv_grey in a list if its a single img
        img_cv_grey = [img_cv_grey] if len(img_cv_grey.shape) == 2 else img_cv_grey
        for grey_img, horizontal_list, free_list in zip(img_cv_grey, horizontal_list_agg, free_list_agg):
            result_agg.append(self.recognize(grey_img, horizontal_list, free_list,\
                                            decoder, beamWidth, batch_size,\
                                            workers, allowlist, blocklist, detail, rotation_info,\
                                            paragraph, contrast_ths, adjust_contrast,\
                                            filter_ths, y_ths, x_ths, False, output_format))

        return result_agg
