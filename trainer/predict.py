import sys
import os
import argparse
import glob
import json
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir ))
import easyocr
import img_utils

class Predictor():

    def __init__(self,
                 lang_list=['ja', 'en'],
                 recog_network='standard',
                 model_storage_directory=None,
                 user_network_directory=None,
                 gpu=True,
                 trim_quotes=True):
        self.model = self.load_model(
            lang_list=lang_list,
            gpu=gpu,
            recog_network=recog_network,
            model_storage_directory=model_storage_directory,
            user_network_directory=user_network_directory
            )
        self.trim_quotes = trim_quotes

    def gen_reader(self,
                lang_list,
                gpu=False,
                recog_network='standard',
                model_storage_directory=None,
                user_network_directory=None):
        reader = easyocr.Reader(
            lang_list=lang_list,
            gpu=gpu,
            recog_network=recog_network,
            model_storage_directory=model_storage_directory,
            user_network_directory=user_network_directory
        )
        return reader

    def load_model(self,
                lang_list: list,
                recog_network: str,
                model_storage_directory: str,
                user_network_directory: str,
                gpu=True):
        return self.gen_reader(
            lang_list=lang_list,
            gpu=gpu,
            recog_network=recog_network,
            model_storage_directory=model_storage_directory,
            user_network_directory=user_network_directory)


    def run_ocr(self, img_path):

        results = self.model.readtext(img_path, blocklist='')

        annotations = []
        for points, word, conf in results:

            if self.trim_quotes:
                word = word.strip('"')

            x1 = int(min(p[0] for p in points))
            y1 = int(min(p[1] for p in points))
            x2 = int(max(p[0] for p in points))
            y2 = int(max(p[1] for p in points))

            attributes = [{
                'type': 'text',
                'name': 'text',
                'key': 'text',
                'value': word
            }]

            annotations.append({
                'type': 'bbox',
                'value': 'str',
                'attributes': attributes,
                'points': [x1, y1, x2, y2],
                'confidence': conf
            })

        return annotations


    def run_single_image(self, img: Tuple[object, str]) -> list:

        return self.run_ocr(img)


    def run_batch(self,
            in_glob_path: str,
            out_dir: str,
            visualize=False) -> list:

        annos = []

        for img_path in tqdm(glob.glob(in_glob_path)):
            result = self.run_ocr(img_path)

            if visualize:
                img = self.vis(img_path, result)
                cv2.imwrite(os.path.join(out_dir, img_path.split('/')[-1]), img)

            annos.append(result)

        if out_dir:
            with open(os.path.join(out_dir, 'annotations.json'), 'w') as f:
                json.dump(annos, f, indent=4, ensure_ascii=False)

        return annos

    def vis(self, img_path, result):

        img = cv2.imread(img_path)
        font_path = os.getenv('JFONT_PATH')

        for annotation in result:
            bbox = annotation['points']
            if len(annotation['attributes']) > 0 and annotation['attributes'][0]['key'] == 'text':
                text = annotation['attributes'][0]['value']
            else:
                text = ''

            pts = np.array(bbox, np.int32)
            img = cv2.rectangle(img, pts[:2], pts[2:], (0, 255, 255))
            img = img_utils.cv2_putText(
                img, text, [pts[0], pts[1]+10], font_path, 20, (255, 100, 255)).copy()

        return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-l', '--lang_list', type=str,
                        nargs='*', default=['ja', 'en'])
    parser.add_argument('-r', '--recog_network', type=str, default='standard')
    parser.add_argument('-m', '--model_storage_directory',
                        type=str, default=None)
    parser.add_argument('-u', '--user_network_directory',
                        type=str, default=None)
    parser.add_argument('-g', '--gpu', action='store_true', default=False)
    parser.add_argument('-t', '--trim_quotes', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)

    args = parser.parse_args()

    model = Predictor(
                lang_list=args.lang_list,
                recog_network=args.recog_network,
                model_storage_directory=args.model_storage_directory,
                user_network_directory=args.user_network_directory,
                gpu=args.gpu,
                trim_quotes=args.trim_quotes)

    n_input = len(glob.glob(args.input))

    if n_input > 1:
        result = model.run_batch(args.input, args.output, args.visualize)
    elif n_input == 1:
        result = model.run_single_image(args.input)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
    else :
        print('input file not found')


