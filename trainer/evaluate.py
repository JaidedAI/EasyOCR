import argparse
import os
import re
import sys
import unicodedata

import csv
import cv2
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir ))
import easyocr


def join_diacritic(text, mode='NFC'):
    """
    基底文字と濁点・半濁点を結合
    """
    # str -> bytes
    bytes_text = text.encode()

    # 濁点Unicode結合文字置換
    bytes_text = re.sub(b'\xe3\x82\x9b', b'\xe3\x82\x99', bytes_text)
    bytes_text = re.sub(b'\xef\xbe\x9e', b'\xe3\x82\x99', bytes_text)

    # 半濁点Unicode結合文字置換
    bytes_text = re.sub(b'\xe3\x82\x9c', b'\xe3\x82\x9a', bytes_text)
    bytes_text = re.sub(b'\xef\xbe\x9f', b'\xe3\x82\x9a', bytes_text)

    # bytet -> str
    text = bytes_text.decode()

    # 正規化
    text = unicodedata.normalize(mode, text)

    return text
def gen_reader(lang_list, gpu=False, recog_network='standard', model_storage_directory=None, user_network_directory=None):
    reader = easyocr.Reader(
        lang_list=lang_list,
        gpu=gpu,
        recog_network=recog_network,
        model_storage_directory=model_storage_directory,
        user_network_directory=user_network_directory
        )
    return reader


def evaluate(img_dir_path: str, label_path: str,
         gpu:bool =False,
         detail: str=None,
         recog_network: str=None,
         model_storage_directory: str=None,
         user_network_directory: str=None,
         trim_quotes: bool=True) -> float:

    reader = gen_reader(
        lang_list=['ja', 'en'],
        gpu=gpu,
        recog_network=recog_network,
        model_storage_directory=model_storage_directory,
        user_network_directory=user_network_directory
        )

    detail_result = [['filename', 'gt_words', 'pred_words', 'correct', 'confidence']]
    with open(label_path) as f:
        csv_reader = csv.reader(f)

        n_data = 0
        n_correct = 0
        for row in tqdm(csv_reader):
            if row[0] == 'filename':
                continue
            filename = row[0]
            gt_words = row[1]
            img_path = os.path.join(img_dir_path, filename)

            img_cv_grey = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            result = reader.recognize(img_cv_grey, horizontal_list=None, free_list=None,
                        decoder = 'greedy', beamWidth= 5, batch_size = 1,
                        workers = 0, allowlist = None, blocklist = ' ', detail = 1,
                        rotation_info = None,paragraph = False,
                        contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,
                        y_ths = 0.5, x_ths = 1.0, reformat=True, output_format='standard')

            pred_words = result[0][1] # word
            if trim_quotes:
                pred_words = pred_words.strip('"')
            conf = result[0][2] # confidence

            correct = join_diacritic(gt_words) == join_diacritic(pred_words)
            if correct:
                n_correct += 1
            n_data += 1
            detail_result.append([filename, gt_words, pred_words, correct, conf])

    if detail is not None:
        with open(detail, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(detail_result)

    return n_correct / n_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img')
    parser.add_argument('-l', '--label')
    parser.add_argument('-g', '--gpu', action='store_true')
    parser.add_argument('-d', '--detail', default=None)
    parser.add_argument('-r', '--recog_network', default='standard')
    parser.add_argument('-m', '--model_storage_directory', default=None)
    parser.add_argument('-u', '--user_network_directory', default=None)
    parser.add_argument('-t', '--trim_quotes', action='store_true', default=False)
    args = parser.parse_args()

    result = evaluate(args.img, args.label, args.gpu, args.detail,
                  args.recog_network, args.model_storage_directory, args.user_network_directory)
    print(result)