########################
import easyocr
import numpy as np
import cv2
import os
import random
import pickle
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
from functools import reduce
import natsort
from difflib import SequenceMatcher
import random

# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

## calculate accuracy & confidence score 
def calculate_score(file_dir, label_dir, result_dir, data="sentence"):

    list_dir = natsort.natsorted(os.listdir(file_dir))
    random.seed(101)
    sample_index = random.sample(range(len(list_dir)), int(len(list_dir)*0.1))
    sample_index.sort()
    list_dir = [list_dir[i] for i in sample_index]

    f = open(label_dir, 'r')
    lines = f.readlines()
    accuracy_list = []
    confidence_list = []

    with open(result_dir, 'w', encoding ='utf8') as log:
        for i in range(len(list_dir)):
            reader = easyocr.Reader(['ko', 'en'], gpu = True, verbose = 1)
            result = reader.readtext(os.path.join(file_dir, list_dir[i]), paragraph=False)
            result.sort()
            img = cv2.imread(os.path.join(file_dir, list_dir[i]))

            new_result = []
            confidence_result = []
            for item in result:
                tmp_result = reduce(lambda x,y: x+y, item[0])
                tmp_result.append(item[1])
                new_result.append(tmp_result)
                confidence_result.append(item[2])
                
            txts = ""
            for item in new_result:
                txts += " "
                txts += item[-1]
            if len(confidence_result) > 0:
                confidence_mean = sum(confidence_result) / len(confidence_result)
            
                txts = txts.replace(' ', '')
                answer = lines[sample_index[i]]
                # answer = lines[i]     
                answer = answer.replace(' ', '')[:-1]
                accuracy_list.append(SequenceMatcher(None, answer, txts).ratio())
                confidence_list.append(confidence_mean)
                
                print("text:\n", txts,"\nanswer:\n", answer, "\nscore:\n", SequenceMatcher(None, answer, txts).ratio(), "\nconfidence: \n", confidence_mean,"\n\n")
                log.write(f"file: {list_dir[i]} \ntext:\n {txts} \nanswer:\n {answer} \nscore:\n {SequenceMatcher(None, answer, txts).ratio()} \n confidence_result:\n {confidence_result} \n confidence_mean:\n {confidence_mean}\n" + "=" * 80 + '\n')
        

    print("accuracy_mean: ", sum(accuracy_list) / len(accuracy_list))
    print("confidence_mean: ", sum(confidence_list) / len(confidence_list))

    score_dict = {}
    score_dict["accuracy_list"] = accuracy_list
    score_dict["confidence_list"] = confidence_list

    with open(f'score_dict_{data}.pkl', 'wb') as s:
        pickle.dump(score_dict, s)

    f.close()


if __name__ == '__main__':
    
    file_dir = '/workspace/datasets/picturetransfer/13.한국어글자체/02.인쇄체/unzip/syllable'
    label_dir =  '/workspace/datasets/picturetransfer/13.한국어글자체/02.인쇄체/unzip/label_syllable.txt'
    result_dir = '/workspace/inference/result_aihub_syllable.txt'
    
    calculate_score(file_dir, label_dir, result_dir, data="syllable")