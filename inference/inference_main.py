########################
import easyocr
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
from functools import reduce
import natsort
from difflib import SequenceMatcher


# ## calculate accuracy & confidence score 
# file_dir = '/workspace/inference/positive'
# list_dir = natsort.natsorted(os.listdir(file_dir))
# f = open("/workspace/inference/label_positive.txt", 'r')
# lines = f.readlines()
# accuracy_list = []

# with open(f'/workspace/inference/result_main.txt', 'w', encoding ='utf8') as log:
#     for i in range(len(list_dir)):
#         reader = easyocr.Reader(['ko', 'en'], gpu = True, verbose = 0)
#         result = reader.readtext(os.path.join(file_dir, list_dir[i]), paragraph=True)
#         img    = cv2.imread(os.path.join(file_dir, list_dir[i]))

#         new_result = []
#         for item in result:
#             tmp_result = reduce(lambda x,y: x+y, item[0])
#             tmp_result.append(item[1])
#             new_result.append(tmp_result)

#         txts = ""
#         for item in new_result:
#             txts += " "
#             txts += item[-1]

#         txts = txts.replace(' ', '')
#         answer = lines[i]
#         answer = answer.replace(' ', '')
#         accuracy_list.append(SequenceMatcher(None, answer, txts).ratio())
#         print("text:\n", txts,"\nanswer:\n", answer, "score:\n",SequenceMatcher(None, answer, txts).ratio(), "\n\n")
#         log.write(f"file: {list_dir[i]} \ntext:\n {txts} \nanswer:\n {answer} score:\n {SequenceMatcher(None, answer, txts).ratio()} \n" + "=" * 80 + '\n')
    

# print("accuracy_mean: ", sum(accuracy_list) / len(accuracy_list))


## calculate accuracy & confidence score 
file_dir = '/workspace/inference/positive'
list_dir = natsort.natsorted(os.listdir(file_dir))
f = open("/workspace/inference/label_positive.txt", 'r')
lines = f.readlines()
accuracy_list = []
confidence_list = []

with open(f'/workspace/inference/result_ver1.0.txt', 'w', encoding ='utf8') as log:
    for i in range(len(list_dir)):
        reader = easyocr.Reader(['ko', 'en'], gpu = True, verbose = 0)
        result = reader.readtext(os.path.join(file_dir, list_dir[i]), paragraph=False)
        img    = cv2.imread(os.path.join(file_dir, list_dir[i]))

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
        
        confidence_mean = sum(confidence_result) / len(confidence_result)
        
        txts = txts.replace(' ', '')
        answer = lines[i]
        answer = answer.replace(' ', '')
        accuracy_list.append(SequenceMatcher(None, answer, txts).ratio())
        confidence_list.append(confidence_mean)
        
        print("text:\n", txts,"\nanswer:\n", answer, "score:\n",SequenceMatcher(None, answer, txts).ratio(), "\nconfidence: \n", confidence_mean,"\n\n")
        log.write(f"file: {list_dir[i]} \ntext:\n {txts} \nanswer:\n {answer} score:\n {SequenceMatcher(None, answer, txts).ratio()} \n confidence_result:\n {confidence_result} \n confidence_mean:\n {confidence_mean}\n" + "=" * 80 + '\n')

    

print("accuracy_mean: ", sum(accuracy_list) / len(accuracy_list))
print("confidence_mean: ", sum(confidence_list) / len(confidence_list))


f.close() 