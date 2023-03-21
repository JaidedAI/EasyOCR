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


### image visualization
reader = easyocr.Reader(['ko', 'en'], gpu = True) #, verbose = 0)
# result = reader.readtext('./demo_images/positive_image_10.png', paragraph=False)
# result = reader.readtext('./00110011001.jpg') #, paragraph=True)
result = reader.readtext('./datasets/picturetransfer/13.한국어글자체/02.인쇄체/unzip/sentence/03343000.png', paragraph=False)
result.sort()

img    = cv2.imread('./datasets/picturetransfer/13.한국어글자체/02.인쇄체/unzip/sentence/03343000.png')
img = Image.fromarray(img)
font = ImageFont.truetype('./AppleGothic.ttf', 15)
draw = ImageDraw.Draw(img)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(255, 3),dtype="uint8")
for i in result :
    x = i[0][0][0] 
    y = i[0][0][1]
    w = i[0][1][0] - i[0][0][0]
    h = i[0][2][1] - i[0][1][1]

    color_idx = random.randint(0,255) 
    color = [int(c) for c in COLORS[color_idx]]

    draw.rectangle(((x, y), (x+w, y+h)), outline=tuple(color), width=2)
    draw.text(((x + x + w) / 2 , y-2),str(i[1]), font=font, fill=tuple(color)) # (x + x + w) / 2 + 

plt.figure(figsize=(50,50))
plt.imshow(img)
plt.savefig("./test_aihub4.png")
plt.show()


## calculate accuracy & confidence score 
file_dir = '/workspace/datasets/picturetransfer/13.한국어글자체/02.인쇄체/unzip/sentence'
list_dir = natsort.natsorted(os.listdir(file_dir))
f = open("/workspace/inference/label_sentence.txt", 'r')
lines = f.readlines()
accuracy_list = []
confidence_list = []

with open(f'/workspace/inference/result_aihub_sentence.txt', 'w', encoding ='utf8') as log:
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
        
        confidence_mean = confidence_result.mean()
        
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

