

import os
from TaiwanTrainVerificationCode2text import verification_code2text
from TaiwanTrainVerificationCode2text import work_vcode 
from TaiwanTrainVerificationCode2text import download 
import TaiwanTrainVerificationCode2text
PATH = TaiwanTrainVerificationCode2text.__path__[0]
import cv2
import matplotlib.pyplot as plt
import random
from datetime import datetime

download.weight()
download.ttf()


work_vcode.work_vcode_fun(10,'test_data',5)
work_vcode.work_vcode_fun(10,'test_data',6)


file_path = '{}/{}/'.format(PATH,'test_data')
train_image_path = [file_path + i for i in os.listdir(file_path+'/')]

image_name = train_image_path[random.sample( range(len(train_image_path)) ,1)[0]]

s = datetime.now()
image = cv2.imread(image_name)
plt.imshow(image)

text = verification_code2text.main(image)
print(text)

t = datetime.now() - s
print(t)
