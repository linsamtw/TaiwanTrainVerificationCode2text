

import os
import sys
#============================================
# my function / class
import platform
if 'Windows' in platform.platform():
    PATH = "\\".join( os.path.abspath(__file__).split('\\')[:-1])
else:
    PATH = "/".join( os.path.abspath(__file__).split('/')[:-1])
sys.path.append(PATH)
sys.path.append(PATH)
#============================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from load_model import load_model
#===============================================================


'''
import random

file_path = '{}/{}/'.format(PATH,'test_data')
train_image_path = [file_path + i for i in os.listdir(file_path+'/')]

image_name = train_image_path[random.sample( range(100) ,1)[0]]

image = cv2.imread(image_name)
plt.imshow(image)

text = main(image)
print(text)

'''

def validation(test_path):
    
    file_path = 'success_vcode'
    os.chdir(PATH)
    if file_path not in os.listdir():
        os.makedirs(file_path)
    if 'Windows' in platform.platform():
        file_path = '{}\\{}\\'.format(PATH,'success_vcode')
        test_image_path = [file_path + i for i in os.listdir(file_path+'\\')]
    else:
        file_path = '{}/{}/'.format(PATH,'success_vcode')
        test_image_path = [file_path + i for i in os.listdir(file_path+'/')]
    
    sum_count = len(test_image_path)
    data_set = np.ndarray(( sum_count , 60, 200,3), dtype=np.uint8)
    i=0
    #s = time.time()
    while( i < sum_count ):
        image_name = test_image_path[i]
        image = cv2.imread(image_name)
        data_set[i] = image
        i=i+1
        if i%50 == 0: print('Processed {} of {}'.format(i, sum_count ) )
            
#--------------------------------------------------
    real_labels = []
    for text in test_image_path:
        if 'Windows' in platform.platform():
            text = text.split('\\')
        else:
            text = text.split('/')
        text = text[len(text)-1]
        text_set = text.replace('.png','')
        real_labels.append(text_set)
    image = cv2.imread(image_name)
    plt.imshow(image)
    
    text = main(image)
    print(text)

def main(image):
#def verification_code_to_text(image_name):
    
    #os_path = os.getcwd()
    def change_character(pred_prob):
        
        total_set = []
        for i in range(65, 91):
            total_set.append( chr(i) )
        for i in range(10):
            total_set.append(str(i))
        total_set.append('')
        for i in range(len(pred_prob)):
            if pred_prob[i] == max( pred_prob ):
                value = (total_set[i])

        return value
    
    train_set = np.ndarray(( 1 , 60, 200,3), dtype=np.uint8)
    #image = cv2.imread(image_name)
    train_set[0] = image

    model = load_model()
    result = model.predict(train_set)

    resultlist = ''
    for i in range(len(result)):
        resultlist += change_character(result[i][0])

    #os.chdir(os_path)
    return resultlist



