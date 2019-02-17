# Taiwan Train Verification Code 2 text ( 台鐵驗證碼轉文字 )


[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/linsamtw/TaiwanTrainVerificationCode2text/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/TaiwanTrainVerificationCode2text.svg)](https://badge.fury.io/py/TaiwanTrainVerificationCode2text)

-------------------
## introduce
提供各位開發程式訂票，其中驗證碼破解部分的方法，可直接使用，不須再回傳 image 用人工方式辨識。<br>
此 package 使用 keras & Tensorflow 建模&預測，需要安裝相依 package。

model test data 準確率約為 88%，使用 10 萬張圖進行 training。

-------------------

	pip3 install TaiwanTrainVerificationCode2text
    
cv2 比較難裝，以下提供安裝方法

    conda install -c menpo opencv
	# 你還需要這個
    pip3 install h5py

---------------------------------

## demo
	input 
![image](https://raw.githubusercontent.com/linsamtw/TaiwanTrainVerificationCode2text/master/WNBA8S.jpg)

	output
WNBA8S

--------------------

## exmaple 

    import os
    from TaiwanTrainVerificationCode2text import verification_code2text
    from TaiwanTrainVerificationCode2text import work_vcode 
    from TaiwanTrainVerificationCode2text import download 
    import TaiwanTrainVerificationCode2text
    PATH = TaiwanTrainVerificationCode2text.__path__[0]
    import cv2
    import matplotlib.pyplot as plt
    import random

	# 下載我 train 好的 weight，ttf 是驗證碼字形，用於以下生成模擬驗證碼
    download.weight()
    download.ttf()
	# 生成模擬驗證碼
    work_vcode.work_vcode_fun(10,'test_data',5)
    work_vcode.work_vcode_fun(10,'test_data',6)
    file_path = '{}/{}/'.format(PATH,'test_data')
    train_image_path = [file_path + i for i in os.listdir(file_path+'/')]
	# 隨機取一個當作 demo
    image_name = train_image_path[random.sample( range(len(train_image_path)) ,1)[0]]
	# 讀取圖片
    image = cv2.imread(image_name)
    # 畫圖
    plt.imshow(image)
	# 辨識，驗證碼轉文字
    text = verification_code2text.main(image)
    # 印出最後結果
    print(text)

最後結果就會類似 demo ，

-------------------------------

如果想自己 train，可以使用

[build_verification_code_cnn_model.py](https://github.com/linsamtw/TaiwanTrainVerificationCode2text/blob/master/build_verification_code_cnn_model.py)

稍微介紹主要 code

    def main():
        import work_vcode 
        #import time
        # 因為台鐵驗整碼是 5~6 隨機，因此必須生成 5 碼驗證碼& 6 碼驗證碼
        # 500 是 data 數量，建議數字為30000，500 只是 demo
        work_vcode.work_vcode_fun(500,'train_data',5)
        work_vcode.work_vcode_fun(500,'train_data',6)
		# 生成 test data，可根據自己喜好調整 data 數量
        work_vcode.work_vcode_fun(100,'test_data',5)
        work_vcode.work_vcode_fun(100,'test_data',6)
        self = build_verification_code_cnn_model()
        # 建模，最後 weight 會存放在 package_path/cnn_weight/verificatioin_code.h5
        self.build_model_process()  
   
   train 好後，可再使用以上 example code，會讀取你 train 好的 weight。

--------------------------

## 方法
由於驗證碼是26個英文字搭配10個數字，再加上隨機 5~6 碼，我將 class 分成 26 + 10 + null，37類，下去做分類。<br>
null 代表沒有此文字， 我將 NN 結構中的 output 設計為 6 dimension，將5碼也看成6碼，只是最後一個是 null，藉此同時處理 5 or 6 碼問題。<br>

----------------
## future
面臨到最大的問題是，null 佔比例太大，以上面生成 data 的 code 為例

        work_vcode.work_vcode_fun(500,'train_data',5)
        work_vcode.work_vcode_fun(500,'train_data',6)
先不管英文&數字數量，有一半的 class 是 null，因此在分類問題上，會造成嚴重的 imbalance 問題，即使 data 增加也不會改善問題，未來將搭配 object detection，改善準確率。

-----------------------
如有問題，可寄信給我 or 留言在 issues。

email : linsam.tw.github@gmail.com

