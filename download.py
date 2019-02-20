

import urllib
import os

import platform
if 'Windows' in platform.platform():
    PATH = "\\".join( os.path.abspath(__file__).split('\\')[:-1])
else:
    PATH = "/".join( os.path.abspath(__file__).split('/')[:-1])
sys.path.append(PATH)

def weight():

    url = 'https://raw.githubusercontent.com/linsamtw/TaiwanTrainVerificationCode2text/master/cnn_weight/verificatioin_code.h5'  
    try:
        os.makedirs('{}/cnn_weight'.format(PATH))
    except:
        pass
    urllib.request.urlretrieve(url, '/{}/{}/verificatioin_code.h5'.format(PATH,'cnn_weight')) 


def ttf():
    
    url = 'https://raw.githubusercontent.com/linsamtw/TaiwanTrainVerificationCode2text/master/Courier-BoldRegular.ttf'  
    urllib.request.urlretrieve(url, '/{}/Courier-BoldRegular.ttf'.format(PATH)) 

    url = 'https://raw.githubusercontent.com/linsamtw/TaiwanTrainVerificationCode2text/master/Times%20Bold.ttf'  
    urllib.request.urlretrieve(url, '/{}/Times Bold.ttf'.format(PATH)) 


