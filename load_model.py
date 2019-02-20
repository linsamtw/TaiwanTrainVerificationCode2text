
import os,sys
import platform
if 'Windows' in platform.platform():
    PATH = "\\".join( os.path.abspath(__file__).split('\\')[:-1])
else:
    PATH = "/".join( os.path.abspath(__file__).split('/')[:-1])
sys.path.append(PATH)
sys.path.append(PATH)
#-------------------------- set gpu using tf ---------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#=====================================================================


def load_model():

    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    
    tensor_in = Input((60, 200, 3))
    out = tensor_in
    out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Flatten()(out)
    out = Dropout(0.5)(out)
    out = [Dense(37, name='digit1', activation='softmax')(out),\
        Dense(37, name='digit2', activation='softmax')(out),\
        Dense(37, name='digit3', activation='softmax')(out),\
        Dense(37, name='digit4', activation='softmax')(out),\
        Dense(37, name='digit5', activation='softmax')(out),\
        Dense(37, name='digit6', activation='softmax')(out)]
    
    model = Model(inputs=tensor_in, outputs=out)
    
    # Define the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    if 'Windows' in platform.platform():
        model.load_weights('{}\\cnn_weight\\verificatioin_code.h5'.format(PATH)) 
    else:
        model.load_weights('{}/cnn_weight/verificatioin_code.h5'.format(PATH)) 
    
    return model


