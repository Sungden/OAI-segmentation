import matplotlib.pyplot as plt
from data_handling import load_train_data, load_validatation_data
from sklearn.utils import shuffle
from keras import callbacks
import os,time
from keras.models import Model
import numpy as np
import nibabel as nib
from functools import partial
import os
import numpy as np
from keras.models import Model 
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras import callbacks
from keras import backend as K
from keras.utils import plot_model
import h5py
from keras import optimizers
import time
from keras.layers import Dropout, Activation
from keras.models import Model
BASE=64
num_classes=5
patch_size=384


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#dice
def computeDice(img_true,img_pre):
    intersection = np.sum(img_true * img_pre)
    dsc= (2. * intersection+1) / (np.sum(img_true) + np.sum(img_pre)+1)
    return dsc

def get_unet_default():
    
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size,1))
    block1_conv1 = Conv2D(BASE, (3, 3),padding='same')(inputs)
    block1_conv1 = BatchNormalization(axis = 3)(block1_conv1)
    block1_conv1=  Activation('relu')(block1_conv1)
    block1_conv2 = Conv2D(BASE, (3, 3), padding='same')(block1_conv1)
    block1_conv2 = BatchNormalization(axis = 3)(block1_conv2)
    block1_conv2=  Activation('relu')(block1_conv2)
    block1_pool = MaxPooling2D(pool_size=(2, 2))(block1_conv2)

    
    block2_conv1 = Conv2D(BASE*2, (3, 3), padding='same')(block1_pool)
    block2_conv1 = BatchNormalization(axis = 3)(block2_conv1)
    block2_conv1= Activation('relu')(block2_conv1)
    block2_conv2 = Conv2D(BASE*2, (3, 3), padding='same')(block2_conv1)
    block2_conv2 = BatchNormalization(axis = 3)(block2_conv2)
    block2_conv2= Activation('relu')(block2_conv2)
    block2_pool = MaxPooling2D(pool_size=(2, 2))(block2_conv2)

    block3_conv1 = Conv2D(BASE*4, (3, 3), padding='same')(block2_pool)
    block3_conv1 = BatchNormalization(axis = 3)(block3_conv1)
    block3_conv1= Activation('relu')(block3_conv1)
    block3_conv2 = Conv2D(BASE*4, (3, 3),  padding='same')(block3_conv1)
    block3_conv2 = BatchNormalization(axis = 3)(block3_conv2)
    block3_conv2= Activation('relu')(block3_conv2)
    block3_conv3 = Conv2D(BASE*4, (3, 3),  padding='same')(block3_conv2)
    block3_conv3 = BatchNormalization(axis = 3)(block3_conv3)
    block3_conv3= Activation('relu')(block3_conv3)
    block3_pool = MaxPooling2D(pool_size=(2, 2))(block3_conv3)

    block4_conv1 = Conv2D(BASE*8, (3, 3), padding='same')(block3_pool)
    block4_conv1 = BatchNormalization(axis = 3)(block4_conv1)
    block4_conv1= Activation('relu')(block4_conv1) 
    block4_conv2 = Conv2D(BASE*8, (3, 3), padding='same')(block4_conv1)
    block4_conv2 = BatchNormalization(axis = 3)(block4_conv2)
    block4_conv2= Activation('relu')(block4_conv2) 
    block4_conv3 = Conv2D(BASE*8, (3, 3), padding='same')(block4_conv2)
    block4_conv3 = BatchNormalization(axis = 3)(block4_conv3)
    block4_conv3= Activation('relu')(block4_conv3)
    block4_pool = MaxPooling2D(pool_size=(2, 2))(block4_conv3)

    block5_conv1 = Conv2D(BASE*16, (3, 3),  padding='same')(block4_pool)
    block5_conv1 = BatchNormalization(axis = 3)(block5_conv1)
    block5_conv1= Activation('relu')(block5_conv1)
    block5_conv2 = Conv2D(BASE*16, (3, 3), padding='same')(block5_conv1)
    block5_conv2 = BatchNormalization(axis = 3)(block5_conv2)
    block5_conv2= Activation('relu')(block5_conv2)
    block5_conv3 = Conv2D(BASE*16, (3, 3), padding='same')(block5_conv2)
    block5_conv3 = BatchNormalization(axis = 3)(block5_conv3)
    block5_conv3= Activation('relu')(block5_conv3)
    
    up6 = concatenate([Conv2DTranspose(BASE*8, (2, 2), strides=(2, 2), padding='same')(block5_conv3), block4_conv3], axis=3)
    conv6 = Conv2D(BASE*8, (3, 3), padding='same')(up6)
    conv6 = BatchNormalization(axis = 3)(conv6)
    conv6= Activation('relu')(conv6)
    conv6 = Conv2D(BASE*8, (3, 3),  padding='same')(conv6)
    conv6 = BatchNormalization(axis = 3)(conv6)
    conv6= Activation('relu')(conv6)
    
    up7 = concatenate([Conv2DTranspose(BASE*4, (2, 2),strides=(2, 2), padding='same')(conv6), block3_conv3], axis=3)
    conv7 = Conv2D(BASE*4, (3, 3),  padding='same')(up7)
    conv7 = BatchNormalization(axis = 3)(conv7)
    conv7= Activation('relu')(conv7)
    conv7 = Conv2D(BASE*4, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization(axis = 3)(conv7)
    conv7= Activation('relu')(conv7)
    
    up8 = concatenate([Conv2DTranspose(BASE*2, (2, 2), strides=(2, 2), padding='same')(conv7), block2_conv2], axis=3)
    conv8 = Conv2D(BASE*2, (3, 3), padding='same')(up8)
    conv8 = BatchNormalization(axis = 3)(conv8)
    conv8= Activation('relu')(conv8)
    conv8 = Conv2D(BASE*2, (3, 3), padding='same')(conv8)
    conv8 = BatchNormalization(axis = 3)(conv8)
    conv8= Activation('relu')(conv8)
    
    up9 = concatenate([Conv2DTranspose(BASE, (2, 2), strides=(2, 2), padding='same')(conv8), block1_conv2], axis=3)
    conv9 = Conv2D(BASE, (3, 3), padding='same')(up9)
    conv9 = BatchNormalization(axis = 3)(conv9)
    conv9= Activation('relu')(conv9)
    conv9 = Conv2D(BASE, (3, 3), padding='same')(conv9)
    conv9 = BatchNormalization(axis = 3)(conv9)
    conv9= Activation('relu')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model



model=get_unet_default()
model.load_weights("/data/ydeng1/OAI/Unet_2D/outputs2/TC_new_model_last.h5")

output='/data/ydeng1/OAI/Unet_2D/results3/'

test_img=sorted(os.listdir("/data/ydeng1/OAI/slice_data/test_data/image/"))
test_gt=sorted(os.listdir("/data/ydeng1/OAI/slice_data/test_data/label/"))

DSC_FB=[]
DSC_FC=[]
DSC_TB=[]
DSC_TC=[]

time1=time.time()
# separate labels
def separate_labels(patch_2d_volume):
    result =np.empty(shape=[0,patch_size,patch_size,num_classes], dtype='int16')
    patch_3d_volume=np.expand_dims(patch_2d_volume, axis=0)
    N = patch_3d_volume.shape[0]
    # for each class do:
    for V in range(N):
        V_patch = patch_3d_volume[V , :, :]
        U  = np.unique(V_patch)
        unique_values = list(U)
        result_v =np.empty(shape=[patch_size,patch_size,0], dtype='int16')

        for label in range(0,num_classes):
            if label in unique_values:
                im_patch = V_patch == label
                im_patch = im_patch*1
            else:
                im_patch = np.zeros((V_patch.shape))
             
            im_patch = np.expand_dims(im_patch, axis=2) 
            result_v  = np.append(result_v,im_patch, axis=2)
        result_v = np.expand_dims(result_v, axis=0) 
        result  = np.append(result,result_v, axis=0)
    return result

for X,Y in zip(test_img,test_gt):
    X_test=np.load(os.path.join("/data/ydeng1/OAI/slice_data/test_data/image/",X))
    X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
    X_test=np.expand_dims(X_test,axis=0)
    X_test=np.expand_dims(X_test,axis=3)
    Y_test=np.load(os.path.join('/data/ydeng1/OAI/slice_data/test_data/label/',Y))
    #print(Y_test.shape,'888888888')
    #Y_test=separate_labels(label)
    
    
    get_6rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[6].output])
                                  
                                  
    get_70rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[70].output])
    # output in test mode = 0
    y_pred = get_6rd_layer_output([X_test, 0])[0]
    
    y_pred70 = get_70rd_layer_output([X_test, 0])[0]
    print(y_pred.shape,'tttttttttttt')
    print(y_pred50.shape,'tttttttttttt')
    plt.subplot(1,3,1)
    plt.imshow(X_test[0,:,:,0])
    plt.subplot(1,3,2)
    plt.imshow(y_pred[0,:,:,32])    
    plt.subplot(1,3,3)
    plt.imshow(y_pred70[0,:,:,2])
    plt.show()
        
    #y_pred = model.predict(X_test)
    #print(y_pred.shape,'tttttttttttt')
    '''
    plt.subplot(1,4,1)
    plt.imshow(y_pred[0,:,:,0],'gray')
    plt.subplot(1,4,2)
    plt.imshow(X_test[0,:,:,0],'gray')
    plt.subplot(1,4,3)
    plt.imshow(Y_test,'gray')
    y_pred[y_pred>=0.4]=1
    y_predi=np.squeeze(y_pred)
    plt.subplot(1,4,4)
    plt.imshow(y_predi,'gray')
    plt.show()    
    '''
    y_pred[y_pred>=0.7]=1
    y_pred[y_pred<0.7]=0
    y_predi=np.squeeze(y_pred)    
    #print(y_predi.shape,Y_test.shape,'iiiiiiiiiiiiiii')
    TC=(y_predi==1)*1
    TC_gt=(Y_test==4)*1  
   
    dice_TC=computeDice(TC,TC_gt)
    DSC_TC.append(dice_TC)
    a=dice_TC
    
    f = open("TC_result.txt", 'a')
    f.write(str(a))
    f.write("\n")
    
time2=time.time()    
print('mean dice of TC is:',np.mean(DSC_TC))
print('max dice of TC is:',np.max(DSC_TC))
print('min dice of TC is:',np.min(DSC_TC)) 
print('time is:',time2-time1)
