import random,os
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import h5py, keras
from keras import optimizers
from keras import callbacks
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import plot_model
from sklearn.feature_extraction.image import extract_patches
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Dropout,BatchNormalization
from functools import partial
from keras.optimizers import Adam
from tensorflow.python.ops import array_ops
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout, Activation
# ###multi-GPU
# import tensorflow as tf
# import keras.backend as K
# import keras.layers as KL
# import keras.models as KM
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
n_epoch = 500
batch_size =20
num_classes=5 #background are  included
patch_size=384
BASE=64

K.set_image_data_format('channels_last')  # TF dimension ordering in this code      

img_dir='/data/ydeng1/OAI/slice_data/train_data/image/'
label_dir='/data/ydeng1/OAI/slice_data/train_data/label/'

X_samples =os.listdir(img_dir)
Y_samples=os.listdir(label_dir)
X_samples=sorted(X_samples)
Y_samples=sorted(Y_samples)
train_X=X_samples
train_Y=Y_samples

test_dir='/data/ydeng1/OAI/slice_data/test_data/image/'
test_label_dir='/data/ydeng1/OAI/slice_data/test_data/label/'

X_test =os.listdir(test_dir)
Y_test=os.listdir(test_label_dir)
X_test=sorted(X_test)
Y_test=sorted(Y_test)
validation_X=X_test[0:3000]
validation_Y=Y_test[0:3000]

for m in range(len(validation_X)):
    validation_X[m] = test_dir + validation_X[m]
    validation_Y[m] = test_label_dir + validation_Y[m]  
    
for m in range(len(train_X)):
    train_X[m] = img_dir + train_X[m]
    train_Y[m] = label_dir + train_Y[m] 
    
# compute dsc
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# proposed loss function
# proposed loss function
def dice_coef_loss(y_true, y_pred):
    distance = 0
    for label_index in range(0,num_classes):  # background are not included
        dice_coef_class = dice_coef(y_true[:,:,:,label_index], y_pred[:, :,:,label_index])
        distance = 1 - dice_coef_class + distance
    return distance

# dsc per class
def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coef(y_true[:,:,:,label_index], y_pred[:, :,:,label_index])

# get label dsc
def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

# focal loss with multi label
def Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2):
    """
    focal loss for multi-class classification
    fl(pt) = -alpha*(1-pt)^(gamma)*log(pt)
    :param y_true: ground truth one-hot vector shape of [batch_size, nb_class]
    :param y_pred: prediction after softmax shape of [batch_size, nb_class]
    :param alpha:
    :param gamma:
    :return:
    """
    # # parameters
    # alpha = 0.25
    # gamma = 2

    # To avoid divided by zero
    y_pred += K.epsilon()

    # Cross entropy
    ce = -y_true * K.log(y_pred)

    # Not necessary to multiply y_true(cause it will multiply with CE which has set unconcerned index to zero ),
    # but refer to the definition of p_t, we do it
    weight = np.power(1 - y_pred, gamma) * y_true

    # Now fl has a shape of [batch_size, nb_class]
    # alpha should be a step function as paper mentioned, but it doesn't matter like reason mentioned above
    # (CE has set unconcerned index to zero)
    #
    # alpha_step = tf.where(y_true, alpha*np.ones_like(y_true), 1-alpha*np.ones_like(y_true))
    fl = ce * weight * alpha

    # Both reduce_sum and reduce_max are ok
    reduce_fl = K.max(fl, axis=-1)

    return reduce_fl


'''
# 2D U-net depth=5
def get_unet_default():
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size,1))
    block1_conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(inputs)
    block1_conv1 = BatchNormalization(axis = 3)(block1_conv1)
    block1_conv2 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(block1_conv1)
    block1_conv2 = BatchNormalization(axis = 3)(block1_conv2)
    block1_pool = MaxPooling2D(pool_size=(2, 2))(block1_conv2)

    
    block2_conv1 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(block1_pool)
    block2_conv1 = BatchNormalization(axis = 3)(block2_conv1)
    block2_conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(block2_conv1)
    block2_conv2 = BatchNormalization(axis = 3)(block2_conv2)
    block2_pool = MaxPooling2D(pool_size=(2, 2))(block2_conv2)

    block3_conv1 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(block2_pool)
    block3_conv1 = BatchNormalization(axis = 3)(block3_conv1)
    block3_conv2 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(block3_conv1)
    block3_conv2 = BatchNormalization(axis = 3)(block3_conv2)
    block3_conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(block3_conv2)
    block3_conv3 = BatchNormalization(axis = 3)(block3_conv3)
    block3_pool = MaxPooling2D(pool_size=(2, 2))(block3_conv3)

    block4_conv1 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(block3_pool)
    block4_conv1 = BatchNormalization(axis = 3)(block4_conv1)
    block4_conv2 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(block4_conv1)
    block4_conv2 = BatchNormalization(axis = 3)(block4_conv2)
    block4_conv3 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(block4_conv2)
    block4_conv3 = BatchNormalization(axis = 3)(block4_conv3)
    block4_pool = MaxPooling2D(pool_size=(2, 2))(block4_conv3)

    block5_conv1 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(block4_pool)
    block5_conv1 = BatchNormalization(axis = 3)(block5_conv1)
    block5_conv2 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(block5_conv1)
    block5_conv2 = BatchNormalization(axis = 3)(block5_conv2)
    block5_conv3 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(block5_conv2)
    block5_conv3 = BatchNormalization(axis = 3)(block5_conv3)

    up6 = concatenate([Conv2DTranspose(BASE*8, (2, 2), strides=(2, 2), padding='same')(block5_conv3), block4_conv3], axis=3)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization(axis = 3)(conv6)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization(axis = 3)(conv6)

    up7 = concatenate([Conv2DTranspose(BASE*4, (2, 2),strides=(2, 2), padding='same')(conv6), block3_conv3], axis=3)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization(axis = 3)(conv7)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization(axis = 3)(conv7)

    up8 = concatenate([Conv2DTranspose(BASE*2, (2, 2), strides=(2, 2), padding='same')(conv7), block2_conv2], axis=3)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization(axis = 3)(conv8)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization(axis = 3)(conv8)

    up9 = concatenate([Conv2DTranspose(BASE, (2, 2), strides=(2, 2), padding='same')(conv8), block1_conv2], axis=3)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization(axis = 3)(conv9)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization(axis = 3)(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model
'''

# 2D U-net depth=5
def get_unet_default():
    metrics = dice_coef
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

    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model




    
def load_batch_image(img_name,train_set = True):
    im=np.load(img_name)  
    im=(im-np.min(im))/(np.max(im)-np.min(im))               
    return im

def load_batch_label(label_name,train_set = True):
    label=separate_labels(np.load(label_name) )
    return label

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


def GET_DATASET_SHUFFLE(X_samples,y_samples,batch_size,train_set = True):  
    while True:
        cc = list(zip(X_samples, y_samples))
        random.shuffle(cc)
        X_samples[:], y_samples[:] = zip(*cc)

        batch_num = int(len(X_samples) / batch_size) #125
        max_len = batch_num * batch_size #125

        X_samples = np.array(X_samples[:max_len]) #125
        y_samples = np.array(y_samples[:max_len])

        X_batches = np.split(X_samples, batch_num) #125
        y_batches = np.split(y_samples, batch_num)

        for i in range(len(X_batches)):
            x = np.array(list(map(load_batch_image,  X_batches[i],[True for _ in range(batch_size)])))
            y = np.array(list(map(load_batch_label,  y_batches[i],[True for _ in range(batch_size)])))
            
            #print(x.shape,y.shape,'$$$$$$$$$$$$$')
            
            x=np.expand_dims(x,axis=3)
            y=y.reshape(y.shape[0]*y.shape[1],y.shape[2],y.shape[3],y.shape[4])            
            #print(x.shape,y.shape,'%%%%%%%%%')
            
            # extract patches which has center pixel lying inside mask    
            rows = []
            for m in range(y.shape[0]):                      
              if np.count_nonzero(y[m,:,:,3])>=20:     #TC at  least 500 pixels
                rows.append(m)
            yield x[rows,:,:,:],y[rows,:,:,:]

model=get_unet_default()
#model.load_weights("/data/ydeng1/OAI/Unet_2D/outputs2/weights_old.h5",by_name=True)

# GPU_COUNT = 2
# model = ParallelModel(model, GPU_COUNT)
model.summary()

metrics = dice_coef
include_label_wise_dice_coefficients = True;

if not isinstance(metrics, list):
        metrics = [metrics]

if include_label_wise_dice_coefficients and num_classes > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(num_classes)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

#sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)            
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])     
#model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
   
if 'outputs2' not in os.listdir(os.curdir):
    os.mkdir('outputs2')

print('-'*30)
print('Fitting model...')
print('-'*30)
#============================================================================
print('training starting..')
log_filename = 'outputs2/' + 'new_model_train2.csv' 
#Callback that streams epoch results to a csv file.

csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='min')
checkpoint_filepath = 'outputs2/' + 'weights2.h5'
checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log, early_stopping, checkpoint]

model.fit_generator(
  GET_DATASET_SHUFFLE(train_X, train_Y, batch_size),
  validation_data=GET_DATASET_SHUFFLE(validation_X, validation_Y,batch_size),
  steps_per_epoch=int(len(train_X)/batch_size),
  max_queue_size=1,
  validation_steps=int(len(validation_X)/batch_size), 
  epochs=n_epoch,
  verbose=1,
  callbacks=callbacks_list
  )

model_name = 'outputs2/' +  'new_model_last.h5'
model.save(model_name)  # creates a HDF5 file 'my_model.h5'
