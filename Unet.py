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
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Dropout
from functools import partial
from keras.optimizers import Adam

# ###multi-GPU
# import tensorflow as tf
# import keras.backend as K
# import keras.layers as KL
# import keras.models as KM
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
n_epoch = 200
batch_size = 1
num_classes=5
patch_size=384
BASE=64

K.set_image_data_format('channels_last')  # TF dimension ordering in this code      

img_dir='/data/ydeng1/OAI/OrganSegRSTN/data_path/images/'
label_dir='/data/ydeng1/OAI/OrganSegRSTN/data_path/labels/'

X_samples =os.listdir(img_dir)
Y_samples=os.listdir(label_dir)
X_samples=sorted(X_samples)
Y_samples=sorted(Y_samples)

train_X=X_samples[0:200]
train_Y=Y_samples[0:200]
validation_X=X_samples[251:301]
validation_Y=Y_samples[251:301]

for m in range(len(validation_X)):
    validation_X[m] = img_dir + validation_X[m]
    validation_Y[m] = label_dir + validation_Y[m]  
    
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
    for label_index in range(1,num_classes):
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

# 2D U-net depth=5
def get_unet_default():
    metrics = dice_coef
    include_label_wise_dice_coefficients = True;
    inputs = Input((patch_size, patch_size,1))
    block1_conv1 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(inputs)
    block1_conv2 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(block1_conv1)
    block1_pool = MaxPooling2D(pool_size=(2, 2))(block1_conv2)

    
    block2_conv1 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(block1_pool)
    block2_conv2 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(block2_conv1)
    block2_pool = MaxPooling2D(pool_size=(2, 2))(block2_conv2)

    block3_conv1 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(block2_pool)
    block3_conv2 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(block3_conv1)
    block3_conv3 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(block3_conv2)
    block3_pool = MaxPooling2D(pool_size=(2, 2))(block3_conv3)

    block4_conv1 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(block3_pool)
    block4_conv2 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(block4_conv1)
    block4_conv3 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(block4_conv2)
    block4_pool = MaxPooling2D(pool_size=(2, 2))(block4_conv3)

    block5_conv1 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(block4_pool)
    block5_conv2 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(block5_conv1)
    block5_conv3 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(block5_conv2)


    up6 = concatenate([Conv2DTranspose(BASE*8, (2, 2), strides=(2, 2), padding='same')(block5_conv3), block4_conv3], axis=3)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conv6)


    up7 = concatenate([Conv2DTranspose(BASE*4, (2, 2),strides=(2, 2), padding='same')(conv6), block3_conv3], axis=3)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(BASE*4, (3, 3), activation='relu', padding='same')(conv7)


    up8 = concatenate([Conv2DTranspose(BASE*2, (2, 2), strides=(2, 2), padding='same')(conv7), block2_conv2], axis=3)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(BASE*2, (3, 3), activation='relu', padding='same')(conv8)


    up9 = concatenate([Conv2DTranspose(BASE, (2, 2), strides=(2, 2), padding='same')(conv8), block1_conv2], axis=3)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(BASE, (3, 3), activation='relu', padding='same')(conv9)


    conv10 = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
#     if not isinstance(metrics, list):
#         metrics = [metrics]

#     if include_label_wise_dice_coefficients and num_classes > 1:
#         label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(num_classes)]
#         if metrics:
#             metrics = metrics + label_wise_dice_metrics
#         else:
#             metrics = label_wise_dice_metrics
     
#     model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5',by_name=True)
    
#     sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)

    
    
#     model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
 
# class ParallelModel(KM.Model):
#     """Subclasses the standard Keras Model and adds multi-GPU support.
#     It works by creating a copy of the model on each GPU. Then it slices
#     the inputs and sends a slice to each copy of the model, and then
#     merges the outputs together and applies the loss on the combined
#     outputs.
#     """

#     def __init__(self, keras_model, gpu_count):
#         """Class constructor.
#         keras_model: The Keras model to parallelize
#         gpu_count: Number of GPUs. Must be > 1
#         """
#         self.inner_model = keras_model
#         self.gpu_count = gpu_count
#         merged_outputs = self.make_parallel()
#         super(ParallelModel, self).__init__(inputs=self.inner_model.inputs,
#                                             outputs=merged_outputs)

#     def __getattribute__(self, attrname):
#         """Redirect loading and saving methods to the inner model. That's where
#         the weights are stored."""
#         if 'load' in attrname or 'save' in attrname:
#             return getattr(self.inner_model, attrname)
#         return super(ParallelModel, self).__getattribute__(attrname)

#     def summary(self, *args, **kwargs):
#         """Override summary() to display summaries of both, the wrapper
#         and inner models."""
#         super(ParallelModel, self).summary(*args, **kwargs)
#         self.inner_model.summary(*args, **kwargs)

#     def make_parallel(self):
#         """Creates a new wrapper model that consists of multiple replicas of
#         the original model placed on different GPUs.
#         """
#         # Slice inputs. Slice inputs on the CPU to avoid sending a copy
#         # of the full inputs to all GPUs. Saves on bandwidth and memory.
#         input_slices = {name: tf.split(x, self.gpu_count)
#                         for name, x in zip(self.inner_model.input_names,
#                                            self.inner_model.inputs)}

#         output_names = self.inner_model.output_names
#         outputs_all = []
#         for i in range(len(self.inner_model.outputs)):
#             outputs_all.append([])

#         # Run the model call() on each GPU to place the ops there
#         for i in range(self.gpu_count):
#             with tf.device('/gpu:%d' % i):
#                 with tf.name_scope('tower_%d' % i):
#                     # Run a slice of inputs through this replica
#                     zipped_inputs = zip(self.inner_model.input_names,
#                                         self.inner_model.inputs)
#                     inputs = [
#                         KL.Lambda(lambda s: input_slices[name][i],
#                                   output_shape=lambda s: (None,) + s[1:])(tensor)
#                         for name, tensor in zipped_inputs]
#                     # Create the model replica and get the outputs
#                     outputs = self.inner_model(inputs)
#                     if not isinstance(outputs, list):
#                         outputs = [outputs]
#                     # Save the outputs for merging back together later
#                     for l, o in enumerate(outputs):
#                         outputs_all[l].append(o)

#         # Merge outputs on CPU
#         with tf.device('/cpu:0'):
#             merged = []
#             for outputs, name in zip(outputs_all, output_names):
#                 # If outputs are numbers without dimensions, add a batch dim.
#                 def add_dim(tensor):
#                     """Add a dimension to tensors that don't have any."""
#                     if K.int_shape(tensor) == ():
#                         return KL.Lambda(lambda t: K.reshape(t, [1, 1]))(tensor)
#                     return tensor
#                 outputs = list(map(add_dim, outputs))

#                 # Concatenate
#                 merged.append(KL.Concatenate(axis=0, name=name)(outputs))
#         return merged

    
def load_batch_image(img_name,train_set = True):
    im=np.load(img_name)                 
    return im

def load_batch_label(label_name,train_set = True):
    label=separate_labels(np.load(label_name) )
    
    return label

# separate labels
def separate_labels(patch_3d_volume):
    result =np.empty(shape=[0,patch_size,patch_size,num_classes], dtype='int16')
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
            
            x=x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3]) #3D data to 2D data
            x=np.expand_dims(x,axis=3)
            y=y.reshape(y.shape[0]*y.shape[1],y.shape[2],y.shape[3],y.shape[4])            
            #print(x.shape,y.shape,'%%%%%%%%%')
                # extract patches which has center pixel lying inside mask    
            '''
            rows = []
            for j in range(y.shape[0]):        
              if np.count_nonzero(y[j,:,:,2])>=2000:     #TC at  least 500 pixels
                rows.append(j)
            rows.append(60)
            rows.append(100)
        
            yield x[min(rows):max(rows),:,:,:],y[min(rows):max(rows),:,:,:]
            '''
            
            x=np.array(np.split(x,4,axis=0)) #if not split, the memory will be out
            y=np.array(np.split(y,4,axis=0))
           
            
            for j in range(x.shape[0]):      #split 160 into 2x80

                 yield x[j,:,:,:,:],y[j,:,:,:,:]
 
model=get_unet_default()
model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5",by_name=True)

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

sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)            
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])     
   
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
early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=500, verbose=0, mode='min')
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
