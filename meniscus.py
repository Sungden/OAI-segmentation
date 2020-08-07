# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:00:59 2020

@author: Yang.D
"""

import numpy as np
import os
import nibabel as nib
import cv2
import imageio

FC_dir="/data/pancreas_data/time_series_OAI_data/18_predicted/FC_predict_slice/"
TC_dir="/data/pancreas_data/time_series_OAI_data/18_predicted/TC_predict_slice/"
cartlige_dir='/data/pancreas_data/time_series_OAI_data/18_predicted/cartlige_predict/'
cartlige_nii_dir='/data/pancreas_data/time_series_OAI_data/18_predicted/cartilage predcit nii/'
img_nii_dir='/data/pancreas_data/time_series_OAI_data/18_nii/' ##original images directory
cartlige20_dir='/data/pancreas_data/time_series_OAI_data/18_predicted/cartlige_20slices_nii/'
img20_dir='/data/pancreas_data/time_series_OAI_data/18_predicted/img_20slice_nii/'   #original images

dst_dir="/data/pancreas_data/time_series_OAI_data/18_predicted/meniscus/2D_slice_padding/" 
dst_dir1="/data/pancreas_data/time_series_OAI_data/18_predicted/meniscus/2D_slice_yuantu/" 
    
step=5    ##set to 1,2,3,4 for different steps

if step==1:
    ##segment cartlige##
    if not os.path.exists(cartlige_dir):
       os.makedirs(cartlige_dir)
    for i,j in zip(sorted(os.listdir(FC_dir)),sorted(os.listdir(TC_dir))):
      #print(i,j)
      FC=np.load(FC_dir+i)
      TC=np.load(TC_dir+j)
      cartlige=FC+TC
      np.save(cartlige_dir+i,cartlige)
      
elif step==2:
    ## save cartlige .npy format to .nii format ##
    img_list=sorted(os.listdir(cartlige_dir))    
    if not os.path.exists(cartlige_nii_dir):
       os.makedirs(cartlige_nii_dir)      
    print(len(img_list),'%%%%%%%%%%')
    for i in range(1,int(len(img_list)/160)+1):
      data=np.zeros((160,384,384))
      for num,j in zip(range(160),img_list[(i-1)*160:i*160]):
        image=np.load(cartlige_dir+j)
        data[num,:,:]=image
        img = nib.Nifti1Image(data, np.eye(4))
      nib.save(img, os.path.join(cartlige_nii_dir,j[0:7]+'.nii.gz'))

elif step==3:
    ## select the max slice, The front 10 slices and the back 10 slices##
    if not os.path.exists(cartlige20_dir):
       os.makedirs(cartlige20_dir)
    if not os.path.exists(img20_dir):
       os.makedirs(img20_dir)
       
    for i,j in zip(sorted(os.listdir(cartlige_nii_dir)),sorted(os.listdir(img_nii_dir))): 
      img=nib.load(cartlige_nii_dir+i).get_data()  #cartilage
      image=nib.load(img_nii_dir+j).get_data()     #original image
      num=[] 
      print(img.shape,'%%%%%%%%%%')
      for slice in range(img.shape[0]):
        im1=img[slice,:,:]
        a1=np.count_nonzero(im1)
        num.append(a1)
      max_num=num.index(max(num))
      data = nib.Nifti1Image(img[max_num-10:max_num+10,:,:], np.eye(4))
      nib.save(data, os.path.join(cartlige20_dir,i[0:7]+'.nii.gz'))
    
      data = nib.Nifti1Image(image[max_num-10:max_num+10,:,:], np.eye(4))
      nib.save(data, os.path.join(img20_dir,i[0:7]+'.nii.gz'))
      
else:       
    ## segment the meniscus region ##
    if not os.path.exists(dst_dir):
       os.makedirs(dst_dir)
    if not os.path.exists(dst_dir1):
       os.makedirs(dst_dir1)
       
    cartlige=sorted(os.listdir(cartlige20_dir))
    image=sorted(os.listdir(img20_dir))
    
    def load_data(img,gt,s):  
        x, y, w, h = cv2.boundingRect(gt)
        xi=x-30
        yi=y-30
        wi=w+60
        hi=h+60 
        gt1=gt[yi:(yi+hi),xi:(xi+wi)]     
        img1=img[yi:(yi+hi),xi:(xi+wi)]  
        s[10:(10+img1.shape[0]),10:(10+img1.shape[1])] =img1  
        return img1, gt1,xi,yi,wi,hi,s 
    
    for i,j in zip(cartlige,image):
        gt=nib.load(os.path.join(cartlige20_dir,i)).get_data()
        gt=np.array(gt,np.uint8)
        im=nib.load(os.path.join(img20_dir,j)).get_data()
        for n in range(gt.shape[0]):
          im_name=i[0:7]+'_'+str(n)+'.jpg'
          sample=np.zeros((400,400))
          X_train,Y_train,xi,yi,wi,hi,s=load_data(im[n,:,:],gt[n,:,:],sample) 
          print(X_train.shape,i,n,'$$$$$$$$$')
          print(s.shape,i,n,'##########') 
          path1=dst_dir+i[0:7]
          path2=dst_dir1+i[0:7]
          if not os.path.exists(path1):
             os.makedirs(path1)
          if not os.path.exists(path2):
             os.makedirs(path2)
          try:         
            imageio.imsave(os.path.join(path1,im_name),s)
            imageio.imsave(os.path.join(path2,im_name),X_train)
          except:
            pass
    


'''
## select those slices belong different KL score
from shutil import copyfile
cartlige_dir= '/data/pancreas_data/time_series_OAI_data/18_predicted/cartlige_20slices_nii/'
img_dir='/data/pancreas_data/time_series_OAI_data/18_predicted/img_20slice_nii/'  
dst1='/data/ydeng1/OAI/cartilage_data_KL/4/image/'   
dst2='/data/ydeng1/OAI/cartilage_data_KL/4/cartilage/'   

cartlige=sorted(os.listdir(cartlige_dir))
img=sorted(os.listdir(img_dir))

KL=[9031426 ,
9036287 ,
9065272 ,
9075815 ,
9081306 ,
9101066 ,
9114036 ,
9145695 ,
9156694 ,
9158391 ,
9160801 ,
9173792 ,
9177337 ,
9197466 ,
9208400 ,
9218935 ,
9225592 ,
9230504 ,
9267719 ,
9277154 ,
9319367 ,
9326657 ,
9341240 ,
9344856 ,
9351700 ,
9365968 ,
9390064 ,
9394203 ,
9395979 ,
9401202 ,
9413071 ,
9438523 ,
9473858 ,
9478504 ,
9487462 ,
9494867 ,
9495873 ,
9504935 ,
9680800 ,
9710479 ,
9781749 ,
9933836 ,
9951449 ,

]
for i in cartlige:
  #print(i[0:7])
  #print(KL)
  j=i[0:7]
  if int(j) in KL:   
    copyfile(os.path.join(img_dir,i),os.path.join(dst1,i))
    copyfile(os.path.join(cartlige_dir,i),os.path.join(dst2,i))
'''


