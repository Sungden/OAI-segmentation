# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 08:56:53 2019

@author: ydeng1
"""

import SimpleITK as sitk
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import pydicom
import cv2

import SimpleITK as sitk
import numpy as np
from PIL import Image as image

path1 = 'C:\\Users\\ydeng1\\Desktop\\OAI_data\\2018_ATEZ_MEDIA-Supplementary-Material-OAI-ZIB\\OAI-ZIB\\segmentation_masks'
path2 = 'C:\\Users\\ydeng1\\Desktop\\labels'



N = 507
W = 384
H = 384

if not os.path.exists(path2):
   os.makedirs(path2)

for volumeID in os.listdir(path1):
    mhd_file_name=volumeID[0:26]+'.mhd'
    mhd_file=os.path.join(path1,mhd_file_name)
    itkimage = sitk.ReadImage(mhd_file)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyImage=numpyImage.swapaxes(0,2)
    name=volumeID[0:26]+'.nii.gz'
    file2 = os.path.join(path2, name)
    img = nib.Nifti1Image(numpyImage, np.eye(4))
    nib.save(img,file2)
print ('File ' + volumeID + ' is saved in ' + path2 + ' .')
