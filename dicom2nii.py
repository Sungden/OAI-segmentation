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


N = 507
W = 384
H = 384

path1 = '/data/ydeng1/OAI/2018_ATEZ_MEDIA-Supplementary-Material-OAI-ZIB/OAI-ZIB/ZIB_images'
path2 = '/data/ydeng1/OAI/2018_ATEZ_MEDIA-Supplementary-Material-OAI-ZIB/images'
if not os.path.exists(path2):
   os.makedirs(path2)

for volumeID in os.listdir(path1):
    print(volumeID)
    print ('Processing File ' + volumeID)
    filename1 =  volumeID
    directory1 = os.path.join(path1, filename1)
    filename2 = volumeID + '.nii.gz'
    for path_, _, file_ in os.walk(directory1):
        L = len(file_)
        if L > 0:
            print ('  ' + str(L) + ' slices along the axial view.')
            data = np.zeros((W, H, L), dtype = np.int16)
            for f in sorted(file_):
                file1 = os.path.abspath(os.path.join(path_, f))
                image = pydicom.read_file(file1)
                sliceID = image.data_element("InstanceNumber").value - 1
                if image.pixel_array.shape[0] > 384 or image.pixel_array.shape[1] > 384:
                    exit('  Error: DICOM image does not fit ' + str(W) + 'x' + str(H) + ' size!')
                data[:, :, sliceID] = image.pixel_array
            file2 = os.path.join(path2, filename2)
            print(data.shape)
            data = cv2.flip(data, 0)
            data=data.swapaxes(0,2)
            print(data.shape)
            img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img,file2)
print ('File ' + volumeID + ' is saved in ' + file2 + ' .')
