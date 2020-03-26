# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 08:51:25 2019

@author: ydeng1
"""
import shutil
from shutil import copyfile
import os
dst="/data/ydeng1/OAI/2018_ATEZ_MEDIA-Supplementary-Material-OAI-ZIB/OAI-ZIB/ZIB_images/"

f=open("/data/ydeng1/OAI/2018_ATEZ_MEDIA-Supplementary-Material-OAI-ZIB/OAI-ZIB/doc/oai_mri_paths.txt")
result=[]

for line2 in f:
    result.append(line2.strip('\n').split(',')[0])

for i in range(len(result)):
    print(result[i][9:40])
   # new_name=line2[30:37]
    dst1=result[i][15:22]
    dst1=os.path.join(dst,dst1)
    print(dst1)
    img_path=os.path.join('/data/ywang60/OAI_data/OAIBaselineImages/results/',result[i][9:40])
    shutil.copytree(img_path, dst1)