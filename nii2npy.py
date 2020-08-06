import numpy as np
import nibabel as nib
import os
nii_dir='/data/pancreas_data/time_series_OAI_data/72_nii/'
dst='/data/pancreas_data/time_series_OAI_data/72_slice/'
if not os.path.exists(dst):
   os.makedirs(dst)
for i in sorted(os.listdir(nii_dir)):
  img=nib.load(os.path.join(nii_dir,i)).get_data()
  print(img.shape)
  for j in range(img.shape[0]):
    
    if j < 10:
        j = "00%d" % j
    elif j < 100:
        j = "0%d" % j
    else:
        j = str(j)
    name=i[0:7]+'_'+str(j)+'.npy'
    np.save(os.path.join(dst,name),img[int(j),:,:])