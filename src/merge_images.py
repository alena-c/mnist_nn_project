#!/usr/bin/env python
# coding: utf-8

# In[31]:


import cv2
import numpy
import glob
import os

dir = '.' # current directory
ext = ".png" # any extension

pathname = os.path.join(dir, "lr_*" + ext)
images = [cv2.imread(img) for img in glob.glob(pathname)]

height = max(image.shape[0] for image in images)
width = sum(image.shape[1] for image in images)
output = numpy.zeros((height,width,3))

y = 0
for image in images:
    h,w,d = image.shape
    output[0:h,y:y+w] = image
    y +=w
    
cv2.imwrite('filename', output)


# In[ ]:




