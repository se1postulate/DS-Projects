# -*- coding: utf-8 -*-
"""
Created on Tue May 11 08:03:22 2021

@author: AMK
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import easygui
ImagePath=easygui.fileopenbox()
img = cv2.imread(ImagePath)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((2,2),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(closing,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)

# Threshold
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [0,0,0]


originalmage = cv2.imread(ImagePath)
originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)
    #print(image)  # image is stored in form of numbers
    # confirm that image is chosen
gaussian=cv2.GaussianBlur(originalmage,(5,5),0)
#gaussian = cv2.resize(gaussian, (960, 540))

ReSized1 = cv2.resize(originalmage, (960, 540))
#plt.imshow(ReSized1, cmap='gray')
grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
ReSized2 = cv2.resize(grayScaleImage, (960, 540))
#
smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
ReSized3 = cv2.resize(smoothGrayScale, (960, 540))
getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, 
cv2.ADAPTIVE_THRESH_MEAN_C, 
cv2.THRESH_BINARY, 9, 9)
ReSized4 = cv2.resize(getEdge, (960, 540))
    #plt.imshow(ReSized4, cmap='gray')
    #applying bilateral filter to remove noise 
    #and keep edge sharp as required
colorImage = cv2.bilateralFilter(originalmage, 9, 300, 300)
    
ReSized5 = cv2.resize(colorImage, (960, 540))
    #plt.imshow(ReSized5, cmap='gray')
    #masking edged image with our "BEAUTIFY" image
#    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
cartoonImage = cv2.bitwise_and(gaussian,img, mask=getEdge)
ReSized6 = cv2.resize(cartoonImage, (960, 540))

    
images=[ReSized1, ReSized2, ReSized4, gaussian, markers ,ReSized6]# figsize=(6,9),
tle=["Real Image","GrayScale Image","ThreshHold Image","GaussianBlur","Water Shed","Cartoonyfy Image"]
fig, axes = plt.subplots(3,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.set_title(tle[i])
    ax.imshow(images[i], cmap='gray')
        #        save button code

plt.show()