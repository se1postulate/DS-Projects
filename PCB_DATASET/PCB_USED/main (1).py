# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:40:05 2021

@author: postulate-31
"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
import easygui
ImagePath=easygui.fileopenbox("D:/PCB/PCB dataset/PCB_DATASET/PCB_USED/01.JPG",0)
img_m=cv2.imread(ImagePath)

hsv_img = rgb2hsv(img_m)
hue=hsv_img[:,:,0]
sat=hsv_img[:,:,1]
val=hsv_img[:,:,2]
# plt.imshow(hsv_img[:,:,0],cmap='gray')
# plt.imshow(hsv_img[:,:,1],cmap='gray')
# plt.imshow(hsv_img[:,:,2],cmap='gray')
cv2.imshow('saturation',sat)
cv2.imshow('value',val)

s_gau=cv2.GaussianBlur(sat,(1,1),0 )
cv2.imshow('saturation gaussian Blur',s_gau)

v_gau=cv2.GaussianBlur(val,(1,1),0 )
cv2.imshow('Value gaussian Blur',v_gau)

filterSize =(3, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                   filterSize)
s_morph_black= cv2.morphologyEx(s_gau, 
                              cv2.MORPH_BLACKHAT,
                              kernel)

v_morph_black= cv2.morphologyEx(v_gau, 
                              cv2.MORPH_BLACKHAT,
                              kernel)
cv2.imshow('saturation Black hat morphological operation',s_morph_black)

cv2.imshow('Value Black hat morphological operation',v_morph_black)

# kernel = np.ones((5,5), np.uint8)

s_img_erosion = cv2.erode(s_gau, kernel, iterations=1)
s_img_dilation = cv2.dilate(s_gau, kernel, iterations=1)

cv2.imshow('saturation Erosion morphological operation',s_img_erosion)

cv2.imshow('saturation dilation morphological operation',s_img_dilation)


v_img_erosion = cv2.erode(v_gau, kernel, iterations=1)
v_img_dilation = cv2.dilate(v_gau, kernel, iterations=1)

cv2.imshow('value Erosion morphological operation',v_img_erosion)

cv2.imshow('value dilation morphological operation',v_img_dilation)

s_opening = cv2.morphologyEx(s_gau, cv2.MORPH_OPEN, kernel)
cv2.imshow('saturation OPEN morphological operation',s_opening)

v_opening = cv2.morphologyEx(v_gau, cv2.MORPH_OPEN, kernel)
cv2.imshow('value OPEN morphological operation',v_opening)

#cv2.destroyAllWindows()
