# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:45:27 2021

@author: postulate-31
"""

import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth

import cv2 #for image processing
import easygui #to open the filebox
import numpy as np #to store image
import imageio #to read image stored at particular path
import sys
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import Label,Button,TOP
from PIL import ImageTk, Image

import matplotlib.pyplot as plt

vidObj = cv2.VideoCapture('m.mp4')
video_count = 0

out = cv2.VideoWriter('outputvideo/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (500,500))
  
    # checks whether frames were extracted
success = 1
  
while success:
    success, image = vidObj.read()
    sha=image.shape
    image=cv2.resize(image,(500,500))
    img_main=image
    
    
    
    #MEAN SHIFT
    originImg=image
    # Shape of original image    
    originShape = originImg.shape
    
      
    flatImg=np.reshape(originImg, [-1, 3])
    
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)    
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
    
    # Performing meanshift on flatImg    
    ms.fit(flatImg)
    
    # (r,g,b) vectors corresponding to the different clusters after meanshift    
    labels=ms.labels_
    
    # Remaining colors after meanshift    
    cluster_centers = ms.cluster_centers_    
    
    # Finding and diplaying the number of clusters    
    labels_unique = np.unique(labels)    
    n_clusters_ = len(labels_unique)    
    #print("number of estimated clusters : %d" % n_clusters_) 
    
    segmentedImg2 = cluster_centers[np.reshape(labels, originShape[:2])]
    
    cv2.imwrite('meanshift.jpg',segmentedImg2)
    
    '''' main '''
    originalmage = img_main
    originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)

    if originalmage is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()
   # ReSized1 = cv2.resize(originalmage, (960, 540))
    #plt.imshow(ReSized1, cmap='gray')
    grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
    #ReSized2 = cv2.resize(grayScaleImage, (960, 540))
    imr=originalmage
    imr=cv2.cvtColor(imr, cv2.COLOR_BGR2RGB)
    gaussian=cv2.GaussianBlur(imr,(7,7),0 )
    gaussian=cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
    #ReSized3 = cv2.resize(gaussian, (960, 540))
    
    #Meanshift here
    means=cv2.imread('meanshift.jpg')
    means = cv2.cvtColor(means, cv2.COLOR_BGR2RGB)

    Meanshift=means
    grayScaleImage = cv2.cvtColor(Meanshift, cv2.COLOR_BGR2GRAY)
    
    getEdge = cv2.adaptiveThreshold(grayScaleImage, 70, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 11)

    #ReSized4 = cv2.resize(getEdge, (960, 540))
    
    
    global cartoonImage
    cartoonImage = cv2.bitwise_and(Meanshift, Meanshift, mask=getEdge)
    #ReSized6 = cv2.resize(cartoonImage, (960, 540))
    #plt.imshow(ReSized6, cmap='gray')
    # Plotting the whole transition
    # images=[ReSized1, ReSized2,  ReSized4, ReSized3,Meanshift, ReSized6]
    # tle=["Real Image","GrayScale Image","ThreshHold Image","GaussianBlur",'MeanShift Cluster',"Cartoonyfy Image"]

    # fig, axes = plt.subplots(3,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    # for i, ax in enumerate(axes.flat):
    #     ax.set_title(tle[i])
    #     ax.imshow(images[i], cmap='gray')

    # plt.show()
    #cv2.imwrite('cartoonified_Image.jpg',cv2.cvtColor(cartoonImage, cv2.COLOR_BGR2RGB))


    #cv2.imwrite(f"cv/frame{video_count}.jpg", segmentedImg2)
    # rim=cv2.resize(src, sha)
    ct=cv2.cvtColor(cartoonImage, cv2.COLOR_BGR2RGB)
    out.write(ct)
    cv2.imwrite(f"cv/frame{video_count}.jpg",ct)

    video_count+=1

vidObj.release()
out.release()
cv2.destroyAllWindows()


