# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:15:15 2021

@author: AMK
"""

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
def meanshift(ImagePath):
    originImg=cv2.imread(ImagePath)
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
    print("number of estimated clusters : %d" % n_clusters_)    
    
    # Displaying segmented image    
#    segmentedImg = np.reshape(labels, originShape[:2])    
    segmentedImg2 = cluster_centers[np.reshape(labels, originShape[:2])]
    
    cv2.imwrite('meanshift.jpg',segmentedImg2)



def upload():
    ImagePath=easygui.fileopenbox()
    cartoonify(ImagePath)
#upload()
def cartoonify(ImagePath):
    #read the image
    originalmage = cv2.imread(ImagePath)
    originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)

    if originalmage is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()
    ReSized1 = cv2.resize(originalmage, (960, 540))
    #plt.imshow(ReSized1, cmap='gray')
    grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
    ReSized2 = cv2.resize(grayScaleImage, (960, 540))
    imr=originalmage
    imr=cv2.cvtColor(imr, cv2.COLOR_BGR2RGB)
    gaussian=cv2.GaussianBlur(imr,(7,7),0 )
    gaussian=cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
    ReSized3 = cv2.resize(gaussian, (960, 540))
    
    #Meanshift here
    meanshift(ImagePath)
    means=cv2.imread('meanshift.jpg')
    means = cv2.cvtColor(means, cv2.COLOR_BGR2RGB)

    Meanshift=means
    grayScaleImage = cv2.cvtColor(Meanshift, cv2.COLOR_BGR2GRAY)
    
    getEdge = cv2.adaptiveThreshold(grayScaleImage, 70, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 11)

    ReSized4 = cv2.resize(getEdge, (960, 540))
    
    
    global cartoonImage
    cartoonImage = cv2.bitwise_and(Meanshift, Meanshift, mask=getEdge)
    ReSized6 = cv2.resize(cartoonImage, (960, 540))
    #plt.imshow(ReSized6, cmap='gray')
    # Plotting the whole transition
    images=[ReSized1, ReSized2,  ReSized4, ReSized3,Meanshift, ReSized6]
    tle=["Real Image","GrayScale Image","ThreshHold Image","GaussianBlur",'MeanShift Cluster',"Cartoonyfy Image"]

    fig, axes = plt.subplots(3,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.set_title(tle[i])
        ax.imshow(images[i], cmap='gray')

    plt.show()
    cv2.imwrite('cartoonified_Image.jpg',cv2.cvtColor(cartoonImage, cv2.COLOR_BGR2RGB))
#
#    
#def save(cartoonImage, ImagePath):
#    #saving an image using imwrite()
#    newName="cartoonified_Image"
#    path1 = os.path.dirname(ImagePath)
#    extension=os.path.splitext(ImagePath)[1]
#    path = os.path.join(path1, newName+extension)
#    cv2.imwrite(path, cv2.cvtColor(cartoonImage, cv2.COLOR_BGR2RGB))
#    I = "Image saved by name " + newName +" at "+ path
#    tk.messagebox.showinfo(title=None, message=I)
##    cv2.imwrite('cartoonified_Image.jpg',)
#
#    
 
top=tk.Tk()
top.geometry('400x400')
top.title('Cartoonify Your Image !')
top.configure(background='white')
label=Label(top,background='#CDCDCD', font=('calibri',20,'bold'))
upload=Button(top,text="Cartoonify an Image",command=upload,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('calibri',10,'bold'))
upload.pack(side=TOP,pady=50)
#save1=Button(top,text="Save cartoon image",command=lambda: save(cartoonImage,ImagePath),padx=30,pady=5)
#save1.configure(background='#364156', foreground='white',font=('calibri',10,'bold'))
#save1.pack(side=TOP,pady=50)
top.mainloop()