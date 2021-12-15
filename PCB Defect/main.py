# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:32:42 2021

@author: Roshini
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import easygui
from skimage.feature import match_template

import os, numpy, PIL
from PIL import Image
from tkinter import filedialog as fd

from skimage.measure import compare_ssim
#from skimage.metrics import structural_similarity as compare_ssim

#pip install scikit-learn==0.16.1
#import argparse
import imutils

from tkinter.messagebox import showinfo
import tkinter as tk
from tkinter import *
import tensorflow as tf

root = Tk()
root.withdraw()


model = tf.keras.models.load_model('best_model.h5')


'''
Match template

The match_template function uses fast, normalized cross-correlation 1 to find instances of the template in the image. Note that the peaks in the output of match_template correspond to the origin (i.e. top-left corner) of the template.

'''

color=[(0,100,255),(255,255,0),(255,0,255),(0,0,255),(205,100,0),(0,255,0)]

color_text=[(0,100,255),(255,255,0),(255,0,255),(0,0,255),(205,100,0),(0,255,0)]
def temp_match(img_rgb):    
#    img_rgb = cv2.imread(img_rgb,0)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # img=cv2.resize(img_rgb,(25,25))
    img=np.resize(img,(30,30))
    # print(img)
    # print(img.shape)
    img=img.reshape(1,30,30,1)
    cl=['Open',
     'Short',
     'Mousebite',
     'Spur',
     'Copper',
     'Pin-hole']
    p1=model.predict(img)
    
    p2=list(p1[0])
    p3=max(p2)
    p4=p2.index(p3)
    
    p5=cl[p4]
    return p5,p4

#ij = np.unravel_index(np.argmax(result), result.shape)
#x, y = ij[::-1]
    
showinfo(
        title='alert',
        message=f"ready to selct original image"
    )
root.update()
original_image_path=fd.askopenfilename()
imageA = cv2.imread(original_image_path)
s=imageA.shape
imageA=cv2.resize(imageA,(500,500))
showinfo(
        title='alert',
        message=f"ready to selct Second image image"
    )

root.update()
edited=fd.askopenfilename()

root.quit()
root.destroy()

imageB = cv2.imread(edited)
imageB=cv2.resize(imageB,(500,500))
# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")

# print("SSIM: {}".format(score))


# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 150, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)



# loop over the contours
i=0
imgc=imageB.copy()
res_all=[]
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(c)
    if w>10 and h>3 and h<70 and w<70:
        print("Defect Number:",i+1)
        print("depth:",w)
        print("Height:",h)
        print("X:",x)
        print("Y:",y)
        print("Area Of Defect:",w*h)
        #cv2.rectangle(imageB, (x, y,z), (x + w, y + h), (0, 0, 255), 2)
        cropped_image = imageB[y:y+h, x:x+w]
    
        res,index=temp_match(cropped_image)
        print("Defect:",res)
        res_all.append(res)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(imgc, (x, y), (x + w, y + h), color[index], 2)
        cv2.putText(imgc, f'{i+1}.{res}', (x,y), font, 0.5,color_text[index] , 1, cv2.LINE_AA)
        
        print("\n")
    
        i+=1

cv2.imwrite(f"plot/result.jpg",imgc)
# show the output images
#print("Total Defect:",res_all)

print("Total Number Of Defect:",len(res_all))
print("\n")
print("list of Defects:")
k=1
for i in res_all:
    print(f"{k}.{i}")
    k=k+1
    

cv2.imshow("original", imageA)
cv2.imshow("Defected Location", imgc)
#imageB=cv2.resize(imageB,(500,500))
cv2.imshow("modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)#press q for close all window
cv2.destroyAllWindows()

 