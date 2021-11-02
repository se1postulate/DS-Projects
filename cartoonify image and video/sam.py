

import numpy as np
import cv2
from matplotlib import pyplot as plt

from sklearn.cluster import MeanShift, estimate_bandwidth
import easygui
ImagePath=easygui.fileopenbox()

img = cv2.imread(ImagePath)
imr=img
imr=cv2.cvtColor(imr, cv2.COLOR_BGR2RGB)
gaussian=cv2.GaussianBlur(img,(7,7),0 )
gaussian=cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)

cv2.imwrite('gaussian.jpg',cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
resize5=cv2.resize(gaussian, (960, 540))
img = cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
resize2 = cv2.resize(gray, (960, 540))

ret, thresh = cv2.threshold(gray,3,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((2,2),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 1)

# sure background area
sure_bg = cv2.dilate(closing,kernel,iterations=1)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)

# Threshold
ret, sure_fg = cv2.threshold(dist_transform,0.05*dist_transform.max(),100,0)

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


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

resize3=cv2.resize(img, (960, 540))


originalmage=gaussian
ReSized1 = cv2.resize(imr, (960, 540))

getEdge = cv2.adaptiveThreshold(gray, 255, 
cv2.ADAPTIVE_THRESH_MEAN_C, 
cv2.THRESH_BINARY, 9, 9)

ReSized4 = cv2.resize(getEdge, (960, 540))

cv2.imwrite('threshold.jpg',getEdge)
cartoonImage = cv2.bitwise_and(gaussian,img, mask=getEdge)

ReSized6 = cv2.resize(cartoonImage, (960, 540))

    
images=[ReSized1,resize2, ReSized4,resize5, resize3 ,ReSized6]# figsize=(6,9),
tle=["Real Image","GrayScale Image","ThreshHold Image","GaussianBlur","Water Shed","Cartoonyfy Image"]
fig, axes = plt.subplots(3,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.set_title(tle[i])
    ax.imshow(images[i], cmap='gray')
    
plt.show()
cartoonImage=cv2.cvtColor(cartoonImage, cv2.COLOR_BGR2RGB)
cv2.imwrite('cr.jpg',cartoonImage)