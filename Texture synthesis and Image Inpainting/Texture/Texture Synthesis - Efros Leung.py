import numpy as np
from numpy import pi, exp, sqrt
import cv2
from PIL import Image as image
import matplotlib.pyplot as plt
from time import time
import os

global filled
global WindowSize
global GaussMask
WindowSize=11
#global figno
#figno=10
def GaussKernel():
    s = WindowSize/6.4
    k = int(WindowSize/2)
    probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)] 
    kernel = np.outer(probs, probs)
    return kernel

GaussMask=GaussKernel()
def get_window(pixel,img):
    
    HalfW=int(WindowSize/2)

    x1=max([pixel[0]-HalfW,0])
    x2=min([pixel[0]+HalfW+1,img.shape[0]])
    y1=max([pixel[1]-HalfW,0])
    y2=min([pixel[1]+HalfW+1,img.shape[1]])
    return img[x1:x2,y1:y2]

def FindMatches(Template,SampleImage,ValidMask,pixel):
    thresh=0.1
    SSD=np.zeros(SampleImage.shape)
    HalfW=int(WindowSize/2)
    I1=list(range(max([HalfW,pixel[0]-50]),min([SampleImage.shape[0]-HalfW,pixel[0]+50])))
    I2=list(range(max([HalfW,pixel[1]-50]),min([SampleImage.shape[1]-HalfW,pixel[1]+50])))
    indices=[(i1,i2) for i1 in I1 for i2 in I2 if filled[i1,i2]]
    TotWeight=np.sum(GaussMask*ValidMask)
    for index in indices:
        Sample=get_window(index,SampleImage)
        dist=np.sum(((Template-Sample)**2)*ValidMask*GaussMask)
        SSD[index]=dist
    SSD=SSD/TotWeight
    #minSSD=np.min(SSD[HalfW:SampleImage.shape[0]-HalfW,HalfW:SampleImage.shape[1]-HalfW])
    minSSD=SSD[tuple(min(indices,key=lambda x:SSD[x]))]
    PixelList=list(filter(lambda x:SSD[x]<=minSSD*(1+thresh),indices))
    #PixelList=[x for x in indices if SSD[tuple(x)]<=minSSD*(1+thresh)]
    errors=[SSD[tuple(index)] for index in PixelList]
    return PixelList,errors
    
def grow_image(SampleImage,Image):
    global filled
    MaxThresh=0.3
    HalfW=int(WindowSize/2)
    print(len(np.transpose(np.nonzero(1-filled))))
    while len(np.transpose(np.nonzero(filled)))<(Image.shape[0]*Image.shape[1]):
        progress=0
        PixelList=GetUnfilledNeighbors(filled)
        for pixel in PixelList:
            pixel=tuple(pixel)
            if HalfW>pixel[0] or pixel[0]>=Image.shape[0]-HalfW or HalfW>pixel[1] or pixel[1]>=Image.shape[1]-HalfW:
                filled[pixel]=1
                progress=1
                continue
            Template=get_window(pixel,Image)
            ValidMask=get_window(pixel,filled)
            BestMatches,errors=FindMatches(Template,SampleImage,ValidMask,pixel)
            #print(BestMatches)
            BestMatchIndex=np.random.randint(len(BestMatches))
            BestMatch=BestMatches[BestMatchIndex]
            if errors[BestMatchIndex]<MaxThresh:
                Image[tuple(pixel)]=SampleImage[tuple(BestMatch)]     
                filled[pixel]=1
                progress=1
        if not(progress):
            MaxThresh*=1.1
        print(len(np.transpose(np.nonzero(1-filled))))
    return Image

def GetUnfilledNeighbors(filled):
    kernel = np.ones((WindowSize,WindowSize))
    dilated=cv2.dilate(filled,kernel)
    diff=dilated-filled
    PixelList=np.transpose(np.nonzero(diff))
    PixelList=sorted(list(PixelList),key=lambda x:np.sum(get_window(x,filled)),reverse=True)
    return PixelList



global figno
figno=20
print("Window",WindowSize)
dim1,dim2=200,200
"""
# read file
#fname="T1.gif"
for fname in ["T4.gif","T5.gif"]:
    SampleImage=np.array(image.open(fname))
    SampleImage=np.array(SampleImage/255,dtype=np.float)
    d1h,d2h=int(SampleImage.shape[0]/2),int(SampleImage.shape[1]/2)
    Image=np.zeros((dim1,dim2))
    global filled
    filled=np.zeros(Image.shape,dtype=np.uint8)
    Image[d1h:d1h+WindowSize,d2h:d2h+WindowSize]=SampleImage[d1h:d1h+WindowSize,d2h:d2h+WindowSize]
    filled[d1h:d1h+WindowSize,d2h:d2h+WindowSize]=1
    t_start=time()
    Image=grow_image(SampleImage,Image,WindowSize)
    Image=np.array(Image*255,dtype=np.uint8)
    print("Time "+fname,time()-t_start)
    cv2.imwrite(fname[0:2]+"-"+str(WindowSize)+"-out.jpg",Image)
"""
"""
fname="test_im1_cropped.jpg"
SampleImage=cv2.imread(fname,0)
plt.imshow(SampleImage,"gray")
SampleImage=np.array(SampleImage/255,dtype=np.float)

offset=20

Image=np.zeros((SampleImage.shape[0]+offset*2,SampleImage.shape[1]+offset*2))
global filled
filled=np.zeros(Image.shape,dtype=np.uint8)
Image[offset:offset+SampleImage.shape[0],offset:offset+SampleImage.shape[1]]=SampleImage


filled[offset:offset+SampleImage.shape[0],offset:offset+SampleImage.shape[1]]=1
t_start=time()
Image=grow_image(SampleImage,Image)
Image=np.array(Image*255,dtype=np.uint8)

print("Time "+fname,time()-t_start)
cv2.imwrite(fname[0:2]+"-"+str(WindowSize)+"-out.jpg",Image)
"""

fname="test_im2.bmp"
SampleImage=np.array(image.open(fname))
SampleImage=np.array(SampleImage/255,dtype=np.float)
Image=SampleImage.copy()
SampleImage=np.concatenate((SampleImage[:140],SampleImage[200:]),axis=0)
offset=25

global filled
filled=np.zeros(Image.shape,dtype=np.uint8)


filled[:148,:]=1
filled[190:,:]=1
t_start=time()
plt.imshow(Image,"gray")

Image=grow_image(SampleImage,Image)
Image=np.array(Image*255,dtype=np.uint8)
plt.figure(1)
plt.imshow(Image,"gray")
print("Time "+fname,time()-t_start)
cv2.imwrite("test2-"+str(WindowSize)+"-out.jpg",Image)
