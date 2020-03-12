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
global INDICES
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
    x1,x2,y1,y2=INDICES[pixel[0]][pixel[1]]
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
    MaxThresh=0.6
    HalfW=int(WindowSize/2)
    print(len(np.transpose(np.nonzero(1-filled))))
    #for _ in range(2):
    while len(np.transpose(np.nonzero(filled)))<(Image.shape[0]*Image.shape[1]):
        if MaxThresh>0.66:
            MaxThresh=1
        progress=0
        PixelList=GetUnfilledNeighbors()
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

def GetUnfilledNeighbors():
    kernel = np.ones((WindowSize,WindowSize))
    dilated=cv2.dilate(filled,kernel)
    diff=dilated-filled
    PixelList=np.transpose(np.nonzero(diff))
    PixelList=sorted(list(PixelList),key=lambda x:np.sum(get_window(x,filled)),reverse=True)
    return PixelList



global figno
figno=20
print("Window",WindowSize)
#dim1,dim2=200,200

fname="test_im3-sign.jpg"
Image=cv2.imread(fname)

OmegaMask=np.zeros((Image.shape[0],Image.shape[1]),dtype=np.uint8)
for x in range(Image.shape[0]):
    for y in range(Image.shape[1]):
        if Image[x,y,0]<50 and Image[x,y,1]<50 and Image[x,y,2]>200:
            OmegaMask[x,y]=1
kernel=np.ones((5,5),dtype=np.uint8)
OmegaMask=cv2.morphologyEx(OmegaMask,cv2.MORPH_CLOSE,kernel)
Image=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
filled=1-OmegaMask
SampleImage=Image.copy()
HalfW=int(WindowSize/2)


INDICES=[[[] for j in range(SampleImage.shape[1])] for i in range(SampleImage.shape[0])]
for i0 in range(SampleImage.shape[0]):
    for i1 in range(SampleImage.shape[1]):
        x1=max([i0-HalfW,0])
        x2=min([i0+HalfW+1,SampleImage.shape[0]])
        y1=max([i1-HalfW,0])
        y2=min([i1+HalfW+1,SampleImage.shape[1]])
        INDICES[i0][i1]=[x1,x2,y1,y2]

#Confidence=np.ones(Image.shape,dtype=np.uint8)
#Confidence=Confidence - OmegaMask*1
#plt.imshow(Image,"gray")
#Icopy=Image.copy()
Image=grow_image(SampleImage,Image)

plt.imshow(Image,"gray")
print(np.sum(Image-Image.copy()))
cv2.imwrite(fname[:-5]+"-out.jpg",Image)
