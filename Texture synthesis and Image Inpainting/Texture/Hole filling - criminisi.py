import numpy as np
from numpy import pi, exp, sqrt
import cv2
from PIL import Image as image
import matplotlib.pyplot as plt
from time import time
import os

global WindowSize
WindowSize=35#
global figno
#figno=10
global OmegaMask
global Confidence


def get_window(pixel,img,wsize=WindowSize):
    HalfW=int(wsize/2)
    x1=max([pixel[0]-HalfW,0])
    x2=min([pixel[0]+HalfW+1,img.shape[0]])
    y1=max([pixel[1]-HalfW,0])
    y2=min([pixel[1]+HalfW+1,img.shape[1]])
    return img[x1:x2,y1:y2]

def FindMatches(Template,SampleImage,ValidMask,pixel):
    SSD=np.zeros(SampleImage.shape)
    HalfW=int(WindowSize/2)
    indices=np.transpose(np.nonzero(1-OmegaMask))
    th=50
    while True:
        print(th)
        indices=[tuple(index) for index in indices if np.sum(get_window(index,OmegaMask))==0 and abs(index[0]-pixel[0])<th and abs(index[1]-pixel[1])<th]
        minimum=99999999
        for i1,i2 in indices:
            try:
                index=tuple([i1,i2])
                Sample=get_window(index,SampleImage)
                SampleMask=get_window(index,1-OmegaMask)
                if np.sum(ValidMask*SampleMask)!=np.sum(ValidMask):
                    continue
                dist=np.sum(((Template-Sample)**2)*ValidMask*SampleMask)
                SSD[index]=dist
                if dist<minimum:
                    minimum=dist
                    final=index
            except: continue
        try:
            #minSSD=np.unravel_index(np.argmin(SSD[HalfW:SampleImage.shape[0]-HalfW,HalfW:SampleImage.shape[1]-HalfW]), SSD.shape)
            #print(minimum)
            return final
        except:
            th+=50
    
def grow_image(SampleImage,Image):
    global Confidence
    global OmegaMask
    HalfW=int(WindowSize/2)
    while np.sum(OmegaMask):
    #for _ in range(50):
        #try:
            pixel=GetUnfilledNeighbors()
            print(pixel)
            pixel=tuple(pixel)
            Template=get_window(pixel,Image)
            ValidMask=get_window(pixel,1-OmegaMask)
            BestMatch=FindMatches(Template,SampleImage,ValidMask,pixel)
            print(BestMatch, Image[pixel],Image[BestMatch], OmegaMask[BestMatch])
            Image[pixel[0]-HalfW:pixel[0]+HalfW+1,pixel[1]-HalfW:pixel[1]+HalfW+1]=get_window(BestMatch,SampleImage)*get_window(pixel,OmegaMask)+get_window(pixel,1-OmegaMask)*Image[pixel[0]-HalfW:pixel[0]+HalfW+1,pixel[1]-HalfW:pixel[1]+HalfW+1]
            """
            plt.figure(11)
            plt.imshow(Image,"gray")
            plt.figure(12)
            plt.imshow(OmegaMask,"gray")
            plt.figure(13)
            plt.imshow(Image,"gray")
            """
            OldOmegaMask=OmegaMask.copy()
            OmegaMask[pixel[0]-HalfW:pixel[0]+HalfW+1,pixel[1]-HalfW:pixel[1]+HalfW+1]=0
            Confidence=Confidence+(OldOmegaMask-OmegaMask)*Confidence[pixel]
            print(np.sum(OmegaMask))
        #except: 
        #    return Image
    return Image


def GetUnfilledNeighbors():
    global Confidence
    alpha=255
    Sobel_x=[[-1,0,1],[-2,0,2],[-1,0,1]]
    Sobel_y=[[-1,2,-1],[0,0,0],[1,2,1]]    
    kernel=np.ones((3,3),dtype=np.uint8)
    dilated=cv2.dilate(1-OmegaMask,kernel)
    diff=dilated-(1-OmegaMask)
    boundary=np.transpose(np.nonzero(diff))
    maximum=0
    for p in boundary:
        im_window=get_window(p,Image)
        C=np.sum(get_window(p,Confidence))
        mag=im_window.shape[0]*im_window.shape[1]
        C_term=C/mag
        Confidence[tuple(p)]=C_term
        gradient_x=np.sum(get_window(p,Image,3)*Sobel_x)
        gradient_y=np.sum(get_window(p,Image,3)*Sobel_y)
        gradient=[gradient_x,gradient_y]
        normal_x=np.sum(get_window(p,diff,3)*Sobel_x)
        normal_y=np.sum(get_window(p,diff,3)*Sobel_y)
        normal=[normal_x,normal_y]
        unit_normal=normal/np.linalg.norm(normal)
        D_term=abs(np.dot(unit_normal,gradient))/alpha
        P=D_term*C_term
        #print(P,maximum,D_term,C_term, unit_normal,gradient)
        if P>=maximum:
            maximum=P
            final=p
    return final
    
fname="test_im3-floor3.jpg"
Image=cv2.imread(fname)

OmegaMask=np.zeros((Image.shape[0],Image.shape[1]),dtype=np.uint8)
for x in range(Image.shape[0]):
    for y in range(Image.shape[1]):
        if Image[x,y,0]<50 and Image[x,y,1]<50 and Image[x,y,2]>200:
            OmegaMask[x,y]=1
kernel=np.ones((5,5),dtype=np.uint8)
OmegaMask=cv2.morphologyEx(OmegaMask,cv2.MORPH_CLOSE,kernel)
Image=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
SampleImage=Image.copy()

Confidence=np.ones(Image.shape,dtype=np.uint8)
Confidence=Confidence - OmegaMask*1
#plt.imshow(Image,"gray")
Icopy=Image.copy()
Image=grow_image(SampleImage,Image)
print(np.sum(Image-Icopy))
plt.imshow(Image,"gray")
cv2.imwrite(fname[:-4]+"-out.jpg",Image)
        

    


    
