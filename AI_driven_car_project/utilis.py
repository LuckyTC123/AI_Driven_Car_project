import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.image as npimg
from imgaug import augmenters as iaa
import cv2
import random

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D,Flatten,Dense
from keras.optimizers import Adam


def getName(filpath):
    return filpath.split('\\')[-1]

def importDataInfo(path):
    coloums =['center','left','right','steering','throttle','brake','speed']
    data =pd.read_csv(os.path.join(path,'driving_log.csv'),names = coloums)
    data['center'] =data['center'].apply(getName)
    return data
def balaceData(data,display=True):
    nbins=31
    sampleperbin=500
    hist,bins = np.histogram(data['steering'],nbins)
    if display:
        
        center=(bins[:-1] + bins[1:]) * 0.5
        plt.bar(center,hist,width =0.06)
        plt.plot((-1,1),(sampleperbin,sampleperbin))
        plt.show()
    removeIndexList = []
    for j in range (nbins):
        bidatalist = []
        for i in range(len(data['steering'])):
            if data['steering'][i] >=bins[j] and data['steering'][i] <=bins[j+1]:
                bidatalist.append(i)
        bidatalist = shuffle(bidatalist)  
        bidatalist = bidatalist[sampleperbin:]
        removeIndexList.extend(bidatalist)  
    print(len(removeIndexList))
    data.drop(data.index[removeIndexList],inplace = True)
    print('remaining images',len(data))
    if display:
        hist, _ = np.histogram(data['steering'],nbins)
    return data
        
def loaddata(path,data):
    imagespath = []
    steering = []
    for i in range(len(data)):
        indexdata = data.iloc[i]
        #print(indexdata)
 
        imagespath.append(os.path.join(path,'IMG',indexdata[0]))
       
        steering.append(float(indexdata[3]))
    imagespath =np.asarray(imagespath)
    steering=np.asarray(steering)
    return imagespath, steering

def augmentimg(imagespath,steering):
    img = npimg.imread(imagespath)
    #pan
    if np.random.rand() < 0.5:
        pan =iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img =pan.augment_image(img)
    #zoo
    if np.random.rand() < 0.5:
        zoom =iaa.Affine(scale=(1,1.2))
        img =zoom.augment_image(img)
    #brightness
    if np.random.rand() < 0.5:
        brt =iaa.Multiply(0.3,1.0)
        img =brt.augment_image(img)
    #flip
    if np.random.rand() < 0.5:
        img =cv2.flip(img,1)
        steering = -steering
    
    return img, steering
def prprocess(img):
    img =img[60:135,:,:]
    img =cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img =cv2.GaussianBlur(img,(3,3),0)
    img =cv2.resize(img,(200,66))
    img =img/255
    return img
def batchgen(imagespath,steerings,batchsize,trainflag):
    while True:
        imgBatch = []
        steerigbatch =[]
        
        for i in range(batchsize):
            index = random.randint(0,len(imagespath)-1)
            if trainflag:
                img, steering =augmentimg(imagespath[index],steerings[index])
            else:
                img =npimg.imread(imagespath[index])
                steering =steerings[index]
            img = prprocess(img)
            imgBatch.append(img)
            steerigbatch.append(steerings)
        yield(np.asarray(imgBatch),np.asarray(steerigbatch))
        
def creatmodel():
    model = Sequential()
    model.add(Convolution2D(24,(5,5),(2,2),input_shape= (66,200,3),activation='elu'))
    model.add(Convolution2D(36,(5,5),(2,2),activation='elu'))   
    model.add(Convolution2D(48,(5,5),(2,2),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    model.add(Convolution2D(64,(3,3),activation='elu'))
    
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))
    
    model.compile(Adam(learning_rate=0.0001),loss='mse')
    return model