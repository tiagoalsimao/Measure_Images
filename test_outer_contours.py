from install import install
from _ast import If
from cv2 import drawContours
from scipy.ndimage.filters import uniform_filter1d
import scipy
install("numpy")
install("cv2")
install("PIL")
install("matplotlib")

from scipy.signal import argrelextrema,argrelmax
from skimage.draw import line
from shapely.geometry import LineString,MultiLineString,Point,MultiPoint
from numpy import *
import numpy as np
import cv2 as cv
import cv2
from matplotlib.pyplot import *
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os

# get equidistante points (check if it is possible to use pchip to smooths changes)
def getEquidistantPoints(outerLine):
    x,y = outerLine.T
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])
    
    dist_array = (x[:-1]-x[1:])**2 + (y[:-1]-y[1:])**2

    lineLength = sum(sqrt(sum(diff(outerLine,axis=0)**2,axis=1)))

    # create points with distances of 10 pixels
    t = np.linspace(0,u.max(),(lineLength/10).astype(int))
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    
    return xn, yn

def mDrawContours(image,contours):
    imgDraw = image.copy()
    if len(imgDraw.shape) == 2:
        imgDraw = imgDraw*255/np.max(imgDraw)
        imgDraw = np.stack((imgDraw,)*3, axis=-1)
    
    cv2.drawContours(imgDraw, contours, -1, (0,255,0), 3)
    imshow(imgDraw)

def intersections2coords(intersectionMultiPoint):
    
    # There is no intersection
    if type(intersectionMultiPoint) is LineString:
        return [], []
    
    # check intersection is a single point
    if type(intersectionMultiPoint) is Point:
        return intersectionMultiPoint.coords.xy
    
    # return multi points of intersection
    nPoints = len(intersectionMultiPoint)
    xi = np.zeros(nPoints)
    yi = np.zeros(nPoints)
    for i in range(nPoints):
        xi[i] = intersectionMultiPoint[i].x
        yi[i] = intersectionMultiPoint[i].y
    
    return xi,yi

# return indexes it array for first -1 and the first 1 after it
def getStartEndIndexes(arrayDiff):
    indexStart = np.where(arrayDiff==-1)[0]
    indexEnd = np.where(arrayDiff==1)[0]
    
    for indS in indexStart:
        for indE in indexEnd:
            if indS < indE:
                return indS, indE
            
    return -1, -1

def getThreshImage(image,tbin,ko,kc,kd,mode='binary',blockSize=21):
    
    # adaptiveThreshold better for images with brightness differences
    
    if mode=='binary':
        _,imgMask = cv2.threshold(image,20,255,cv2.THRESH_BINARY)
        
    elif mode=='adaptative':
        imgMask = cv2.adaptiveThreshold(255-image,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize,2)
    else:
        print('mode \''+ mode +'\' not recugnized.', file=sys.stderr)
        sys.exit(1)
    
    # remove small noise
    if ko > 0:
        kernelOpening = np.ones((ko, ko), np.uint8)
        imgMask = cv2.morphologyEx(imgMask, cv2.MORPH_OPEN, kernelOpening)
    
    # Merge blocks
    if kc > 0:
        kernelClose = np.ones((kc, kc), np.uint8)    
        imgMask = cv2.morphologyEx(imgMask, cv2.MORPH_CLOSE, kernelClose)
        
    # Dilate
    if kd > 0:
        kernelDilate = np.ones((kd, kd), np.uint8)    
        imgMask = cv2.morphologyEx(imgMask, cv2.MORPH_DILATE, kernelDilate)
        
    return imgMask

def getPerpendicularPointsThresh (imgMask,bresenhamLine):
    # get line pixels from image
    imageValues = imgMask[bresenhamLine[:,1],bresenhamLine[:,0]]
    
    iS,iE = getStartEndIndexes(diff(imageValues))
    
    if iS == -1 or iE == -1 :
        return None, None
    
    pointS = bresenhamLine[iE]
    pointE = bresenhamLine[iS]
    
    return pointS, pointE

plt.switch_backend('QT5Agg')
# To maximaze window
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.ion()
# imageName = "9024KO_db11_16h_1.czi.tiff"
# imageName = "9024KO_db11_3h_4.czi.tiff"
# imageName = "9024KO_db11_3h_7.czi.tiff"
# imageName = "9024KO_db11_16h_7.czi.tiff"
# imageName = "w1118_db11_16h_3.czi.tiff"
# imageName = "w1118_sucrose_16h_9.czi.tiff"
# imageName = "w1118_db11_16h_7.czi.tiff"
# imageName = "9024KO_db11_3h_8.czi.tiff"
# imageName = "w1118_db11_16h_1.czi.tiff"
# imageName = "9024KO_db11_16h_8.czi.tiff"
# imageName = "9024KO_db11_16h_3.czi.tiff"
imageName = "w1118_db11_16h_1.czi.tiff"

# Input variables
srcFolder = './data/20201125_CG1139KO/tif/'
fileNameSpec = "*.tiff"

# Destination folder to save images as tif
plotFolder = './data/20201125_CG1139KO/plot/'

# List of Images
imageList = glob.glob(srcFolder + fileNameSpec)

# List length
nList = len(imageList)

distanceList=[]

# Loop through each image
for i in range(0,nList):
# for i in range(1):
    
    clf()
    
    # Image name
    imageName = os.path.basename(imageList[i])
    
    print("Processing " + str(i) + " of " + str(nList) + ": " + imageName)
    
    img = cv.imread(srcFolder + imageName,0)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Filter image to eliminate high frequency noise
    kg = 11
    imgray = cv2.GaussianBlur(imgray, (kg, kg), 0)
    
    # get image Mask
    tbin = 20
    ko = 2
    kc = 31
    kd = 10
    imgMask = getThreshImage(imgray,tbin,ko,kc,kd)
    
    
    imgAdaptMask = getThreshImage(imgray,tbin,ko,kc,kd)
    
    # Image to int array
    imgAsInt = imgray.astype(int)
    
#     imshow(imgMask)
    
    # get contours of borders
    contours, _ = cv2.findContours(imgMask,cv.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    #     titleStr = "blocksize " + str(blockSize)
#     mDrawContours(img, contours)

    # TODO: remove small noise with contours areas and contours fill

    # get Contour Points
    contoursList = []
    for contour in contours:
        if cv2.contourArea(contour) > 1e5:
            contoursList.append(contour)
    
    contours = array(contoursList).reshape(-1,2)

    
    # Close contour
#     contours = contours+contours[0,:]
    
    ## Split Contours into 2 lines
    breakContourIndexes = any(
        c_[contours [:,0] == 0,
        contours [:,1] == 0,
        contours [:,0] == img.shape[1]-1,
        contours [:,1] == img.shape[0]-1],
        axis=1)
    
    contourIndexes = np.where(breakContourIndexes)
    
    indexesToSplit = c_[contourIndexes[0],contourIndexes[0]+1].flatten()
    
    contourLines = np.split(contours,indexesToSplit)

    # matrix that performs a rotation of 90 degrees clockwise
    rotationMatrix = array([[0, -1],[1, 0]])
        
    pointSList=[]
    pointEList=[]
    pointFList=[]
    contourIgnored = []
    for outerLine in contourLines:
        
        # ignore short lines
        if len(outerLine) < 50:
            contourIgnored.append(outerLine)
            continue
        
        x,y = getEquidistantPoints(outerLine)
        
        # TODO: use pchip to make contours and respective tangents smoother
        # x,y = ppval(ppchip(x,y))
        
        equidistantLine = np.c_[x, y]
        
        tanjentVectors = equidistantLine[2:] - equidistantLine[:-2]
        
        perpendicularVectors = matmul(tanjentVectors,rotationMatrix)
        
        # normalize vectors
        vectorsNorm = linalg.norm(perpendicularVectors,axis=1)
        perpendicularVectors = perpendicularVectors/c_[vectorsNorm,vectorsNorm]
        
        perpendicularPoints = perpendicularVectors*200 + equidistantLine[1:-1]
        
        # TODO: sort or remove perpendicularPoints if they intersect (remove points that intersect the most lines
        
        # loop through each point to find the distance to the nearest contours
        for p in range(perpendicularPoints.shape[0]):
            
            # Get image indexes using bresenham Line
            bresenhamLine = array(line(round(equidistantLine[p+1,0]),round(equidistantLine[p+1,1]),
                      round(perpendicularPoints[p,0]),round(perpendicularPoints[p,1]))).T
            
            # remove indexes outside image
            bresenhamLine = bresenhamLine[where((bresenhamLine[:,0]>=0)*(bresenhamLine[:,1]>=0)*
                    (bresenhamLine[:,0]<img.shape[1])*(bresenhamLine[:,1]<img.shape[0]))]
            
            # get line pixels from image
            imageValues = imgAsInt[bresenhamLine[:,1],bresenhamLine[:,0]]
            
            # Low pass filter to smooth the values
            imageValues = uniform_filter1d(imageValues,10)
#             imageValues[imageValues<20]=0
            
            # get local maxima
#             peakIndexes = scipy.signal.find_peaks(diff(imageValues))
            peakIndexes = scipy.signal.find_peaks(imageValues)
            
            # Ignore point if there are no peaks
            if len(peakIndexes[0]) < 2 or peakIndexes[0][0]>15:
                if len(peakIndexes[0]) == 1:
                    pointF = bresenhamLine[peakIndexes[0][0]]
                    pointFList.append(pointF)
                continue
            
            # Get Start Point and End Point
            pointS = bresenhamLine[peakIndexes[0][0]]
            pointE = bresenhamLine[peakIndexes[0][1]]
            
#             plot(pointS[0],pointS[1],'r.')
#             plot(pointE[0],pointE[1],'b.')
            
            # Add Points to Lists
            pointSList.append(pointS)
            pointEList.append(pointE)
        
        
    pointSList = array(pointSList)
    pointEList = array(pointEList)
    pointFList = array(pointFList)
    allLines = c_[pointSList,pointEList,pointSList].reshape(-1,2)
    
    # save Images in png file
    
    imshow(img)
    plot(pointSList[:,0],pointSList[:,1],'r.')
    plot(pointEList[:,0],pointEList[:,1],'b.')
#     plot(pointFList[:,0],pointFList[:,1],'g.')
    plot(allLines[:,0],allLines[:,1])
    
#     savefig(plotFolder + imageName + '.eps', format='eps')
    
    # compute Distance
    distance = sqrt(sum((pointSList-pointEList)**2,axis=1))
    
    distanceList.append(mean(distance))

plot(distanceList)
print("end")