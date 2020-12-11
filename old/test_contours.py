from install import install
from _ast import If
from cv2 import drawContours

install("numpy")
install("cv2")
install("PIL")
install("matplotlib")

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

def drawContours(image,contours):
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
imageName = "9024KO_db11_16h_3.czi.tiff"

# Input variables
srcFolder = './data/20201125_CG1139KO/tif/'
fileNameSpec = "*.tiff"

# Destination folder to save images as tif
plotFolder = './data/20201125_CG1139KO/plot/'

# List of Images
imageList = glob.glob(srcFolder + fileNameSpec)

# List length
nList = len(imageList)

# Loop through each image
# for i in range(0,nList):
for i in range(1):
    
    # Image name
    imageName = os.path.basename(imageList[0])
    
    img = cv.imread(srcFolder + imageName,0)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Filter image to eliminate
#     imgrayBlur = imgray.copy()
    kg = 11
    imgrayBlur = cv2.GaussianBlur(imgray, (kg, kg), 0)
#     titleStr = "gaussBlurr " + str(kg)
    
    # adaptiveThreshold better for images with brightness differences
    blockSize = 31
    thresh = cv2.adaptiveThreshold(255-imgrayBlur,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize,2)
#     titleStr = "blocksize " + str(blockSize)
    
    # remove small noise
    ko = 2
    kernelOpening = np.ones((ko, ko), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelOpening)
    thresh01 = thresh.astype(int)
    
    # Merge blocks
    kc = 8
    kernelClose = np.ones((kc, kc), np.uint8)    
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernelClose)
    
    # TODO: remove small noise with contours areas and contours fill
    
    # Get Outer Contours
    threshClosed = thresh.copy()
    kcbig = 70
    kernel = np.ones((kcbig, kcbig))
    threshClosed = cv2.morphologyEx(threshClosed, cv2.MORPH_CLOSE, kernel)
    # Erode to have contours pass in the middle outer borders
#     ke = 5
#     kernelErode = np.ones((ke, ke))
#     threshClosed = cv2.morphologyEx(threshClosed, cv2.MORPH_ERODE, kernel = kernelErode)
    
    # get contours of borders
    contoursBorder, _ = cv2.findContours(threshClosed,cv.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # remove small contours
    contoursBorderNew = []
    for cb in contoursBorder:
        if cv2.contourArea(cb) > 1e4:
            contoursBorderNew.append(cb)
        else
            cv2.fillPoly(cb)
    
    
    
    contours, _ = cv2.findContours(thresh,cv.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
#     subplot(2,5,i+1)
#     drawContours(img, contours)
# 
# if True:
    newContours =[]
    for c in contours:
        if len(c) > 500:
            newContours.append(c)
    if len(newContours) != 1:
        print("not Done")
    
    contours = newContours
    # get Contour Points
    pixelsBorder = contoursBorder[0].reshape(-1,2)
        
    ## Split Contours into 2 lines
    breakContourIndexes = any(
        c_[pixelsBorder [:,0] == 0,
        pixelsBorder [:,1] == 0,
        pixelsBorder [:,0] == img.shape[1],
        pixelsBorder [:,1] == img.shape[0]],
        axis=1)
    
    borderIndexes = np.where(breakContourIndexes)
    
    indexesToSplit = c_[borderIndexes[0],borderIndexes[0]+1].flatten()
    
    outerContourLines = np.split(pixelsBorder,indexesToSplit)

    # matrix that performs a rotation of 90 degrees clockwise
    rotationMatrix = array([[0, -1],[1, 0]])
        
    minDistPoints=[]
    minDistList=[]
    linePoints=[]
    for outerLine in outerContourLines:
        # ignore short lines
        if len(outerLine) < 5:
#             print("tratar")
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
        
        perpendicularPoints = perpendicularVectors*50 + equidistantLine[1:-1]
        
        # TODO: sort or remove perpendicularPoints if they intersect (remove points that intersect the most lines
        
        # loop through each point to find the distance to the nearest contours
        for p in range(perpendicularPoints.shape[0]):
            
            # Get image indexes using bresenham Line
            bresenhamLine = array(line(round(equidistantLine[p+1,0]),round(equidistantLine[p+1,1]),
                      round(perpendicularPoints[p,0]),round(perpendicularPoints[p,1]))).T
            
            # remove indexes outside image
            bresenhamLine = bresenhamLine[where((bresenhamLine[:,0]>=0)*(bresenhamLine[:,1]>=0)*
                    (bresenhamLine[:,0]<=img.shape[1])*(bresenhamLine[:,1]<=img.shape[0]))]
            
            # get line pixels from image
            imageValues = thresh01[bresenhamLine[:,1],bresenhamLine[:,0]]
            
            iS,iE = getStartEndIndexes(diff(imageValues))
            
            if iS == -1 or iE == -1 :
                continue
            
            pointS = bresenhamLine[iE]
            pointE = bresenhamLine[iS]
            
            minDistPoints.append(pointS)
            linePoints.append(pointE)
        
        
    minDistPoints = array(minDistPoints)
    linePoints = array(linePoints)
    allLines = c_[linePoints,minDistPoints,linePoints].reshape(-1,2)
    
    # save Images in png file
    plt.switch_backend('QT5Agg')
    
    imshow(img)
    plot(minDistPoints[:,0],minDistPoints[:,1],'r.')
    plot(linePoints[:,0],linePoints[:,1],'b.')
    plot(allLines[:,0],allLines[:,1])
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    savefig(plotFolder + imageName + '.eps', format='eps')
    
    # To maximaze window
    plt.switch_backend('QT5Agg')
    drawContours(thresh, contoursBorder)
    
    plot(minDistPoints[:,0],minDistPoints[:,1],'r.')
    plot(linePoints[:,0],linePoints[:,1],'b.')
    plot(allLines[:,0],allLines[:,1])
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    savefig(plotFolder + imageName + '_contours.png',dpi=300)
    
    # compute Distance
    distance = sqrt(sum((b-a)**2,axis=1))

print("end")