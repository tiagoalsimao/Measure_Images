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
import cv2 as cv2
from matplotlib.pyplot import *
from PIL import Image
import matplotlib.pyplot as plt

def getEquidistantPoints(outerLine):
    x,y = outerLine.T
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])
    
    dist_array = (x[:-1]-x[1:])**2 + (y[:-1]-y[1:])**2

    lineLength = sum(sqrt(sum(diff(outerLine,axis=0)**2,axis=1)))

    t = np.linspace(0,u.max(),(lineLength/5).astype(int))
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    
    return xn, yn

def drawContours(img,contours):
    imgDraw = img.copy()
    cv2.drawContours(imgDraw , contours, -1, (0,255,0), 3)
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

plt.ion()
# imageName = "9024KO_db11_16h_1.czi.tiff"
# imageName = "9024KO_db11_3h_4.czi.tiff"
# imageName = "9024KO_db11_3h_7.czi.tiff"
# imageName = "9024KO_db11_16h_7.czi.tiff"
imageName = "w1118_db11_16h_3.czi.tiff"
# imageName = "w1118_sucrose_16h_9.czi.tiff"
# imageName = "w1118_db11_16h_7.czi.tiff"
# imageName = "9024KO_db11_3h_8.czi.tiff"
# imageName = "w1118_db11_16h_1.czi.tiff"
# imageName = "9024KO_db11_16h_8.czi.tiff"
imageName = "9024KO_db11_16h_3.czi.tiff"
img = cv.imread('./data/20201125_CG1139KO/tif/' + imageName,0)
subplot(2,3,1)
imshow(img)
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

for i in range(1):
    
    # Filter image to eliminate
    kg = 11
    imgrayBlur = cv2.GaussianBlur(imgray, (kg, kg), 0)
    
#     ret,thresh = cv2.threshold(imgrayBlur,20,1,cv2.THRESH_BINARY)
#     subplot(2,5,i+1)
#     imshow(imgrayBlur)
    
#     subplot(232)
#     imshow(thresh)
    
    blockSize = 61
#     blockSize = 21+i*8
    C = 2
    
    # adaptiveThreshold better for images with brightness differences
    thresh = cv2.adaptiveThreshold(255-imgrayBlur,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize,C)
    
#     subplot(2,5,i+1)
#     imshow(thresh)

    ko = 2
    kernelOpening = np.ones((ko, ko), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelOpening)
    
    kc = 8
    kernelClose = np.ones((kc, kc), np.uint8)    
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernelClose)
    
    threshClosed = thresh.copy()
    kcbig = 70
    kernel = np.ones((kcbig, kcbig))
    threshClosed = cv2.morphologyEx(threshClosed, cv2.MORPH_CLOSE, kernel)
    threshClosed = cv2.morphologyEx(threshClosed, cv2.MORPH_ERODE, kernel = np.ones((3, 3)))
    
#     n = 5
#     kcbig = 20
#     for j in range (5):
#         
#         kernel = np.ones((kcbig, kcbig))
#         threshClosed = cv2.morphologyEx(threshClosed, cv2.MORPH_DILATE, kernel)
# 
#     for j in range (5):
#         kernel = np.ones((kcbig, kcbig))
#         threshClosed = cv2.morphologyEx(threshClosed, cv2.MORPH_ERODE, kernel)
#     
#     imshow(threshClosed)

    contoursExternal, _ = cv2.findContours(threshClosed,cv.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    
    if len(contoursExternal) != 1:
        print("not yet done")
    
    threshWithBorderContours = thresh.copy()
    
    # Add External contour to image with adaptative threshhold
    cv2.polylines(threshWithBorderContours ,contoursExternal,True,1,thickness=3)
    
    threshWithBorderContours = cv2.morphologyEx(threshWithBorderContours, cv2.MORPH_CLOSE, kernelClose)
    
#     imshow(threshWithBorderContours)
    
    contours, hierarchy = cv2.findContours(threshWithBorderContours,cv.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy.shape[0] != 1:
        print ("not done")
    hierarchy = hierarchy[0]
    
#     # filter contours that are points
#     # how to filter contours and hierarchy? maybe remove contouturs firts
#     # OR DRAW SMALL CONTOURS OVER IMAGE AND GET CONTOURS AGAIN
#     contoursFilteredIndexes = []
#     for c in range(len(contours)):
#         area = cv2.contourArea(contours[c])
#         if area > 1e2:
#             contoursFilteredIndexes.append(c)
#     
#     
#     contoursFiltered =  array(contours,dtype=object)[contoursFilteredIndexes]

    
#     drawContours(contoursFiltered)
    # get level 1 hierarchy where contours have contour 0 as Parent
    hierarchy_Level1_indexes = where(hierarchy[:,3]==0)[0]
    
    # get level 2 hierarchy where contours have contours of Level 1 as Parent
    hierarchy_Level2_indexes = where(np.in1d(hierarchy[:,3],hierarchy_Level1_indexes))[0]
    
#     imgDraw = img.copy()
#     cv2.drawContours(imgDraw , contours, -1, (0,255,0), 3)
    
    # get Contour Points
    outerContourPoints = contours[0].reshape(-1,2)
    
    ## Split Contours lines
    breakContourIndexes = any(
        c_[outerContourPoints [:,0] == 0,
        outerContourPoints [:,1] == 0,
        outerContourPoints [:,0] == img.shape[1],
        outerContourPoints [:,1] == img.shape[0]],
        axis=1)
    
    borderIndexes = np.where(breakContourIndexes)
    
    indexesToSplit = c_[borderIndexes[0],borderIndexes[0]+1].flatten()
    
    outerContourLines = np.split(outerContourPoints,indexesToSplit)

    minDistPoints=[]
    minDistList=[]
    for outerLine in outerContourLines:
        # ignore short lines
        if len(outerLine) < 5:
#             print("tratar")
            continue
        
        x,y = getEquidistantPoints(outerLine)

        equidistantLine = np.c_[x, y]
        
        tanjentVectors = equidistantLine[2:] - equidistantLine[:-2]
        
        # matrix that performs a rotation of 90 degrees clockwise
        rotationMatrix = array([[0, -1],[1, 0]])
        
        perpendicularVectors = matmul(tanjentVectors,rotationMatrix)
        
        # normalize vectors
        vectorsNorm = linalg.norm(perpendicularVectors,axis=1)
        perpendicularVectors = perpendicularVectors/c_[vectorsNorm,vectorsNorm]
        
        perpendicularPoints = perpendicularVectors*50 + equidistantLine[1:-1]
        
#         
# # plot
# a = equidistantLine
# a = a[1:-1]
# b = perpendicularPoints
# # plot(b[:,0],b[:,1],'ro')
# c = c_[a,b,a].reshape(-1,2)
# plot(c[:,0],c[:,1])
# # plot(equidistantLine[:,0],equidistantLine[:,1])
#         
        # loop through each point to find the distance to the nearest contours
        for p in range(perpendicularPoints.shape[0]):
            
            # Get image indexes using bresenham Line
            bresenhamLine = array(line(round(equidistantLine[p+1,0]),round(equidistantLine[p+1,1]),
                      round(perpendicularPoints[p,0]),round(perpendicularPoints[p,1]))).T
            
            bresenhamLine = bresenhamLine[where((bresenhamLine[:,0]>=0)*(bresenhamLine[:,1]>=0)*
                    (bresenhamLine[:,0]<=img.shape[1])*(bresenhamLine[:,1]<=img.shape[0]))]
            
            imageValues = threshWithBorderContours[bresenhamLine[:,0],bresenhamLine[:,1]]
            index = argmax(diff(imageValues)==1)
            point = bresenhamLine[index]
            
                    
                
            pPoint = Point(equidistantLine[p+1])
            perpendicularLine = LineString([equidistantLine[p+1],perpendicularPoints[p]])
            
            # get level2 contours to find the closest point
            minDist_L2 = Inf
            minDistPoint_2 = None
            for h in hierarchy_Level2_indexes:
                # filter contours. need to find a way to filter before the for p loop
                if cv2.contourArea(contours[h]) < 1e2:
                    continue
                
                # get contour Line (see if need to add last point to close contour)
                contourLine = LineString(contours[h].reshape(-1,2))
                
                # verifit if perpendicular intersects contour Line
                if perpendicularLine.intersects(contourLine):
                    # get all intersection points
                    iPoints = perpendicularLine.intersection(contourLine)
                    # loop through each intersection point to get the distance
#                     print(p,h)
                    if type(iPoints) is Point:
                        if pDist < minDist_L2:
                            minDist_L2 = pDist
                            minDistPoint_2 = ip
                    else:
                        for ip in iPoints:
                            pDist = pPoint.distance(iPoints)
                            # get the point with the minimum distance
                            if pDist < minDist_L2:
                                minDist_L2 = pDist
                                minDistPoint_2 = ip
            
            # get level1 contours to find the closest point
            minDist_L1 = Inf
            minDistPoint_1 = None
            for h in hierarchy_Level1_indexes:
                # filter contours. need to find a way to filter before the for p loop
                if cv2.contourArea(contours[h]) < 1e2:
                    continue
                
                # get contour Line (see if need to add last point to close contour)
                contourLine = LineString(contours[h].reshape(-1,2))
                
                # verify if perpendicular intersects contour Line
                if perpendicularLine.intersects(contourLine):
                    # get all intersection points
                    iPoints = perpendicularLine.intersection(contourLine)
                    # loop through each intersection point to get the distance
#                     print(p,h)
                    if type(iPoints) is Point:
                        # ignore
                        continue
                    else:
                        for ip_index in range(1,len(iPoints)):
                            ip = iPoints[ip_index]
                            pDist = pPoint.distance(iPoints)
                            # get the point with the minimum distance
                            if pDist < minDist_L1:
                                minDist_L1 = pDist
                                minDistPoint_1 = ip
            
            # add min distance point to list
            if minDistPoint_2 == None and minDistPoint_1 == None:
                continue
            if minDistPoint_2 != None:
                minDistPoints.append([minDistPoint_2.x,minDistPoint_2.y])
                minDistList.append(minDist_L2)
            elif minDistPoint_1 != None:
                minDistPoints.append([minDistPoint_1.x,minDistPoint_1.y])
                minDistList.append(minDist_L1)
            else:
                if minDist_L1 < minDist_L2:
                    minDistPoints.append([minDistPoint_1.x,minDistPoint_1.y])
                    minDistList.append(minDist_L1)
                else:
                    minDistPoints.append([minDistPoint_2.x,minDistPoint_2.y])
                    minDistList.append(minDist_L2)
        
    a = array(minDistPoints)
    plot(a[:,0],a[:,1],'r.')
#             for 
        
#         print (max(x),max(y))
#         contourImage[y.astype(int),x.astype(int)] = 1
        
#         subplot(233)
#         plot(x,y)
#         subplot(234)
#         plot(x,y)
        
        # compute normal distances to inner contour
        
#     
#     subplot(2,5,i+1)
#     imshow(threshClosed1)

drawContours(img,contours)
plot(a[:,0],a[:,1],'ro')


subplot(233)
imshow(thresh)

img1 = cv2.drawContours(img, contours, -1, (0,255,0), 3)
subplot(2,3,3)
imshow(img1)

contoursFiltered=[]
for i in range(len(contours)):
    
    cnt = contours[i]
    # ignore contours with few points
    if len(cnt) < 1e3:
        continue
    
    area = cv.contourArea(cnt)
    # ignore small areas
    if area < 1e4:
        continue
    
    print (area)
    contoursFiltered.append(cnt)

imgdraw = img.copy()
cv2.drawContours(imgdraw, contoursFiltered, -1, (0,255,0), 3)
subplot(234)
imshow(imgdraw)

contourImage = np.zeros(img.shape[0:2])
for contourPoints in contoursFiltered:
    if contourPoints.shape[1] != 1:
        print ("verify shape of points")
        contourImage = Image.fromarray(contourPoints)
    contourPoints = contourPoints[:,0,:]
#     
#     # get points to remove
#     c = np.concatenate((contourPoints[[0]],contourPoints))
#      
#     # remove distances above kernel 30,30
#      
# #     d = np.sum(np.abs(np.diff(c,axis=0)),axis=1)>100
#     pointDistances = sqrt(sum(diff(c,axis=0)**2,axis=1))
#  
#     indexesD = np.where(pointDistances>100)

    """
    Instead of separating contour polylines by distance, simply
    separate the if they reach the borders
    if x == 0 or x == width or y == 0 or y == height
    """
    
    # get indexes of points in the border of the image
    breakContourIndexes = any(
        c_[contourPoints[:,0] == 0,
        contourPoints[:,1] == 0,
        contourPoints[:,0] == img.shape[1],
        contourPoints[:,1] == img.shape[0]],
        axis=1)
    
    borderIndexes = np.where(breakContourIndexes)
    
    indexesToSplit = c_[borderIndexes[0],borderIndexes[0]+1].flatten()
    
    contourLines = np.split(contourPoints,indexesToSplit)
    
    contourEquidistantLines = []
    for outerLine in contourLines:
        # ignore short lines
        if len(outerLine) < 5:
            continue
        
        x,y = getEquidistantPoints(outerLine)

        contourEquidistantLines.append(np.c_[x, y])
        print (max(x),max(y))
        contourImage[y.astype(int),x.astype(int)] = 1
        
        subplot(233)
        plot(x,y)
        subplot(234)
        plot(x,y)
#     contourPointsFiltered = contourPoints[d]



# get inner contour
for i in range(1):
    k= 60
    kernel = np.ones((k, k))
    threshClosedEroded = cv2.erode(threshClosed,kernel,iterations = 1)
    
    innerthresh = thresh*threshClosedEroded # logical and
    
    ko = 3
    kernelOpening = np.ones((ko, ko), np.uint8)
    innerthresh = cv2.morphologyEx(innerthresh, cv2.MORPH_OPEN, kernelOpening)
    
    kc = 5
    kernelClose = np.ones((kc, kc), np.uint8)
    innerthresh = cv2.morphologyEx(innerthresh, cv2.MORPH_CLOSE, kernelClose)
    
#     subplot(2,5,6)
#     imshow(innerthresh)


innerContours, hierarchy = cv2.findContours(innerthresh,cv.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

innerthreshDraw = innerthresh*0
innerthreshDraw = cv2.drawContours(innerthreshDraw, innerContours, -1, (0,255,0), 3)

subplot(236)
imshow(innerthreshDraw)

subplot(236)
# newMask = np.fmax(0,thresh*(1-contourImageDilated/2))
newMask = thresh*(1-contourImageDilated/2)

interiorImage = imgray*(1-contourImageDilated/2)
imshow(interiorImage)

ret,innerthresh = cv2.threshold(interiorImage,30,1,cv2.THRESH_BINARY)
subplot(236)
imshow(innerthresh)

"""
use approxPolyDP to get points of borders without any if the insides
or use te contrary (ingnore points to far from source outerLine)
"""


cnt = contoursFiltered[0]
M = cv.moments(cnt)
print( M )

# cx = int(M['m10']/M['m00'])
# cy = int(M['m01']/M['m00'])

area = cv.contourArea(cnt)

perimeter = cv2.arcLength(cnt,True)
epsilon = 1e3*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

imgdraw2 = img
cv2.drawContours(imgdraw2, [approx], -1, (255,0,0), 3)
subplot(2,3,3)
imshow(imgdraw2)

hull = cv.convexHull(cnt)

k = cv.isContourConvex(cnt)


