
from numpy import *
import numpy as np
import cv2 as cv
import cv2 as cv2
from matplotlib.pyplot import *
from PIL import Image
import matplotlib.pyplot as plt

def getEquidistantPoints(line):
    x,y = line.T
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])
    
    dist_array = (x[:-1]-x[1:])**2 + (y[:-1]-y[1:])**2

    lineLength = sum(sqrt(sum(diff(line,axis=0)**2,axis=1)))

    t = np.linspace(0,u.max(),(lineLength/2).astype(int))
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    
    return xn, yn

ion()
imageName = "9024KO_db11_16h_1.czi.tiff"
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
# imageName = "9024KO_db11_3h_8.czi.tiff"
imageName = "w1118_db11_16h_1.czi.tiff"
img = cv.imread('./data/20201125_CG1139KO/tif/' + imageName,0)
subplot(2,3,1)
imshow(img)
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,20,1,cv2.THRESH_BINARY)

# thresh = cv2.adaptiveThreshold(imgray,1,cv2.ADAPTIVE_THRESH_MEAN_C,
#             cv2.THRESH_BINARY,11,2)
# thresh = cv2.adaptiveThreshold(imgray,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,11,2)

# kernel = np.ones((30, 30), np.uint8)
# for i in range(10):
# #     blockSize = 31
#     blockSize = 3+i*100
# #     C = 2+i*10
#     C = 2
#     # adaptiveThreshold better for images with brightness differences
#     thresh = cv2.adaptiveThreshold(255-imgray,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize,C)
#     
# #     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     
#     subplot(2,5,i+1)
#     imshow(thresh)

subplot(2,3,2)
imshow(thresh)

kernel = np.ones((30, 30), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
subplot(2,3,3)
imshow(thresh)

contours, hierarchy = cv2.findContours(thresh,cv.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# img1 = cv2.drawContours(img, contours, -1, (0,255,0), 3)
#    
# subplot(2,3,4)
# imshow(img1)

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

# imgdraw = img
# cv2.drawContours(imgdraw, contoursFiltered, -1, (0,255,0), 3)
# subplot(2,3,4)
# imshow(imgdraw)

contourImage = np.zeros(img.shape[0:2])
for contourPoints in contoursFiltered:
    if contourPoints.shape[1] != 1:
        print ("verify shape of points")
#         contourImage = Image.fromarray(contourPoints)
    contourPoints = contourPoints[:,0,:]
    
    # get points to remove
    c = np.concatenate((contourPoints[[0]],contourPoints))
    
    # remove distances above kernel 30,30
    
    """
    Instead of separating contour polylines by distance, simply
    separate the if they reach the borders
    if x == 0 or x == width or y == 0 or y == height
    """
#     d = np.sum(np.abs(np.diff(c,axis=0)),axis=1)>100
    pointDistances = sqrt(sum(diff(c,axis=0)**2,axis=1))

    indexesD = np.where(pointDistances>100)

    contourLines = np.split(contourPoints,indexesD[0])
    
    contourEquidistantLines = []
    for line in contourLines:
        # ignore short lines
        if len(line) < 5:
            continue
        
        x,y = getEquidistantPoints(line)

        contourEquidistantLines.append(np.c_[x, y])
        print (max(x),max(y))
        contourImage[y.astype(int),x.astype(int)] = 1
#     contourPointsFiltered = contourPoints[d]

    
kernel = np.ones((25, 25))
contourImageDilated = cv2.dilate(contourImage,kernel,iterations = 1)

subplot(235)
imshow(contourImageDilated)

subplot(236)
newMask = np.fmax(0,thresh-contourImageDilated)

interiorImage = newMask*imgray
imshow(interiorImage)


"""
use approxPolyDP to get points of borders without any if the insides
or use te contrary (ingnore points to far from source line)
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

print()