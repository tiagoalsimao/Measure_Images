import csv
from install import install
import os

install("numpy")
install("cv2") # opencv-python
install("matplotlib")
install("scipy")
install("skimage")

import scipy.ndimage
import scipy.signal
import skimage.draw
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from tkinter import Tk,messagebox # plt.plot msgBox

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

################### Input variables ###################
srcFolder = './data/20201125_CG1139KO/tif/'
fileNameSpec = "*.tiff"

# number of points per pixel to compute distances
distanceBetweenPointsInPixes = 5

# Sort point such that there are no intersections (more or less)
bSortPoints = True
# bSortPoints = False

# Write values to CSV
writeFile = False

# Save each image plot one at a time
showPlotImages = True
# showPlotImages = False

# Save image plot (Takes much longer)
savePlotImages = False
# savePlotImages = True

# Destination folder to save images as tif
plotFolder = './data/20201125_CG1139KO/plot/'

#######################################################

perpendicularVectorsLenght = 100

def msgBox(text):  
#     top = Tk()
#     top.geometry("100x100")
    messagebox.showinfo('Information',text)
#     top.mainloop()

# Checks intersection between to segments AB and CD
# Source: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) >= (B[1]-A[1])*(C[0]-A[0])

# Return true if line segments AB and CD intersect
# Source: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def sortNoIntersections(originalPoints,parallelPoints):
    
    # Selection Sort algorithm
    for p in range(originalPoints.shape[0]-1):
        A = originalPoints[p]
        B = parallelPoints[p]
        for t in range(p+1,np.min([p+100,parallelPoints.shape[0]])):
            C = originalPoints[t]
            D = parallelPoints[t]

            # Verify intersections between AB and CD segments and Swap B and D if true
            if intersect(A,B,C,D):
                # Swap B and D
                parallelPoints[[p,t]] = parallelPoints[[t,p]]
    
    return parallelPoints
        
# Get equidistant points from outerLine
# Source: https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values
def getEquidistantPoints(line):
    # get x and y coordinates of each point of line
    x,y = line.T
    
    # get distance between each point
    dist = np.sqrt(np.sum(np.diff(line,axis=0)**2,axis=1))
    
    # get array of cumulative sum of distances and add zero at first position
    u = np.r_[[0],np.cumsum(dist)]
    
    # get length total
    lineLength = np.sum(dist)

    # Get evenly spaced values. Number of points is lineLength/distanceBetweenPointsInPixes
    t = np.linspace(0,lineLength,(lineLength/distanceBetweenPointsInPixes).astype(int))
    
    # Linear Interpolation of x and y
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    
    # return line as np.array with [x,y] as coordinates
    return np.c_[xn, yn]

def getThreshImage(image,tbin,ko,kc,kd,mode='binary',blockSize=21):
    
    # adaptiveThreshold better for images with brightness differences
    
    if mode=='binary':
        _,imgMask = cv2.threshold(image,20,1,cv2.THRESH_BINARY)
        
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

# return indexes it np.array for first -1 and the first 1 after it
def getStartEndIndexes(arrayDiff):
    indexStart = np.where(arrayDiff==-1)[0]
    indexEnd = np.where(arrayDiff==1)[0]
    
    for indS in indexStart:
        for indE in indexEnd:
            if indS < indE:
                return indS, indE
            
    return -1, -1

def getPerpendicularPointsThresh (imgMask,bresenhamLine):
    # get line pixels from image
    imageValues = imgMask[bresenhamLine[:,1],bresenhamLine[:,0]]
    
    iS,iE = getStartEndIndexes(np.diff(imageValues))
    
    if iS == -1 or iE == -1 :
        return None, None
    
    pointS = bresenhamLine[iE]
    pointE = bresenhamLine[iS]
    
    return pointS, pointE


if showPlotImages or savePlotImages:
    # To maximaze window
    plt.switch_backend('QT5Agg')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    
    # Make plots appear immediately
    plt.ion()

# List of Images
imageList = glob.glob(srcFolder + fileNameSpec)

# List length
nList = len(imageList)

# List to write to CSV output file
listToCSV=[]

# try to open file
if writeFile:
    f = open('numbers2.csv', 'w', newline='')

# Loop through each image
for i in range(0,nList):
# for i in range(20,50):
# for i in range(1):
#     i = 24
    
    plt.clf()
    
    # Image name
    imageName = os.path.basename(imageList[i])
    
    print("Processing " + str(i) + " of " + str(nList) + ": " + imageName)
    
    img = cv2.imread(srcFolder + imageName,0)
    
    # Rotate Image to horizontal to better view
    if img.shape[0] > img.shape[1]:
        img = np.rot90(img)
        
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Filter image to eliminate high frequency noise
    kg = 11
    imgray = cv2.GaussianBlur(imgray, (kg, kg), 0)
    
    # get image Mask based on binary threshold
    tbin = 20
    ko = 2
    kc = 31
    kd = 10
    imgMask = getThreshImage(imgray,tbin,ko,kc,kd)
    
    # get image Mask based on adaptative gaussian threshold
    imgAdaptMask = getThreshImage(imgray,tbin,ko,kc,kd)
    
    # Image to int np.array to compute bresenhamLine
    imgAsInt = imgray.astype(int)
    
#     plt.imshow(imgMask)
    
    # get contours of borders
    contours, _ = cv2.findContours(imgMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Get only big contour
    contoursList = []
    for contour in contours:
        # Take in account contours with big areas
        if cv2.contourArea(contour) > 1e5:
            # Append first point to close contour
            contoursList.append(np.r_[contour,contour[[0]]])
    
    # merge all loops in single 2D array
    contours = np.array(contoursList).reshape(-1,2)
    
    # TODO: Get gut thickness
    
#     contourArea = cv2.contourArea(contours)
#     
#     import skimage.morphology
#     
#     skeleton = skimage.morphology.skeletonize(imgMask)
#     
#     gutThickness = contourArea/np.sum(skeleton)
    
    ## Split Contours into 2 lines
    # find indexes that are in image borders
    breakContourIndexes = np.any(
        np.c_[contours [:,0] == 0,
        contours [:,1] == 0,
        contours [:,0] == img.shape[1]-1,
        contours [:,1] == img.shape[0]-1],
        axis=1)
    
    # find indexes to split
    contourIndexes = np.where(breakContourIndexes)
    indexesToSplit = np.c_[contourIndexes[0],contourIndexes[0]+1].flatten()
    
    # split contours
    contourLines = np.split(contours,indexesToSplit)
    
    # matrix that performs a rotation of 90 degrees clockwise
    # as image has y coordinates inverted (top is 0 and bottom is max)
    # this rotation matrix seems reflected
    rotationMatrix = np.array([[0, -1],[1, 0]])
    
    # Initialize lists
    pointSList=[]
    pointEList=[]
    pointFList=[]
    contourIgnored = []
    
#     newContourLines = np.array([])
    # Loop through each Line
    for outerLine in contourLines:
        
        # ignore short lines
        if len(outerLine) < 50:
            contourIgnored.append(outerLine)
            continue
        
        # Get equidistant border Points/tangent Points
        equidistantLine = getEquidistantPoints(outerLine)
        
        # Get tanjentVectors with are the diference of the adjacent points
        # i.e. if points are [P1, P2, P3, ... , Pn] then the tangent vectors
        # will get [P3-P1, P4-P2, ..., Pn-P(n-2)]
        tanjentVectors = equidistantLine[2:] - equidistantLine[:-2]
        
        # Get perpendicular vector by multiplying vectors with rotation matrix
        perpendicularVectors = np.matmul(tanjentVectors,rotationMatrix)
        
        # normalize vectors
        vectorsNorm = np.linalg.norm(perpendicularVectors,axis=1)
        perpendicularVectors = perpendicularVectors/np.c_[vectorsNorm,vectorsNorm]
        
        # get perpendicular points by adding perpendicularVectors to Border Points
        perpendicularPoints = perpendicularVectors*perpendicularVectorsLenght + equidistantLine[1:-1]
        
        # Sort perpendicular Points so lines don't intersect
        if bSortPoints and distanceBetweenPointsInPixes < perpendicularVectorsLenght:
            perpendicularPoints = sortNoIntersections(equidistantLine[1:-1],perpendicularPoints)
        
        # loop through each point to find the distance to the nearest contours
        for p in range(perpendicularPoints.shape[0]):
            
            # Get image indexes using bresenham Line
            bresenhamLine = np.array(skimage.draw.line(round(
                equidistantLine[p+1,0]),round(equidistantLine[p+1,1]),
                round(perpendicularPoints[p,0]),round(perpendicularPoints[p,1]))).T
            
            # remove indexes outside image
            bresenhamLine = bresenhamLine[np.where((bresenhamLine[:,0]>=0)*(bresenhamLine[:,1]>=0)*
                    (bresenhamLine[:,0]<img.shape[1])*(bresenhamLine[:,1]<img.shape[0]))]
            
            # get line pixels from image
            imageValues = imgAsInt[bresenhamLine[:,1],bresenhamLine[:,0]]
            
            # Low pass filter to smooth the values
            imageValues = scipy.ndimage.filters.uniform_filter1d(imageValues,10)
#             imageValues[imageValues<20]=0
            
            # get local maxima
#             peakIndexes = scipy.signal.find_peaks(np.diff(imageValues))
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
            
            # Add Points to Lists
            pointSList.append(pointS)
            pointEList.append(pointE)
    
    # ignore if no points found
    if len(pointSList) == 0:
        continue
    
    # Convert Points List to Array
    pointSList = np.array(pointSList)
    pointEList = np.array(pointEList)
    pointFList = np.array(pointFList)
        
    if showPlotImages or savePlotImages:
        
        # Line to connect border points and respect perpendicular points 
        allLines = np.c_[pointSList,pointEList,pointSList].reshape(-1,2)
        
        # Plot image and points
        fig = plt.imshow(img)
        plt.plot(pointSList[:,0],pointSList[:,1],'r.')
        plt.plot(pointEList[:,0],pointEList[:,1],'b.')
    #     plt.plot(pointFList[:,0],pointFList[:,1],'g.')
        plt.plot(allLines[:,0],allLines[:,1])
    
    if showPlotImages:
        msgBox("Click OK for next figure")
        
    # Save Images as png file in plotFolder
    if savePlotImages:
        plt.savefig(plotFolder + imageName + '_plot.png',dpi=200)
    
    # compute cell thickness
    cell_thickness = np.sqrt(np.sum((pointSList-pointEList)**2,axis=1))
    
    # add values to row list 
    rowToCSV = imageName[:-9].split("_")
    rowToCSV.append(np.round(np.mean(cell_thickness),4))
    rowToCSV.append(np.round(np.median(cell_thickness),4))
    
    # add row line to list to write in CSV file
    listToCSV.append(rowToCSV)

# Write values to csv file
if writeFile:
    with f:
    
        writer = csv.writer(f)
        # Write Header
        writer.writerow(['Line','treat','td','gut','cell mean (um)','cell median (um)', 'gut thickness (um)'])
        
        for row in listToCSV:
            writer.writerow(row)

msgBox("Finished.")
