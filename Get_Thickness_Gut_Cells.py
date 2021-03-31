print("Loading modules...")
import csv
from install import install
import os
import sys

# install("numpy")    
# install("cv2") # opencv-python
# install("matplotlib")
# install("scipy")
# install("skimage")
# install("aicsimageio")

import datetime
import aicsimageio
import scipy.ndimage
import scipy.signal
import skimage.draw
import skimage.morphology
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from tkinter import Tk,messagebox # plt.plot msgBox

################### Input variables ###################
srcFolder = "./data/20201125_CG1139KO/"
fileNameSpec = "*.czi"

# Compute gut_thickness average and median
bGet_gut_thickness = True
# bGet_gut_thickness = False

# Sort point such that there are no intersections (more or less)
bSortPoints = True
# bSortPoints = False

# Write values to CSV
writeFile = True
# writeFile = False

# Save each image plot one at a time
showPlotImages = True
# showPlotImages = False

# Save image plot (Takes much longer)
savePlotImages = False
# savePlotImages = True

# Destination folder to save images as tif
plotFolder = './data/20201125_CG1139KO/plot/'

#######################################################

# number of points per pixel to compute distances
distanceBetweenPoints_px = 5

# length of perpendicular vector
perpendicularVectorsLenght_px = 100
binaryThreshold = 20

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
        for t in range(p+1,np.min([p+perpendicularVectorsLenght_px,parallelPoints.shape[0]])):
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
    
    # get distance between each point
    dist = np.linalg.norm(np.diff(line,axis=0),axis=1)
    
    # get array of cumulative sum of distances and add zero at first position
    u = np.r_[[0],np.cumsum(dist)]
    
    # get length total
    lineLength = np.sum(dist)

    # Get evenly spaced values. Number of points is lineLength/distanceBetweenPoints_px
    t = np.linspace(0,lineLength,(lineLength/distanceBetweenPoints_px).astype(int))
    
    # get x and y coordinates of each point of line
    x,y = line.T
    
    # Linear Interpolation of x and y
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    
    # return line as np.array with [x,y] as coordinates
    return np.c_[xn, yn]

def getThreshImage(image,ko,kc,kd,ke,th_bin=binaryThreshold,mode='binary',blockSize=21):
    
    # adaptiveThreshold better for images with brightness differences
    
    if mode=='binary':
        _,imgMask = cv2.threshold(image,th_bin,1,cv2.THRESH_BINARY)
        
    elif mode=='adaptative':
        # Avoid error by ensuring blockSize is odd
        blockSize = blockSize + (0 if ((blockSize % 2) == 1) else 1)
        imgMask = cv2.adaptiveThreshold(255-image,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize,2)
    else:
        print('mode \''+ mode +'\' not recognized.', file=sys.stderr)
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
    
    # Erode
    if ke > 0:
        kernelErode = np.ones((ke, ke), np.uint8)    
        imgMask = cv2.morphologyEx(imgMask, cv2.MORPH_ERODE, kernelErode)
        
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

def getGutThickness(imgMask):
    # make copy of Mask
    imgMaskCopy = imgMask.copy()
     
    # Put image border pixels to zero
    imgMaskCopy[[0,-1],:] = 0
    imgMaskCopy[:,[0,-1]] = 0
     
    # Get distance Map to find and get biggest inscribed circle
    # Source: answer by Yang, https://stackoverflow.com/questions/4279478/largest-circle-inside-a-non-convex-polygon?rq=1
    dist_map = cv2.distanceTransform(imgMaskCopy, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius, _, center = cv2.minMaxLoc(dist_map)
     
    # Get gut maximum thickness
    max_gut_thickness = radius*2
    
    if bGet_gut_thickness:
        # Skeletonize image (this is slow try other method: ) 
        skeleton = skimage.morphology.skeletonize(imgMask,method='zhang')
        
        # ignore values near image borders
        radius_int = int(radius)
        skeleton[:radius_int,:] = False
        skeleton[imHeight-radius_int:,:] = False
        skeleton[:,:radius_int] = False
        skeleton[:,imWidth-radius_int:] = False
        
        # Get gut thickness
        gut_thickness = dist_map[skeleton]*2
        
        return max_gut_thickness, gut_thickness
        
    return max_gut_thickness, None

def getPerpendicularPoints(equidistantLine):
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
    perpendicularPoints = perpendicularVectors*perpendicularVectorsLenght_px + equidistantLine[1:-1]
    
    ''' TODO: A better way would be to get equidistant points from an offset polyline. This
    would eliminate the low performance of the sort algorithm.
    (see polyline offset, see: https://stackoverflow.com/questions/32772638/python-how-to-get-the-x-y-coordinates-of-a-offset-spline-from-a-x-y-list-of-poi)
    '''
    
    # Sort perpendicular Points so lines don't intersect
    if bSortPoints and distanceBetweenPoints_px < perpendicularVectorsLenght_px:
        perpendicularPoints = sortNoIntersections(equidistantLine[1:-1],perpendicularPoints)
    
    return perpendicularPoints

def initializePlot():
    if showPlotImages or savePlotImages:
        # To maximaze window
        plt.switch_backend('QT5Agg')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        
        # Make plots appear immediately
        plt.ion()

################## Script Start ##################
print("starting script...")
initializePlot()

# List of Images
imageList = glob.glob(srcFolder + fileNameSpec)

# List length
nList = len(imageList)

# List to write to CSV output file
listToCSV=[]

# Get current date to name output and figures
dateTimeStr = str(datetime.datetime.now())[0:19]
dateTimeStr = dateTimeStr.replace(':', 'h',1).replace(':','m')+'s'

# try to open file
if writeFile:
    f = open('output/numbers2' + dateTimeStr + '.csv', 'w', newline='')

# number of errors
nErrors = 0

# Loop through each image
for i in range(0,nList):
# for i in range(20,50):
# for i in range(5):
#     i = 24
    
    # for plot purpose
    if showPlotImages or savePlotImages:
        plt.clf()
    
    # Image name
    imageName = os.path.basename(imageList[i])
    
    print("Processing " + str(i+1) + " of " + str(nList) + ": " + imageName)
    
    # Open Image
    cziImage = aicsimageio.readers.CziReader(srcFolder + imageName)
    
    # Get Image as numpy array
    try:
        img = cziImage.data[0]
    except:
        print ("Unable to open: " + imageName)
        nErrors += 1
        continue
    
    # Get real pixel size and convert to micrometers
    pixelMetric = cziImage.get_physical_pixel_size(1)[0]*1e6
    
    # get Image dimensions in pixels (Image is transposed)
    imHeight,imWidth = img.shape 
    
    # Rotate image to landscape for better view
    if imHeight > imWidth:
        img = np.rot90(img)
        
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Filter image to eliminate high frequency noise
    kg = 11
    imgray = cv2.GaussianBlur(imgray, (kg, kg), 0)
    
    # Image to int np.array to compute bresenhamLine
    imgAsInt = imgray.astype(int)
    
    # get image Mask based on binary threshold
    binaryThreshold = 20
    ko = 2
    kc = 31
#     kd = 10
    kd = 0
    ke = 3
    imgMask = getThreshImage(imgray,ko,kc,kd,ke,binaryThreshold)
#     plt.imshow(imgMask)
    
    ''' TODO: Use a adapative threashold instead of a binary binary Threshold.
    This would have a better result for image with variant brightness.
    It could be computed two diferent adaptative thresholds,
    1. the first one is to get the outer border of the gut, where a Contour Approximation
    could be then be used to ignore small holes in the borders. We can then use drawline
    to print the border to the image to connect all objects in the border. then we could
    use the skeletonize only for the outer objects to get a perfect line passing in the
    middle of the outer objects (the gut wall)
    2. the second threshold should be use to get the inner contours to then compute the
    distance from the borders to them. by drawing a perpendicular braham line from the
    border to the interior of the gut, we should get the first True pixel of this line.
    
    for Contour Approximation see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    '''
#     # get image Mask based on adaptative gaussian threshold
#     blockSize = 21
#     ko = 2
#     kc = 31
#     kd = 10
#     imgAdaptMask = getThreshImage(imgray,ko,kc,kd,ke,mode='adaptative',blockSize=blockSize)
#     plt.imshow(imgAdaptMask)
    
    # Get gut thickness
    max_gut_thickness, gut_thickness = getGutThickness(imgMask)
    
    # get contours ocf borders
    contours, _ = cv2.findContours(imgMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Process contours
    contoursList = []
    for contour in contours:
        # Take in account only contours with big areas
        if cv2.contourArea(contour) > 1e5:
            
            # convert to 2D array
            contour = contour.reshape(-1,2)
            
            # Find indexes of points that are in image borders
            breakContourIndexes = np.any(
                np.c_[contour [:,0] == 0,
                contour [:,1] == 0,
                contour [:,0] == imWidth-1,
                contour [:,1] == imHeight-1],
                axis=1)
            
            # Get consecutive points on image border
            contourIndex = np.where(np.all(
                np.c_[breakContourIndexes[1:],breakContourIndexes[:-1]],
                axis=1))[0]
            
            # Rearrange contour to start and finish at image border
            contour = np.r_[contour[contourIndex[0]+1:],contour[:contourIndex[0]+1]]
            
            # Add contour to list
            contoursList.append(contour)
    
    # merge all contours in single 2D array
    contours = np.array(contoursList).reshape(-1,2)
    
    ## Split Contours
    # Find indexes of points that are in image borders
    breakContourIndexes = np.any(
        np.c_[contours [:,0] == 0,
        contours [:,1] == 0,
        contours [:,0] == imWidth-1,
        contours [:,1] == imHeight-1],
        axis=1)
    
    # find indexes to split
    contourIndexes = np.where(breakContourIndexes)[0]
    indexesToSplit = np.c_[contourIndexes,contourIndexes+1].flatten()
    
    # split contours into lines when contour touches image border
    contourLines = np.split(contours,indexesToSplit)
    
    # Sort contours by reverse order of length
    contourLines.sort(key=len, reverse=True)
    
    # matrix that performs a rotation of 90 degrees clockwise
    # as image has y coordinates inverted (top is 0 and bottom is max)
    # this rotation matrix seems reflected
    rotationMatrix = np.array([[0, -1],[1, 0]])
    
    # Initialize lists
    pointSList=[]
    pointEList=[]
    
    # Loop through each outerLine
    nContours = 0
    for outerLine in contourLines:
        
        # ignore short lines
        if len(outerLine) < 50:
            continue
        
        # Increment number of contours
        nContours += 1
        
        # only allow two biggest contours
        if nContours > 2:
            break
        
        # Get equidistant border Points/tangent Points
        equidistantLine = getEquidistantPoints(outerLine)
        
        # Get perpendicular Points
        perpendicularPoints = getPerpendicularPoints(equidistantLine)
        
        # Initialize lists
        pointSPeakList=[]
        pointEPeakList=[]
    
        # loop through each point to find the distance to the nearest contours
        for p in range(perpendicularPoints.shape[0]):
            
            # Get image indexes using bresenham line from 
            # point equidistantLine[p+1] to point perpendicularPoints[p]
            bresenhamLine = np.array(skimage.draw.line(round(
                equidistantLine[p+1,0]),round(equidistantLine[p+1,1]),
                round(perpendicularPoints[p,0]),round(perpendicularPoints[p,1]))).T
            
            # remove indexes outside image
            bresenhamLine = bresenhamLine[np.where((bresenhamLine[:,0]>=0)*(bresenhamLine[:,1]>=0)*
                    (bresenhamLine[:,0]<imWidth)*(bresenhamLine[:,1]<imHeight))]
            
            # get line pixels from image
            imageValues = imgAsInt[bresenhamLine[:,1],bresenhamLine[:,0]]
            
            # Low pass filter to smooth the values
            imageValues = scipy.ndimage.filters.uniform_filter1d(imageValues,2)
#             imageValues[imageValues<20]=0
            
            # get local maxima
            peakIndexes = scipy.signal.find_peaks(imageValues)[0]
            
            # Ignore point if there are no peaks
            if len(peakIndexes) < 2:# or peakIndexes[0]>15:
#                 if len(peakIndexes) == 1:
#                     pointF = bresenhamLine[peakIndexes[0]]
#                     pointFList.append(pointF)
                continue
            
            # Get Start Point and End Point
#             pointS = bresenhamLine[peakIndexes[0]]
            pointS = equidistantLine[p+1]
            if peakIndexes[0] > 35:
                pointE = bresenhamLine[peakIndexes[0]]
            else:
                pointE = bresenhamLine[peakIndexes[1]]
            
            # Add Points to Lists
            pointSPeakList.append(pointS)
            pointEPeakList.append(pointE)
        
        # Convert Points List to Array
        pointSPeakList = np.array(pointSPeakList)
        pointEPeakList = np.array(pointEPeakList)
        
        # Line to connect border points and respect perpendicular points 
        allLines = np.c_[pointSPeakList,pointEPeakList,pointSPeakList].reshape(-1,2)
        
#         plt.imshow(img)
#         plt.plot(pointSPeakList[:,0],pointSPeakList[:,1],'b.')
#         plt.plot(pointEPeakList[:,0],pointEPeakList[:,1],'r.')
#         plt.plot(allLines[:,0],allLines[:,1])
        
        # Get distances
        cell_thickness_array = np.linalg.norm(pointEPeakList-pointSPeakList,axis=1)
        
        # Remove outliers
        cell_thickness_array = scipy.signal.medfilt(cell_thickness_array)
        
        # Low Pass Filter (Smooth values)
        cell_thickness_array = scipy.ndimage.filters.uniform_filter1d(cell_thickness_array,2)
        
        # Get peaks 
        ind = scipy.signal.find_peaks(cell_thickness_array,distance = np.ceil(25/distanceBetweenPoints_px))[0]
        
        # Get Peak Points
        pointSPeak = pointSPeakList[ind]
        pointEPeak = pointEPeakList[ind]
        
        # Store peak Points
        pointSList.append(pointSPeak)
        pointEList.append(pointEPeak)

    # ignore if no points found
    if len(pointSList) == 0:
        continue
    
    # Convert Points List to 2D Array
    pointSList = np.vstack(pointSList)
    pointEList = np.vstack(pointEList)
#     pointFList = np.array(pointFList)
        
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
        plt.savefig(plotFolder + imageName + '_' + dateTimeStr + '_plot.png',dpi=200)
    
    # compute cell thickness (distance between points S and E
    cell_thickness = np.linalg.norm(pointSList-pointEList,axis=1)
    
    ### Add values to row list (to populate csv file)
    rowToCSV = imageName[:-len(fileNameSpec)+1].split("_")
    
    # Add cell_thickness
    rowToCSV.append(np.round(np.mean(cell_thickness)*pixelMetric,4))
    rowToCSV.append(np.round(np.median(cell_thickness)*pixelMetric,4))
    
    # Add gut_thickness
    if bGet_gut_thickness:
        rowToCSV.append(np.round(max_gut_thickness*pixelMetric,4))
        rowToCSV.append(np.round(np.mean(gut_thickness)*pixelMetric,4))
        rowToCSV.append(np.round(np.median(gut_thickness)*pixelMetric,4))
    
    # add row line to list to write in CSV file
    listToCSV.append(rowToCSV)

# Write values to csv file
if writeFile:
    with f:
        
        # open writer with semicolon delimiter
        writer = csv.writer(f)
#         writer = csv.writer(f,delimiter=";")
        
        # Write Header
        writer.writerow(['Line','treat','td','gut','cell mean (um)','cell median (um)',\
            'gut thickness max (um)','gut thickness mean (um)', 'gut thickness median (um)'])
        
        for row in listToCSV:
            writer.writerow(row)

if nErrors > 0:
    msgBox("Finished.\n\nWarning! Some images were not processed: " + str(nErrors))
else:
    msgBox("Finished.")
