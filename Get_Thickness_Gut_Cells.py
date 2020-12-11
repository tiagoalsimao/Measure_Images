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

# Write values to CSV
writeFile = False

# Save each image plot one at a time
showPlotImages = True

# Save image plot (Takes longer)
savePlotImages = False

# Destination folder to save images as tif
plotFolder = './data/20201125_CG1139KO/plt.plot/'

#######################################################

def msgBox(text):  
#     top = Tk()
#     top.geometry("100x100")
    messagebox.showinfo('Information',text)
#     top.mainloop()

# get equidistante points (check if it is possible to use pchip to smooths changes)
def getEquidistantPoints(outerLine):
    x,y = outerLine.T
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd**2+yd**2)
    u = np.cumsum(dist)
    u = np.hstack([[0],u])
    
    dist_array = (x[:-1]-x[1:])**2 + (y[:-1]-y[1:])**2

    lineLength = np.sum(np.sqrt(np.sum(np.diff(outerLine,axis=0)**2,axis=1)))

    # create points with distances of 10 pixels
    t = np.linspace(0,u.max(),(lineLength/10).astype(int))
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    
    return xn, yn

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


if showPlotImages:
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
# for i in range(0,nList):
for i in range(5):
    
    plt.clf()
    
    # Image name
    imageName = os.path.basename(imageList[i])
    
    print("Processing " + str(i) + " of " + str(nList) + ": " + imageName)
    
    img = cv2.imread(srcFolder + imageName,0)
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
    
    # Image to int np.array
    imgAsInt = imgray.astype(int)
    
#     plt.imshow(imgMask)
    
    # get contours of borders
    contours, _ = cv2.findContours(imgMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # TODO: remove small noise with contours areas and contours fill

    # get Contour Points
    contoursList = []
    for contour in contours:
        if cv2.contourArea(contour) > 1e5:
            contoursList.append(contour)
            # TODO: Get gut thickness
    
    contours = np.array(contoursList).reshape(-1,2)
    
    # Close contour
#     contours = contours+contours[0,:]
    
    ## Split Contours into 2 lines
    breakContourIndexes = np.any(
        np.c_[contours [:,0] == 0,
        contours [:,1] == 0,
        contours [:,0] == img.shape[1]-1,
        contours [:,1] == img.shape[0]-1],
        axis=1)
    
    contourIndexes = np.where(breakContourIndexes)
    
    indexesToSplit = np.c_[contourIndexes[0],contourIndexes[0]+1].flatten()
    
    contourLines = np.split(contours,indexesToSplit)

    # matrix that performs a rotation of 90 degrees clockwise
    rotationMatrix = np.array([[0, -1],[1, 0]])
    
    # Initialize lists
    pointSList=[]
    pointEList=[]
    pointFList=[]
    contourIgnored = []
    
    # Loop through each Line
    for outerLine in contourLines:
        
        # ignore short lines
        if len(outerLine) < 50:
            contourIgnored.append(outerLine)
            continue
        
        # get equidistant points
        x,y = getEquidistantPoints(outerLine)
        
        # TODO: use pchip to make contours and respective tangents smoother
        # x,y = ppval(ppchip(x,y))
        
        # Border Points/tangent as an np.array of [x,y]
        equidistantLine = np.c_[x, y]
        
        # Get tanjentVectors with are the diference of the adjacent points
        # i.e. if points are [P1, P2, P3, ... , Pn] then the tangent vectors
        # will get [P3-P1, P4-P2, ..., Pn-P(n-2)]
        tanjentVectors = equidistantLine[2:] - equidistantLine[:-2]
        
        # Get perpendicular vector by multiplying with rotation matrix
        perpendicularVectors = np.matmul(tanjentVectors,rotationMatrix)
        
        # normalize vectors
        vectorsNorm = np.linalg.norm(perpendicularVectors,axis=1)
        perpendicularVectors = perpendicularVectors/np.c_[vectorsNorm,vectorsNorm]
        
        # get perpendicular points by adding perpendicularVectors to Border Points
        perpendicularPoints = perpendicularVectors*200 + equidistantLine[1:-1]
        
        # TODO: sort or remove perpendicularPoints if they intersect (remove points that intersect the most lines
        
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
            
#             plt.plot(pointS[0],pointS[1],'r.')
#             plt.plot(pointE[0],pointE[1],'b.')
            
            # Add Points to Lists
            pointSList.append(pointS)
            pointEList.append(pointE)
    
    if showPlotImages or savePlotImages:
            
        # convert Points List to Array to be ploted
        pointSList = np.array(pointSList)
        pointEList = np.array(pointEList)
        pointFList = np.array(pointFList)
        allLines = np.c_[pointSList,pointEList,pointSList].reshape(-1,2)
        
        # Plot image and points
        fig = plt.imshow(img)
        plt.plot(pointSList[:,0],pointSList[:,1],'r.')
        plt.plot(pointEList[:,0],pointEList[:,1],'b.')
    #     plt.plot(pointFList[:,0],pointFList[:,1],'g.')
        plt.plot(allLines[:,0],allLines[:,1])
    
    # Save Images as png file in plotFolder
    if savePlotImages:
        plt.savefig(plotFolder + imageName + '.png',dpi=fig.dpi)
    
    # compute cell thickness
    cell_thickness = np.sqrt(np.sum((pointSList-pointEList)**2,axis=1))
    
    # add values to row list 
    rowToCSV = imageName[:-9].split("_")
    rowToCSV.append(np.round(np.mean(cell_thickness),4))
    rowToCSV.append(np.round(np.median(cell_thickness),4))
    
    # add row line to list to write in CSV file
    listToCSV.append(rowToCSV)
    
    if showPlotImages:
        msgBox("Click OK for next figure")

# Write values to csv file
if writeFile:
    with f:
    
        writer = csv.writer(f)
        # Write Header
        writer.writerow(['Line','treat','td','gut','cell mean (um)','cell median (um)', 'gut thickness (um)'])
        
        for row in listToCSV:
            writer.writerow(row)

msgBox("Finished.")
