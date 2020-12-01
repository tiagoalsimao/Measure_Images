import glob
import os
import cv2
from aicsimageio.readers import CziReader
import matplotlib.pyplot as plt

import imageio
import skimage
import skimage.feature
import skimage.viewer

import numpy as np
from numpy import *
from matplotlib.pyplot import *

# Input variables
srcFolder = "./data/20201125_CG1139KO/"
fileNameSpec = "*.czi"

threshholdBlackWhite = 30

# Destination folder to save images as tif
destFolder = srcFolder + "tif/"

# List of Images
cziImageList = glob.glob(srcFolder + fileNameSpec)

# List length
nList = len(cziImageList)

# Loop through each image
for i in range(0,1+0*nList):

    # Full path name of image
    cziImageFullName = cziImageList[i]
    
    # Image name
    cziImageFileName = os.path.basename(cziImageFullName)
    
    # Print current image to open
    toPrint = str(i+1) + " of " + str(nList) + ": " + str(cziImageFileName)
    print("Opening " + toPrint)
    
    # Try opening image
    try:
        # Open Image
        cziImage = CziReader(cziImageFullName)
                
        # get real pixel size in meters
        xPixelSizeMeters,yPixelSizeMeters,junk = cziImage.get_physical_pixel_size(1)
        
        # Convert pixel size to micrometers
        xPixelSize = xPixelSizeMeters*1e6
        yPixelSize = yPixelSizeMeters*1e6
        
        # Get Image as numpy array
        image = cziImage.data[0]
        
        # get Image dimensions in pixels (Image is transposed)
        imHeight,imWidth = image.shape
        
        # Correct image by changing (min,max) values to (0,255)
        image = np.interp(image, (np.min(image), np.max(image)), (0, 255))
        
        # Plot image
        plt.subplot(121)
#         plt.imshow(image,cmap='gray', vmin=0, vmax=255,
#                    extent=[0,xPixelSize*imWidth,0,yPixelSize*imHeight])
        
    except:
        print ("unable to open " + cziImageFullName)

    
    
    
    # correct image brightness
    ratio = 0.5
    im = cv2.convertScaleAbs(image, alpha = 1 / ratio, beta = 0)
    
    # correct image brightness source: https://stackoverflow.com/questions/57030125/automatically-adjusting-brightness-of-image-with-opencv
    brightness = np.sum(image) / (255 * imHeight * imWidth)
    minimum_brightness = 0.4
    ratio = brightness / minimum_brightness
    image = cv2.convertScaleAbs(image, alpha = 1 / ratio, beta = 0)

    plt.subplot(122)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    plt.imshow(imageRGB)
#     plt.imshow(imageRGB,cmap='gray')
    
    # get image mask (0 and 1) using threshold
    ret,imMask = cv2.threshold(image,threshholdBlackWhite,1,cv2.THRESH_BINARY)
    
#     plt.imshow(imMask,cmap='gray')
    
    # automatic threshold
    blur = cv2.GaussianBlur(image,(5,5),0) # gaussian filter with 5 by 5 matrix
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#     plt.imshow(th3,cmap='gray')
    
    # closing  source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    kernel = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(imMask, cv2.MORPH_CLOSE, kernel)

    
    # countour
    contours, hierarchy = cv2.findContours(imMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
# Find Canny edges 
edged = cv2.Canny(image, 30, 200) 
# cv2.waitKey(0) 
  
# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged,  
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
  
cv2.imshow('Canny Edges After Contouring', edged) 
# cv2.waitKey(0) 
  
print("Number of Contours found = " + str(len(contours))) 
  
# Draw all contours 
# -1 signifies drawing all contours 
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
  
cv2.imshow('Contours', image) 
# cv2.waitKey(0) 
# cv2.destroyAllWindows()


    
    
sigma = 10.0
low_threshold = 1.1 
high_threshold = 10.3

edges = skimage.feature.canny(
    image=img,
    sigma=sigma,
    low_threshold=low_threshold,
    high_threshold=high_threshold,
)

viewer = skimage.viewer.ImageViewer(edges)
viewer.show()

# Create the plugin and give it a name
canny_plugin = skimage.viewer.plugins.Plugin(image_filter=skimage.feature.canny)
canny_plugin.name = "Canny Filter Plugin"

# Add sliders for the parameters
canny_plugin += skimage.viewer.widgets.Slider(
    name="sigma", low=0.0, high=7.0, value=2.0
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="low_threshold", low=0.0, high=1.0, value=0.1
)
canny_plugin += skimage.viewer.widgets.Slider(
    name="high_threshold", low=0.0, high=1.0, value=0.2
)

# add the plugin to the viewer and show the window
viewer += canny_plugin
viewer.show()

input("Press Enter to continue...")