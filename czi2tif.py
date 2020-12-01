import glob
import os
import cv2
from aicsimageio.readers import CziReader
import matplotlib.pyplot as plt



# Input variables
srcFolder = "./data/20201125_CG1139KO/"
fileNameSpec = "*.czi"

# Destination folder to save images as tif
destFolder = srcFolder + "tif/"

# List of Images
cziImageList = glob.glob(srcFolder + fileNameSpec)

# List length
nList = len(cziImageList)

# Loop through each image
for i in range(0,nList):

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
        img = cziImage.data[0]
        
        # get Image dimensions in pixels (Image is transposed)
        imHeight,imWidth = img.shape 
        
        # Plot image
        plt.imshow(img,cmap='gray', vmin=0, vmax=255,
                   extent=[0,xPixelSize*imWidth,0,yPixelSize*imHeight])
        
        # Save Image in destFolder
        cv2.imwrite(destFolder + cziImageFileName + ".tiff", img)
    except:
        print ("unable to save " + cziImageFullName)
        
        