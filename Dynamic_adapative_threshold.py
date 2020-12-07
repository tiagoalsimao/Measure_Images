"""
 * Python script to demonstrate Canny edge detection.
 *
 * usage: python CannyEdge.py <filename> <sigma> <low_threshol> <high_threshold>
"""
from distutils.command.install import install

#install(PyQt5)
import skimage
import skimage.feature
import skimage.viewer
import sys
import cv2
from skimage.filters import threshold_local
import os
import glob
from aicsimageio.readers import CziReader

from matplotlib.pyplot import *
# Get list of files 
filePath = "./data/20201125_CG1139KO/"
fileNameLike = "*.czi"
cziImageList = glob.glob(filePath + fileNameLike)

cziImage = cziImageList[0]
cziImageFileName = os.path.basename(cziImage)

reader = CziReader(cziImage).data[0]

# read command-line arguments
# filename = sys.argv[1]
# sigma = float(sys.argv[2])
# low_threshol = float(sys.argv[3])
# high_threshold = float(sys.argv[4])

# filename = "C:/Users/tiago/eclipse-workspace/Measure_Images/data/07-junk.jpg"
# filename = "C:/Users/tiago/eclipse-workspace/Measure_Images/data/20201125_CG1139KO/9024KO_db11_16h_1.czi"
# filename = "C:/Users/tiago/eclipse-workspace/Measure_Images/data/A5001_Db11_4-1.tif"

imageName = "9024KO_db11_16h_3.czi.tiff"
srcFolder = './data/20201125_CG1139KO/tif/'

filename = srcFolder + imageName

sigma = 10.0
low_threshol = 1.1 
high_threshold = 10.3

# load and display original image as grayscale
image = skimage.io.imread(fname=filename, as_gray=True)
# viewer = skimage.viewer.ImageViewer(image=image)
# viewer.show()

edges = skimage.feature.canny(
    image=image,
    sigma=sigma,
    low_threshold=low_threshol,
    high_threshold=high_threshold,
)


# edges_reader = skimage.feature.canny(
#     image=reader,
#     sigma=sigma,
#     low_threshold=low_threshol,
#     high_threshold=high_threshold,
# )

# fig, ax = try_all_threshold(image, figsize=(10, 8), verbose=False)
# show()


binary_image1 = 1*(image < threshold_local(255-image, block_size=31, method='gaussian',offset=2))

# binary_image1 = threshold_adaptive(image, block_size, method='gaussian', offset=2, mode='reflect', param=None)

# blockSize = 61
# binary_image1 = cv2.adaptiveThreshold(255-image,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize,2)

# display edges
viewer = skimage.viewer.ImageViewer(binary_image1)

# Create the plugin and give it a name
# canny_plugin = skimage.viewer.plugins.Plugin(image_filter=skimage.filters.threshold_local)
# canny_plugin = skimage.viewer.plugins.Plugin(image_filter=skimage.filters.threshold_adaptive)
canny_plugin.name = "Local threshold"

# Add sliders for the parameters
canny_plugin += skimage.viewer.widgets.Slider(
    name="block_size", low=3, high=101, value=15
)

canny_plugin += skimage.viewer.widgets.Slider(
    name="offset", low=0, high=51, value=1
)

# add the plugin to the viewer and show the window
viewer += canny_plugin
viewer.show()
# 
# # display edges
# viewer = skimage.viewer.ImageViewer(edges)
# # viewer.show()
# 
# # Create the plugin and give it a name
# canny_plugin = skimage.viewer.plugins.Plugin(image_filter=skimage.feature.canny)
# canny_plugin.name = "Canny Filter Plugin"
# 
# # Add sliders for the parameters
# canny_plugin += skimage.viewer.widgets.Slider(
#     name="sigma", low=0.0, high=7.0, value=2.0
# )
# canny_plugin += skimage.viewer.widgets.Slider(
#     name="low_threshold", low=0.0, high=1.0, value=0.1
# )
# canny_plugin += skimage.viewer.widgets.Slider(
#     name="high_threshold", low=0.0, high=1.0, value=0.2
# )
# 
# # add the plugin to the viewer and show the window
# viewer += canny_plugin
# viewer.show()

input("Press Enter to continue...")