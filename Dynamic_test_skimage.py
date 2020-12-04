"""
 * Python script to demonstrate Canny edge detection.
 *
 * usage: python CannyEdge.py <filename> <sigma> <low_threshold> <high_threshold>
"""
from distutils.command.install import install

#install(PyQt5)
import skimage
import skimage.feature
import skimage.viewer
import sys

import os
import glob
from aicsimageio.readers import CziReader


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
# low_threshold = float(sys.argv[3])
# high_threshold = float(sys.argv[4])

# filename = "C:/Users/tiago/eclipse-workspace/Measure_Images/data/07-junk.jpg"
# filename = "C:/Users/tiago/eclipse-workspace/Measure_Images/data/20201125_CG1139KO/9024KO_db11_16h_1.czi"
filename = "C:/Users/tiago/eclipse-workspace/Measure_Images/data/A5001_Db11_4-1.tif"
sigma = 10.0
low_threshold = 1.1 
high_threshold = 10.3

# load and display original image as grayscale
image = skimage.io.imread(fname=filename, as_gray=True)
# viewer = skimage.viewer.ImageViewer(image=image)
# viewer.show()

edges = skimage.feature.canny(
    image=image,
    sigma=sigma,
    low_threshold=low_threshold,
    high_threshold=high_threshold,
)

edges_reader = skimage.feature.canny(
    image=reader,
    sigma=sigma,
    low_threshold=low_threshold,
    high_threshold=high_threshold,
)

# display edges
viewer = skimage.viewer.ImageViewer(edges)
# viewer.show()

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