""" Ideas
- use image processing package tools
- resize image if necessary to improve runtime performance, if it doesn't affect results 
- use edges functions to get points (or splines if possible) based on threshold
- get points for each spline (could be tricky, or already done by edges function) this should be a spline 
- get points in lines to be equaly distributed with interparc function (could be less)
    source: https://stackoverflow.com/questions/18244305/how-to-redistribute-points-evenly-over-a-curve
- get derivatives (tangents) for each point
- get intersection of 2 lines L1 and L2
    source: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines


more ideas after verifying how to compute tangents perpendicular
"""

# Creation de modules qui vont devoir s'imbriquer pour former une fonction ou
# une classe bien particuliere.
# elements necessaires pour l'analyse d'image

# ####base pour analyse image courbes ########

# import re
import json
import os

import imageio
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

import time

import glob

# import sys
# import copy
import matplotlib.pyplot as plt
# from matplotlib.pyplot import *


# ###def######
# Function tangent_point returns a range of 10 points in image looking from
# left to right.
# It returns the coordinates of the first point point found.
#
# this function can be optimized!
# a function getPoint would be better and should return a list of points
# Another function getTangentPoint would then get the list of points to compute de tangent
#
# imageThreshold = 20 shoud be global variable to optimize later
# use of im instead of image unclear
# use of Y-5 near limit of image (top and bottom) should be avoided, as
#   python doesn't return error and
def tangent_point_old(image, Y):
    try:
        # Create range of Y from Y-5 to Y+6
        #r = range(max(0,Y - 5), min(Y + 6,im.shape[0]-1))
        r = range(Y - 5, Y + 6)
        l_tangeante = []

        for i in r:
            im_int = im[i]
            lint = []
            for k, j in enumerate(im_int):
                # k = x position

                if j > 20:
                    lint += [k]
            l_tangeante += [(i, lint[0])]

        return l_tangeante
    except:
        return 'error'

# new tangent_point function
def tangent_point(image, y):
    # ignore values of y in the limits of the image
    if y < 5 or y > image.shape[0]-6:
        return 'error'

    # Create range of Y from Y-5 to Y+6
    #r = range(max(0,Y - 5), min(Y + 6,im.shape[0]-1))
    r = range(y - 5, y + 6)
    l_tangeante = []

    for i in r:
        im_int = image[i]
        lint = []
        # get first point with values above threshold and breaks out of loop
        for k, j in enumerate(im_int):
            # k = x position
    
            if j > 20:
                lint += [k]
                break
        if not lint:
            return 'error'

        l_tangeante += [(i, lint[0])]

    return l_tangeante

# function to calculate the coeficients of the derivative function y = a*x+b to
# reprecent the tangent
# lst = l_tangeante
def coef_tangent(lst):
    # coordonnees des extremite de la tangente
    y1 = lst[0][0]
    x1 = lst[0][1]
    y2 = lst[-1][0]
    x2 = lst[-1][1]

    # coefficients de la tangeante
    if x2 != x1:
        a = (y2 - y1) / (x2 - x1)
    else:
        a = (y2 - y1) / (x2 - x1 + 1) # approximation acceptable for x2 and x1 integers

    b = y1 - a * x1
    return a, b

# Function to get the perpendicular function coeficients of the original 
# tangeant function
# First get midle index y, then compute perpendicular line
# a et b sont les coefficients directeurs de la tangeante
# pente = -1/a (pente du perpendicular = -1/pente tangent)
# lst = l_tangeante
# shape = im.shape
def coef_perpendicular(lst, pente):
    pente = -1 / pente  # a' = -1/a'
    # interception entre tangente et perpendicular
    # coordonnees connues du point d'intersection Y
    # this will always be the middle point, so no need to iterate
    intersect = [i for i in lst if i[0] == Y][0]
    # y = a'x +b'
    # b' = y - a'x avec y = intersect[0] et x = intersect[1]
    origine = intersect[0] - pente * intersect[1]  # b'

    return pente, origine

# This function does what?
# pente du perpendiculaire
# origine du perpendiculaire

def perpendicular_points(shape, pente, origine):
    # Create line point based on perpendicular function for entire image
    l_perp = []
    for x in range(shape[1]):
        y = (int) (pente * x + origine)
#         y = round(y, 0)
        l_perp += [(y, x)]
        
#     # convert to integers (not necessary)
#     l_perp = [(int(i), int(j)) for i, j in l_perp]  # droite
    
    # Remove points that are not in image (not necessary)
    # que les points qui sont dans l'image
    l_perp = [i for i in l_perp if i[0] in range(shape[0])]
    
    # Remove points far from Y (the current image index)
    # qu'un segment pour ne pas aller trop loin
    # sens du segemnt est fonction du signe de la pente
    if pente > 0:
        l_perp = [i for i in l_perp if i[0] in range(Y - 10, Y + 130)]
    else:
        l_perp = [i for i in l_perp if i[0] in range(Y - 130, Y + 10)]

    return l_perp

# Function to get values of images for the points in the perpendicular line
# and returns a table of x,y coordinates, the value, and if it is 
# or not above threshold 
# donne coordonnees du segment le long duquel on va mesurer l'intensité de fluo

# im = im
# lst = l_perp = Points of Line Perpendicular to tangent
# threshold = 150 by default
def fluo_values(im, lst, threshold=150):
    hist = []
    hist2 = []
    
    # get values of image for points at perpendicular line  
    for i, j in lst:
        val = im[i][j]
        hist += [val]
        hist2 += [(i, j, val)]

    tab_of_vals = pd.DataFrame(hist2, columns=['x', 'y', 'value'])
    valuebis = [i if i > threshold else 0 for i in tab_of_vals['value']]
    tab_of_vals['valuebis'] = valuebis

    return tab_of_vals


# Function to find filter the tab lines which values that are not above threshold  
# tab = tab_of_vals
def pic_finders(tab):
    peaks, _ = find_peaks(tab.valuebis)

    tab_of_peaks = tab.iloc[peaks]

    return tab_of_peaks

# Function that gets the distances between the first point in the Perpendicular
# line and the following points (I'm not sure if it should be the first because of 
# the perpendicular_points function that uses the code > range(Y - 10, Y + 130)
# tab = tab_of_peaks
def pic_distances(tab):
    if len(tab) >= 2:
        distance = []
        premierpic = tab.iloc[0]
        for i in range(len(tab)):
            autre = tab.iloc[i] # first autre == premierpic and d = 0 always 
            # pythagoras
            d = np.sqrt(np.square(premierpic.x - autre.x) + np.square(premierpic.y - autre.y))
            distance += [d]
        tab['Distance'] = distance
    else:
        tab['Distance'] = [0]

    return tab

# ### real for loop
# Start of script execution

# Check current directory
print('the current path is')
print(os.getcwd())

# Change current directory to filepath
#filepath = '../201009_Ines_pictures/20201009/totest2/'
filepath = './data/'
fileExtention = '.tif'

os.chdir(filepath)
print('the current path is')
print(os.getcwd())

# get list of files in directory
lstofPic = os.listdir()
print(''.join([str(len(lstofPic)), ' pictures are presents in this directory']))

# loop through each .tif image in folder
start_time2 = time.time()

d = []
for image in lstofPic:

    # Checks if image is of type .tif
    if not image.endswith(fileExtention) or image.startswith('.'):
        continue
    # image filename
    print(image)
    im = imageio.imread(image)  # entree de la fonction ou de la classe

    # image dimensions
    print(im.shape)
    shape = im.shape

    # Create list of vertical indexes (y) incremented by 2 pixels
    # this means that are analising the image by intervals of 2 pixels
    # numPixels 2 shoud be global variable to optimize later
    l_Y_of_i = [i for i in range(shape[0]) if i % 2 == 0]
    #l_Y_of_i = list(range(0, shape[0], 2))

    # Remove y indexes of image that dont have information // enleve les Y qui ne donneront pas de tangentes
    print('Remove y indexes of image that dont have information') #print('we drop off shitty points')

    # Loop through each y index verify if it has points
    # if not, remove point from list. this loop shouldn't be necessary.
    start_time5 = time.time()  # 75 seconds per image
    Y_to_drop = []
    for i in l_Y_of_i:
        tan_lst = tangent_point(im, i)

        # Remove lines that didn't find any points
        if tan_lst == 'error':
            Y_to_drop += [i]

    print("start_time5 %s seconds" % (time.time() - start_time5))

    # Remove y indexes that returned error
    l_Y_of_i = [i for i in l_Y_of_i if i not in Y_to_drop]

    print('data is processing...')
    start_time7 = time.time()  # 140 seconds per image

    # Loop through each y index (again) to perform the magic
    dct_test = []
    for i in l_Y_of_i:
        Y = i # glogal variable
        # print(i)

        # Get points (again! +75 seconds of performance that has already been done)
        tan_lst = tangent_point(im, i)

        a_tan, b_tan = coef_tangent(tan_lst)

        a_per, b_per = coef_perpendicular(tan_lst, a_tan)

        per_lst = perpendicular_points(shape, a_per, b_per)

        tab = fluo_values(im, per_lst, threshold=150)

        tab = pic_finders(tab)
        tab = pic_distances(tab)
        
        # Filter out points with distances over 130
        tab = tab[tab['Distance'] < 130]
        dct_test += [(i, tab)]

    print("start_time7 %s seconds" % (time.time() - start_time7))

    print('fini')

    # Create dictionary of dct_test
    dct_test = dict(dct_test)
    
    # Filter out values in dictionary that don't have info
    dct_test = {ki: vi for ki, vi in dct_test.items() if len(vi) > 1}

    # convert back to list and append
    d += [(image, {ki: vi.to_dict('list') for ki, vi in dct_test.items()})]


print("start_time2 %s seconds" % (time.time() - start_time2)) # 215 seconds total per image

# convert to dictionary
d = dict(d)

# Write data d into file
with open('../output/datafinal.json', 'a') as filetowrite:
    json.dump(d, filetowrite, indent=4)
