#####base pour analyse image courbes ########

import pandas as pd
import os
import re
import json
import imageio
import numpy as np

import imageio
#import numpy
from matplotlib.pyplot import *
import sys
import copy

print('the current path is')
print(os.getcwd())

# data = '../201009_Ines_pictures/datafinal.json'
data = './output/datafinal.json'

with open(data,'r') as file:
	ddd = json.load(file)

#boucle for pour compter les tableaux

# Get List of Images and loop through each Image  
ID = [i for i in ddd]
lst3 = []
for i in ID:
	# convert items to dictionary for this image
	dct = {ki:pd.DataFrame(vi) for ki,vi in ddd[i].items()}
	
	# Filter out points with low distances
	dct = {ki:vi[vi['Distance']>10] for ki,vi in dct.items()}

	# loop through each distance and get max value bis
	dct2 = []
	for ki,vi in dct.items():
		vi.astype({"valuebis": float})
		max = vi.valuebis.max()
		vi = vi[vi['valuebis']==max].drop_duplicates()
		dct2 += [(ki,vi)]

	lst3 +=[(i,dict(dct2))]

dct3 = dict(lst3)

# I can't understand what this does
dct4 = {i:{ki:vi.to_dict('list') for ki,vi in dct3[i].items()}for i in dct3}

with open('./output/tab_filtered.json','a') as filetowrite:
        json.dump(dct4,filetowrite,indent = 4)

# Loop through images again and get lists depending of number of points 
lst = []
for i in ID:
	# list where 2 points with maximum where found 
	t2 = {ki:vi for ki,vi in dct3[i].items() if len(vi['x']) == 2}
	
	# list where 3 points with maximum where found
	t3 = {ki:vi for ki,vi in dct3[i].items() if len(vi['x']) == 3}
	
	# list where 4 points with maximum where found
	t4 = {ki:vi for ki,vi in dct3[i].items() if len(vi['x']) == 4}
	
	# list where more than 4 points with maximum where found
	tautre =  {ki:vi for ki,vi in dct3[i].items() if len(vi['x']) >4}

	lst += [(i,{'2':len(t2),
	'3':len(t3),
	'4':len(t4),
	'>4':len(tautre),
	'tot':len(dct3[i])})]

# list of multiple maximum points 
dct = dict(lst)

outFilename = './output/TabCounts_filtered.csv'
pd.DataFrame.from_dict(dct).T.to_csv(outFilename)
print("Created file " + outFilename)