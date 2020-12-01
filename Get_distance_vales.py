#####base pour analyse image courbes ########

import pandas as pd
import os
import re
import json
import imageio
import numpy as np

import imageio
from matplotlib.pyplot import *
import sys
import copy

#data
# data = '../table_filtered.json'
data = './output/tab_filtered.json'

# Open jason File
with open(data,'r') as file:
	ddd = json.load(file)

# Loop trhough each image
test = []
for image in ddd:
	data = ddd[image]
	test += [(image,[table['Distance'][0] for y,table in data.items() if len(table['Distance']) != 0])]

dtest = dict(test)

tabletosave = pd.DataFrame.from_dict(dtest,orient = 'index').T

#changer nom de colonnes pour faciliter Rstudio analysis

pat = re.compile(r'db11_3h')

#re.sub(pat,'db11-3h','np_iso_db11_3hsuc12h_3-1.tif')

tabletosave.columns = [re.sub(pat,'db11-3h',i) for i in tabletosave.columns]
test = []
for i in tabletosave.columns:
        pat2 = re.compile(r'.*_\d{1,2}-\d.tif')
        if re.match(pat2, i):
                test +=[re.sub('-\d{1,2}.tif','_B',i)]
        else:
                test +=[re.sub('.tif','_A',i)]

tabletosave.columns = test
tabletosave.to_csv('./output/tableofvaluesforRstudio.csv')
