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

data = '../201009_Ines_pictures/datafinal.json'
data = './datafinal.json'

with open(data,'r') as file:
	ddd = json.load(file)

#boucle for pour compter les tableaux

ID = [i for i in ddd]

lst3 = []
for i in ID:
	dct = {ki:pd.DataFrame(vi) for ki,vi in ddd[i].items()}
	dct = {ki:vi[vi['Distance']>10] for ki,vi in dct.items()}

	dct2 = []
	for ki,vi in dct.items():
		vi.astype({"valuebis": float})
		max = vi.valuebis.max()
		vi = vi[vi['valuebis']==max].drop_duplicates()
		dct2 += [(ki,vi)]

	lst3 +=[(i,dict(dct2))]

dct3 = dict(lst3)

dct4 = {i:{ki:vi.to_dict('list') for ki,vi in dct3[i].items()}for i in dct3}

with open('../tab_filtered.json','a') as filetowrite:
        json.dump(dct4,filetowrite,indent = 4)

lst = []
for i in ID:
	t2 = {ki:vi for ki,vi in dct3[i].items() if len(vi['x']) == 2}
	t3 = {ki:vi for ki,vi in dct3[i].items() if len(vi['x']) == 3}
	t4 = {ki:vi for ki,vi in dct3[i].items() if len(vi['x']) == 4}
	tautre =  {ki:vi for ki,vi in dct3[i].items() if len(vi['x']) >4}

	lst += [(i,{'2':len(t2),
	'3':len(t3),
	'4':len(t4),
	'>4':len(tautre),
	'tot':len(dct3[i])})]

dct = dict(lst)

pd.DataFrame.from_dict(dct).T.to_csv('../TabCounts_filtered.csv')
print("Created file " + '../TabCounts_filtered.csv')