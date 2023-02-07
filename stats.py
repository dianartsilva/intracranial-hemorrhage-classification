# SMALL DATASET
# list all files in png dataset - original  

from os import listdir
from os.path import isfile, join
mypath="png dataset - original"

png_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

small_dataset=[]

for f in png_files:
    small_dataset.append(f.replace('.png', ''))
    
#-------------------------------------------------------------------------#

# BIG DATASET
# list all files from csv file with label 

import pandas as pd
 
data = pd.read_csv('stage_2_train.csv')

name = data['ID'].tolist()
classif = data['Label'].tolist()

# create list all_info with name, type and classification
# all_info is organized like: [['ID_xxxxx', 'epidural', 0], ...]

all_info = []

for i in range(len(name)):
    all_info.append(name[i].rsplit('_', 1))
    all_info[i].append(classif[i])
    
#-------------------------------------------------------------------------#
    
#JOIN info from small dataset with big dataset

small_info = []

stats = [['epidural', 0], ['intraparenchymal', 0], ['intraventricular', 0], 
         ['subarachnoid', 0], ['subdural', 0], ['any', 0], ['multi label', 0],
         ['single label', 0], ['no hemorrhage', 0]]

count = 0

for f in small_dataset:

    for k in range(len(all_info)):
        
        if all_info[k][0] == f :
            small_info.append(all_info[k])
            
            if all_info[k][2] == 1:
                if all_info[k][1] == 'epidural':
                    stats[0][1] +=1
                    count +=1
                if all_info[k][1] == 'intraparenchymal':
                    stats[1][1] +=1
                    count +=1
                if all_info[k][1] == 'intraventricular':
                    stats[2][1] +=1
                    count +=1
                if all_info[k][1] == 'subarachnoid':
                    stats[3][1] +=1
                    count +=1
                if all_info[k][1] == 'subdural':
                    stats[4][1] +=1
                    count +=1
                if all_info[k][1] == 'any':
                    stats[5][1] +=1
                
            if count > 1 and all_info[k][1] == 'any':
                small_info.append([all_info[k][0], 'label', 'multi'])
                stats[6][1] +=1
            elif count == 1 and all_info[k][1] == 'any':
                small_info.append([all_info[k][0], 'label', 'single'])
                stats[7][1] +=1
            elif count == 0 and all_info[k][1] == 'any':
                small_info.append([all_info[k][0], 'label', 'none'])
                stats[8][1] +=1
    
    count = 0
        

print(*stats)

            
            
    
    

    


