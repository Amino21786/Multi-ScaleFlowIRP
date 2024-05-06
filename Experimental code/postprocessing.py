import os
import numpy as np
import re
import csv

dir0 = r"R64/resu"
prefix = "vmacro"


# collate the homogenised permeability into a csv file

ktensorname = dir0+"/Ktensor_all.csv"

with open(ktensorname, 'w') as f:

    f.write('%s %s %s %s %s %s %s %s %s %s\n' % ("filename", "K11", "K22", "K33", "K12", "K13", "K23", "K21", "K31", "K32"))

match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')

for dir1 in os.listdir(dir0):
     # sanity check
    if os.path.isfile(dir0+"/"+dir1):    continue #if not a directory -> skip
    # collate the homogenised permeability
    Ktensor = np.zeros((3,3))
    with open(dir0+"/"+dir1+"/"+prefix+".log", 'r') as f:
        line = f.readline()
        while line:
            if 'Ktensor:' in line:
                Ktensor[:,0] = [float(x) for x in re.findall(match_number, f.readline())]
                Ktensor[:,1] = [float(x) for x in re.findall(match_number, f.readline())]
                Ktensor[:,2] = [float(x) for x in re.findall(match_number, f.readline())]
            line = f.readline()

   # write into a csv file
    with open(ktensorname, 'a', newline='') as f:
        writer = csv.writer(f)
    
        # Write the data row
        writer.writerow([dir1, Ktensor[0, 0], Ktensor[1, 1], Ktensor[2, 2],
                        Ktensor[0, 1], Ktensor[0, 2], Ktensor[1, 2],
                        Ktensor[1, 0], Ktensor[2, 0], Ktensor[2, 1]])