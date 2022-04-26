import csv
import numpy as np


def read_orb_data(file_path):
    positions = []
    descriptors = []
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    for row in data:
        position=[]
        for i in range(3):
            position.append(row[i])
        positions.append(position)
        desc = []
        for i in range(3,len(row)):
            desc.append(row[i])
        desc[0] = desc[0][2:]
        desc[-1] = desc[-1][:-1]
        descriptors.append(desc)
    
    return np.array(positions).astype(np.float32), np.array(descriptors).astype(np.float32)
