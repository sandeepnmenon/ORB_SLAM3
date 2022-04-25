import numpy as np
import cv2
import os
import csv
import open3d as o3d
from opencv_utils import get_orb_matches

orb_features1 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor_100.csv"
orb_features2 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor_100_200.csv"

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

positions1, descriptors1 = read_orb_data(orb_features1)
positions2, descriptors2 = read_orb_data(orb_features2)

# Visualize positions of the two datasets with different colors
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(positions1)
pcd1.paint_uniform_color([1, 0, 0])

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(positions2)
pcd2.paint_uniform_color([0, 1, 0])

# o3d.visualization.draw_geometries([pcd1, pcd2])

good_matches = get_orb_matches(descriptors1, descriptors2)
good_matches = sorted(good_matches, key=lambda x: x.distance)
print("Good matches: {}".format(len(good_matches)))

positions = np.concatenate((positions1, positions2))
lines = []

for match in good_matches:
    lines.append([match.queryIdx, len(descriptors1) + match.trainIdx])

colors = [[1, 0, 0] for i in range(len(positions))]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(positions),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd1, pcd2, line_set])
