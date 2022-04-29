import numpy as np
import cv2
import os
import csv
import open3d as o3d
from opencv_utils import get_orb_matches
from orbslam_utils import read_orb_data
from similarity_transform import get_similarity_transform_3d
from shared_utils import transform_points3d_list, visualize_points_with_matching_lines

orb_features1 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor_100.csv"
orb_features2 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor_100_200.csv"

# orb_features1 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor1_570.csv"
# orb_features2 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor1_570_670.csv"


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
print("Minimum distance: {}".format(good_matches[0].distance))

# Visualize the matches
visualize_points_with_matching_lines(positions1, positions2, good_matches)

# Visualise the matches after map matching
optimized_scale, optimized_rotation, optimized_translation, optimized_rotation_matrix = get_similarity_transform_3d(positions1, positions2, good_matches)
# Transform the matched positions
transformed_positions1 = transform_points3d_list(positions1, optimized_scale, optimized_rotation_matrix, optimized_translation)
positions1 = transformed_positions1
visualize_points_with_matching_lines(positions1, positions2, good_matches)

