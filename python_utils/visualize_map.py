import numpy as np
import cv2
import os
import csv
import open3d as o3d
from opencv_utils import get_orb_matches
from orbslam_utils import read_orb_data, read_time_stamped_poses_from_csv_file
from similarity_transform import get_similarity_transform_3d
from shared_utils import transform_points3d_list, visualize_points_with_matching_lines

orb_features1 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor_100.csv"
orb_features2 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor_100_200.csv"

orb_features1 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor1_570.csv"
orb_features2 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor1_570_670.csv"

frame_points1 = orb_features1.replace("map_", "f_")
frame_points2 = orb_features2.replace("map_", "f_")
key_frame_points1 = orb_features1.replace("map_", "kf_")
key_frame_points2 = orb_features2.replace("map_", "kf_")

frame_times1, xyz1, quaternions1 = read_time_stamped_poses_from_csv_file(
    frame_points1, time_scale=1.0)
frame_times2, xyz2, quaternions2 = read_time_stamped_poses_from_csv_file(
    frame_points2, time_scale=1.0)
key_frame_times1, kf_xyz1, kf_quaternions1 = read_time_stamped_poses_from_csv_file(
    key_frame_points1, time_scale=1.0)
key_frame_times2, kf_xyz2, kf_quaternions2 = read_time_stamped_poses_from_csv_file(
    key_frame_points2, time_scale=1.0)

positions1, descriptors1 = read_orb_data(orb_features1)
positions2, descriptors2 = read_orb_data(orb_features2)

good_matches = get_orb_matches(descriptors1, descriptors2)
good_matches = sorted(good_matches, key=lambda x: x.distance)
print("Good matches: {}".format(len(good_matches)))
print("Minimum distance: {}".format(good_matches[0].distance))

display = True
# Visualize the matches
visualize_points_with_matching_lines(
    positions1, positions2, good_matches, trajectory1_points=xyz1, trajectory2_points=xyz2, display=display)
visualize_points_with_matching_lines(
    positions1, positions2, good_matches, trajectory1_points=kf_xyz1, trajectory2_points=kf_xyz2, display=display)

# Visualise the matches after map matching
optimized_scale, optimized_rotation, optimized_translation, optimized_rotation_matrix = get_similarity_transform_3d(
    positions1, positions2, good_matches)
# Transform the matched positions
transformed_positions1 = transform_points3d_list(
    positions1, optimized_scale, optimized_rotation_matrix, optimized_translation)
positions1 = transformed_positions1
transformed_xyz1 = transform_points3d_list(
    xyz1, optimized_scale, optimized_rotation_matrix, optimized_translation)
xyz1 = transformed_xyz1
transformed_kf_xyz1 = transform_points3d_list(
    kf_xyz1, optimized_scale, optimized_rotation_matrix, optimized_translation)
kf_xyz1 = transformed_kf_xyz1

visualize_points_with_matching_lines(
    positions1, positions2, good_matches, trajectory1_points=xyz1, trajectory2_points=xyz2, display=display)

visualize_points_with_matching_lines(
    positions1, positions2, good_matches, trajectory1_points=kf_xyz1, trajectory2_points=kf_xyz2, display=display)