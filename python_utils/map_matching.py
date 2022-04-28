import numpy as np
from scipy import optimize

from opencv_utils import get_orb_matches
from orbslam_utils import read_orb_data, eulerangles_to_rotmat
import open3d as o3d

orb_features1 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor_100.csv"
orb_features2 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor_100_200.csv"

# orb_features1 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor1_570.csv"
# orb_features2 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor1_570_670.csv"


positions1, descriptors1 = read_orb_data(orb_features1)
positions2, descriptors2 = read_orb_data(orb_features2)

good_matches = get_orb_matches(descriptors1, descriptors2)
good_matches = sorted(good_matches, key=lambda x: x.distance)
print("Good matches: {}".format(len(good_matches)))
print("Minimum distance: {}".format(good_matches[0].distance))

matched_positions = []
for match in good_matches:
    matched_positions.append([positions1[match.queryIdx], positions2[match.trainIdx]])

# Initialize a random scale for optimization
scale = np.random.uniform(0.05, 10)
# Initialize random euler angles for optimization
roll = np.random.uniform(-np.pi, np.pi)
pitch = np.random.uniform(-np.pi, np.pi)
yaw = np.random.uniform(-np.pi, np.pi)

# Initialize a random translation for optimization
translation = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

input_data = np.concatenate((np.array([scale, roll, pitch, yaw]), translation.flatten()))
args = (matched_positions)

def optimization_function(input_data, *args):
    scale = input_data[0]
    roll = input_data[1]
    pitch = input_data[2]
    yaw = input_data[3]
    translation_vector = input_data[4:].reshape(3, 1)

    # Get roll pitch yaw and convert to transformation matrix
    rotation_matrix = eulerangles_to_rotmat(roll, pitch, yaw)

    matched_positions = args[0]
    error = 0
    for matched_position in matched_positions:
        # Scale the query point
        query_point = matched_position[0] * scale
        query_point = query_point.reshape(3, 1)
        # Transform the matched position
        transformed_position = np.matmul(rotation_matrix, query_point) + translation_vector
        transformed_position = transformed_position.reshape(3)

        # Calculate the euclidean distance/error
        target_point = matched_position[1]
        # error += np.linalg.norm(transformed_position - target_point)/len(matched_positions)
        error += np.linalg.norm(transformed_position - target_point)

    return error


optimized_output = optimize.minimize(fun=optimization_function, x0=input_data, args=args, method='Powell')

print(f"Optimisation message: {optimized_output.message}")
optimised_values = optimized_output.x
error = optimized_output.fun
optimized_scale = optimised_values[0]
optimized_roll = optimised_values[1]
optimized_pitch = optimised_values[2]
optimized_yaw = optimised_values[3]
optimized_translation = optimised_values[4:].reshape(3, 1)

optimized_rotation_matrix = eulerangles_to_rotmat(optimized_roll, optimized_pitch, optimized_yaw)

print(f"Optimised scale: {optimized_scale}")
print(f"Optimised transformation matrix: {optimized_rotation_matrix}")
print(f"Optimised translation: {optimized_translation}")
print(f"Error: {error}")


# Transform the matched positions
transformed_positions1 = []
for idx, position in enumerate(positions1):
    # Scale the query point
    scaled_position = position * optimized_scale
    scaled_position = scaled_position.reshape(3, 1)

    # Transform the matched position
    transformed_position = np.matmul(optimized_rotation_matrix, scaled_position) + optimized_translation
    transformed_position = transformed_position.reshape(3)
    transformed_positions1.append(transformed_position)

transformed_positions1 = np.array(transformed_positions1).astype(np.float32)

# Visualise the results
visualize = True
if visualize:
    positions1 = transformed_positions1
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

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(positions1)
    pcd1.paint_uniform_color([1, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(positions2)
    pcd2.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pcd1, pcd2, line_set])
