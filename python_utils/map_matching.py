import numpy as np
from scipy import optimize

from opencv_utils import get_orb_matches
from orbslam_utils import read_orb_data

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
# Initialize a random transformation matrix for optimization
R = np.array([[np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)],
                [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]])
# Initialize a random translation for optimization
t = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
tranformation_matrix = np.concatenate((np.concatenate((R, t.reshape(3, 1)), axis=1), np.array([[0, 0, 0, 1]])))

input_data = np.concatenate((np.array(scale).reshape(1,), tranformation_matrix.flatten()))
args = (matched_positions)

def optimization_function(input_data, *args):
    scale = input_data[0]
    tranformation_matrix = input_data[1:].reshape(4, 4)
    matched_positions = args[0]
    error = 0
    for matched_position in matched_positions:
        # Transform the matched position
        query_point_homo = np.append(matched_position[0],[1]).reshape(4,1)
        target_point = matched_position[1]
        transformed_position = np.dot(tranformation_matrix, query_point_homo)
        transformed_position = transformed_position[:3, :].reshape(3)
        # Calculate the euclidean distance/error
        error += np.linalg.norm(transformed_position - target_point)
    
    return error


optimized_output = optimize.minimize(fun=optimization_function, x0=input_data, args=args, method='Powell')

print(f"Optimisation message: {optimized_output.message}")
optimised_values = optimized_output.x
error = optimized_output.fun

print(f"Optimised scale: {optimised_values[0]}")
print(f"Optimised transformation matrix: {optimised_values[1:]}")
print(f"Error: {error}")
