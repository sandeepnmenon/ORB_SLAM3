import math
import numpy as np
from skimage.transform import warp, SimilarityTransform
from skimage.measure import ransac
import open3d as o3d

from opencv_utils import get_orb_matches
from orbslam_utils import read_orb_data, eulerangles_to_rotmat
from shared_utils import transform_points3d_list, visualize_points_with_matching_lines

class SimilarityTransform3D(SimilarityTransform):
    def __init__(self, matrix=None, scale=None, rotation=None, translation=None, *, dimensionality=3):
        super().__init__(matrix, scale, rotation, translation, dimensionality=dimensionality)

orb_features1 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor_100.csv"
orb_features2 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor_100_200.csv"

orb_features1 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor1_570.csv"
orb_features2 = "/home/menonsandu/stereo-callibration/ORB_SLAM3/Examples/Monocular/map_dataset-corridor1_570_670.csv"


def get_similarity_transform_3d(positions1, positions2, good_matches):
    
    src_points = []
    dst_points = []
    for match in good_matches:
        src_points.append(positions1[match.queryIdx])
        dst_points.append(positions2[match.trainIdx])

    src_points = np.array(src_points)
    dst_points = np.array(dst_points)
    print("src_points: {}".format(src_points.shape))
    print("dst_points: {}".format(dst_points.shape))

    # estimate the transformation
    model = SimilarityTransform(dimensionality=3)
    is_success = model.estimate(src_points, dst_points)

    # compare "true" and estimated transform parameters
    print("Affine transformation ", "successfull" if is_success else "failed")
    if is_success:
        print(model.scale, np.rad2deg(model.rotation), model.translation, model.dimensionality)
        print(model.params)
        print(np.mean(model.residuals(src_points, dst_points)))


    # robustly estimate affine transform model with RANSAC
    tranform = SimilarityTransform3D()
    print(tranform.dimensionality)
    model_robust, inliers = ransac((src_points, dst_points), SimilarityTransform3D, min_samples=3, residual_threshold=0.05, max_trials=100)
    outliers = inliers == False
    model = model_robust

    print("RANSAC:")
    print(model.scale, np.rad2deg(model.rotation), model.translation, model.dimensionality)
    print(model.params)
    print(np.mean(model.residuals(src_points, dst_points)))
    optimized_scale = model.scale
    optimized_rotation = model.rotation
    optimized_translation = model.translation.reshape(3, 1)
    optimized_rotation_matrix = eulerangles_to_rotmat(optimized_rotation[0], optimized_rotation[1], optimized_rotation[2])

    # TODO: Use Similartity transform with inliers again
    
    return optimized_scale, optimized_rotation, optimized_translation, optimized_rotation_matrix


if __name__ == "__main__":
    positions1, descriptors1 = read_orb_data(orb_features1)
    positions2, descriptors2 = read_orb_data(orb_features2)
    good_matches = get_orb_matches(descriptors1, descriptors2)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    print("Good matches: {}".format(len(good_matches)))
    print("Minimum distance: {}".format(good_matches[0].distance))

    optimized_scale, optimized_rotation, optimized_translation, optimized_rotation_matrix = get_similarity_transform_3d(positions1, positions2, good_matches)

    # Transform the matched positions
    transformed_positions1 = transform_points3d_list(positions1, optimized_scale, optimized_rotation_matrix, optimized_translation)

    # Visualise the results
    visualize = True
    if visualize:
        positions1 = transformed_positions1
        visualize_points_with_matching_lines(positions1, positions2, good_matches)