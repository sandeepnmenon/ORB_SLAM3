import numpy as np
import open3d as o3d


def transform_points3d_list(points, scale, rotation_matrix, translation):
    transformed_points = []
    for idx, position in enumerate(points):
        # Scale the query point
        scaled_position = position * scale
        scaled_position = scaled_position.reshape(3, 1)

        # Transform the matched position
        transformed_position = np.matmul(
            rotation_matrix, scaled_position) + translation
        transformed_position = transformed_position.reshape(3)
        transformed_points.append(transformed_position)

    transformed_points = np.array(transformed_points).astype(np.float32)

    return transformed_points


def visualize_points_with_matching_lines(positions1, positions2, cv2_matches, trajectory1_points=None, trajectory2_points=None, display=True):
    if not display:
        return

    positions = np.concatenate((positions1, positions2))
    lines = []
    for match in cv2_matches:
        lines.append([match.queryIdx, len(positions1) + match.trainIdx])

    colors = [[1, 0, 0] for i in range(len(positions))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(positions),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(positions1)
    pcd1.paint_uniform_color([0.8, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(positions2)
    pcd2.paint_uniform_color([0, 0.8, 0])

    trajectory1 = o3d.geometry.PointCloud()
    if trajectory1_points is not None:
        trajectory1.points = o3d.utility.Vector3dVector(trajectory1_points)
        trajectory1.paint_uniform_color([1, 1, 0])

    trajectory2 = o3d.geometry.PointCloud()
    if trajectory2_points is not None:
        trajectory2 = o3d.geometry.PointCloud()
        trajectory2.points = o3d.utility.Vector3dVector(trajectory2_points)
        trajectory2.paint_uniform_color([0, 1, 1])

    o3d.visualization.draw_geometries(
        [pcd1, pcd2, line_set, trajectory1, trajectory2])
    # o3d.visualization.draw_geometries([trajectory1, trajectory2])
