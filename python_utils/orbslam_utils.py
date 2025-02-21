import csv
import numpy as np
import math


def read_orb_data(file_path):
    positions = []
    descriptors = []
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    for row in data:
        position = []
        for i in range(3):
            position.append(row[i])
        positions.append(position)
        desc = []
        for i in range(3, len(row)):
            desc.append(row[i])
        desc[0] = desc[0][2:]
        desc[-1] = desc[-1][:-1]
        descriptors.append(desc)

    return np.array(positions).astype(np.float32), np.array(descriptors).astype(np.float32)


def read_time_stamped_poses_from_csv_file(csv_file,  time_scale=1.0):
    """
    Reads time stamped poses from a CSV file.
    Assumes the following line format:
      timestamp [s], x [m], y [m], z [m], qx, qy, qz, qw
    """
    with open(csv_file, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        csv_data = list(csv_reader)
        if 'time' in csv_data[0][0]:
            # The first line is the header
            csv_data = csv_data[1:]
        time_stamped_poses = np.array(csv_data)
        time_stamped_poses = time_stamped_poses.astype(float)

    time_stamped_poses[:, 0] *= time_scale
    # Extract the quaternions from the poses.
    times = time_stamped_poses[:, 0].copy()
    xyz = time_stamped_poses[:, 1:4].copy()
    quaternions = time_stamped_poses[:, 4:8].copy()
    print("Read {} poses from {}.".format(len(times), csv_file))

    return times, xyz, quaternions


def eulerangles_to_rotmat(roll, pitch, yaw):
    rotmat_roll = np.array(
        [
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ]
    )
    rotmat_pitch = np.array(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ]
    )
    rotmat_yaw = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw),  math.cos(yaw), 0],
            [0, 0, 1]
        ]
    )
    rotmat = np.matmul(np.matmul(rotmat_yaw, rotmat_pitch), rotmat_roll)
    return rotmat
