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
