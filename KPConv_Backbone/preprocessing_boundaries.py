#-- preprocessing.py
#-- Compute the normals and the boundaries of the input point clouds
#-- Author: Shenglan Du
#-- Time: 30-03-2021


# OS functions
from os import listdir, makedirs
from os.path import exists, join

from utils import ply
import numpy as np
from sklearn.neighbors import KDTree
import time
import math


def preprocessing():
    # specify the number k
    k = 4

    print('\nPre-processing files')
    t0 = time.time()

    # you can specify the paths
    # --- data_path for original data
    # --- process_path for outputting files with boundaries and directions
    data_path = ''
    process_path = ''
    if not exists(process_path):
        makedirs(process_path)

    # loop over the ply files
    for file_name in listdir(data_path):
        if file_name.split('.')[1] == 'ply':
            cloud_name = file_name.split('.')[0]
            print('Cloud - %s' % (cloud_name))

            # read original ply file
            ply_file = join(data_path, file_name)
            data = ply.read_ply(ply_file)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            colors = np.vstack((data['red'], data['green'], data['blue'])).T
            labels = data['class']

            # build kd tree of the input points
            tree = KDTree(points, leaf_size=50)

            print('Start calculating normals and boundaries...\n')

            # initialize the containers
            normals = []
            boundaries = []

            # retrieve the points to obtain normals and boundaries
            bar = 0.1
            for i in range(len(points)):
                idx = tree.query(points[i:i+1], k=k, return_distance=False)
                idx = np.squeeze(idx, axis=0)

                # compute normals and check if the centroid is boundary or not
                neighbours = points[idx, :]
                normal = (normal_from_pts(neighbours)).astype(np.float32)
                normals.append(normal)
                neigh_labels = labels[idx]
                boundary = np.uint8(check_is_boundary(neigh_labels))
                boundaries.append(boundary)

                # Track the progress
                progress = i * 1.0 / len(points)
                if progress > bar:
                    print('{:.1f}% done in {:.1f}s'.format(progress * 100, time.time() - t0))
                    t0 = time.time()
                    bar += 0.1

            normals = np.array(normals).astype(np.float32)
            boundaries = np.array(boundaries).astype(np.float32)

            print('Start calculating directions...\n')

            # obtain the boundary points and boundary kd-tree
            boundary_bool = np.ma.masked_where(boundaries == 1, boundaries)
            boundary_points = points[boundary_bool.mask]
            boundary_tree = KDTree(boundary_points, leaf_size=10)

            # Create a list to store directions
            dir_x = []
            dir_y = []
            dir_z = []

            # retrieve the points again, obtain the direction w.r.t the closest boundary point
            bar = 0.1
            for i in range(len(points)):
                # if current point is boundary, the directions are 0
                if np.ceil(boundaries[i]) == 1:
                    dx = np.float32(0.0)
                    dy = np.float32(0.0)
                    dz = np.float32(0.0)
                else:
                    # obtain the closest boundary point
                    neigh_dist, neigh_idx = boundary_tree.query(points[i:i+1], k=1, return_distance=True)
                    closest_boundary = boundary_points[neigh_idx[0][0]]

                    # obtain the normalized direction
                    dx = (points[i][0] - closest_boundary[0]).astype(np.float32)
                    dy = (points[i][1] - closest_boundary[1]).astype(np.float32)
                    dz = (points[i][2] - closest_boundary[2]).astype(np.float32)
                    dx = dx / math.sqrt(dx**2 + dy**2 + dz**2)
                    dy = dy / math.sqrt(dx**2 + dy**2 + dz**2)
                    dz = dz / math.sqrt(dx**2 + dy**2 + dz**2)

                # Append to the container
                dir_x.append(dx)
                dir_y.append(dy)
                dir_z.append(dz)

                # Track the progress
                progress = i * 1.0 / len(points)
                if progress > bar:
                    print('{:.1f}% done in {:.1f}s'.format(progress * 100, time.time() - t0))
                    t0 = time.time()
                    bar += 0.1

            dir_x = np.array(dir_x).astype(np.float32)
            dir_y = np.array(dir_y).astype(np.float32)
            dir_z = np.array(dir_z).astype(np.float32)

            print('Write to the output files...\n')

            # output results as new ply files
            process_file = join(process_path, file_name)
            ply.write_ply(process_file,
                          [points, colors, normals, boundaries, dir_x, dir_y, dir_z, labels],
                          ['x', 'y', 'z', 'red', 'green', 'blue', 'nx', 'ny', 'nz', 'is_boundary', 'dx', 'dy', 'dz', 'class'])

    return


def normal_from_pts(pts):
    """
    Computes the normal of a given pointset
    """
    # Check the length of the points
    dim = pts.shape[1]
    if len(pts) < dim:
        return np.zeros(dim)

    # compute the covariance of the points
    pts_cov = np.cov(pts.T)

    # compute the eigen vectors of the matrix
    _, v = np.linalg.eigh(pts_cov)

    # obtain the normal as the first column of the eigen matrix which has the smallest eigen value
    normal = v[:, 0]
    return normal


def check_is_boundary(labels, n=4):
    """
    Check by neighbouring points if the current entry belongs to boundary or not
    """
    # select only the first n neighbour labels
    labels = labels[:n]

    # check if all labels in the array are the same
    result = np.all(labels == labels[0])

    # if result is true, entry is not boundary, otherwise it is the boundary
    if result:
        return 0
    else:
        return 1


if __name__ == '__main__':
    preprocessing()
