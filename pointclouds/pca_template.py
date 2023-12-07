#!/usr/bin/env python
import utils
import numpy
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
###YOUR IMPORTS HERE###

def main():

    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    ###YOUR CODE HERE###
    # Show the input point cloud
    fig = utils.view_pc([pc])
    threshold = 0.01
    #Rotate the points to align with the XY plane
    mean_pt = np.mean(pc, axis=0)
    pc_centered = pc - mean_pt
    X = utils.convert_pc_to_matrix(pc_centered)
    Q = X.T / (X.shape[1] - 1) ** 0.5
    U, S, V_T = np.linalg.svd(Q, full_matrices=True)
    print("V_T", V_T)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(S)[::-1]
    S = S[sorted_indices]
    V_T = V_T[:, sorted_indices]
    s = S ** 2
    num_dim = np.count_nonzero([s>threshold])
    # Compute the transformation matrix (rotation matrix)
    V = V_T.T
    X_rotated = V_T @ X
    # print("pc_rotated",pc_rotated)

    # Show the resulting point cloud
    pc_rotated = utils.convert_matrix_to_pc(X_rotated)
    fig_2 = utils.view_pc([pc_rotated])


    #Rotate the points to align with the XY plane AND eliminate the noise
    V_s = V[:, :num_dim]
    normal = V[:, -(3-num_dim):]
    V_s_T = V_s.T
    print("V_s_T", V_s_T)
    X_rotated_2D = V_s_T @ X
    z = np.zeros((1, X_rotated_2D.shape[1]))
    X_rotated_2D = np.r_[X_rotated_2D, z]
    pc_rotated_2D = utils.convert_matrix_to_pc(X_rotated_2D)
    # Show the resulting point cloud
    fig_3 = utils.view_pc([pc_rotated_2D])


    # draw that plane in green
    utils.draw_plane(fig, normal, mean_pt,color=(0.1, 0.9, 0.1, 0.5))

    ###YOUR CODE HERE###

    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
