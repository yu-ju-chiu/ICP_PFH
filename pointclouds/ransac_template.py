#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np

def fit_plane_ransac(points, num_iterations, inlier_threshold, num_inliers):
    best_plane = None
    best_inliers = None
    best_error = float('inf')
    points_T = points.T

    for _ in range(num_iterations):
        # Randomly sample 3 points from the point cloud
        sample_indices = np.random.choice(points_T.shape[0], 3, replace = True)
        # print("points",points)
        sampled_points = points_T[sample_indices, :]

        # r_points = sampled_points.T
        A = np.column_stack((sampled_points[:, 0], sampled_points[:, 1], np.ones(sampled_points.shape[0])))
        B = sampled_points[:, 2]
        coefficients = np.linalg.lstsq(A, B, rcond=None)[0]


        # Calculate the error for all points
        A = np.column_stack((points_T[:, 0], points_T[:, 1], np.ones(points_T.shape[0])))
        B = points_T[:, 2]
        error = np.abs(B - A * coefficients)
        # print("error_1 shape",error)
        # error = np.dot(sampled_points[:, :3], coefficients) - points[:, 2]

        # error = np.linalg.norm(error)
        # print("error_1 shape",error)
        
        inliers = points_T[np.where(error < inlier_threshold)[0],:]
        
        outliers = points_T[np.where(error >= inlier_threshold)[0],:]

        if inliers.shape[0] > num_inliers:
            # If the number of inliers is valid, and the error is smaller than the best so far, update the best model

            A = np.column_stack((inliers[:, 0], inliers[:, 1], np.ones(inliers.shape[0])))
            B = inliers[:, 2]
            coefficients = np.linalg.lstsq(A, B, rcond=None)[0]
            error_new = B - A * coefficients
            error_new = np.linalg.norm(error_new)

            if error_new < best_error:
                best_plane = coefficients
                best_inliers = inliers
                best_outliers = outliers
                best_error = error_new

    # print("best_inliers",best_inliers.shape)
    print("best_plane",best_plane)

    return best_plane, best_inliers, best_outliers

###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')


    ###YOUR CODE HERE###
    # Show the input point cloud
    utils.view_pc([pc])

    # Define the number of RANSAC iterations and inlier threshold
    num_iterations = 3000
    inlier_threshold = 0.1  # Adjust this threshold based on your data
    num_inliers = 100

    X = utils.convert_pc_to_matrix(pc)
    # pc
    # Fit a plane to the data using ransac
    best_plane, best_inliers, best_outliers = fit_plane_ransac(X, num_iterations, inlier_threshold, num_inliers)
    # print("best_plane", best_plane)
    best_inliers = best_inliers.T
    best_outliers = best_outliers.T
    best_plane[2,:] = -1
    


    #Show the resulting point cloud
    #Draw the fitted plane
    best_outliers = utils.convert_matrix_to_pc(best_outliers)
    fig = utils.view_pc([best_outliers])
    utils.draw_plane(fig, best_plane, best_inliers[:, 0], color=(0.1, 0.9, 0.1, 0.5))
    best_inliers = utils.convert_matrix_to_pc(best_inliers)
    
    # pc = utils.convert_matrix_to_pc(X)
    # print("best_inliers",best_inliers)
    # print("pc",pc)
    # for i in best_inliers:
    #     print("i", np.matrix(i))
    #     pc = pc.remove(np.matrix(i))
    
    utils.view_pc([best_inliers],fig=fig, color='r')

    ###YOUR CODE HERE###
    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
