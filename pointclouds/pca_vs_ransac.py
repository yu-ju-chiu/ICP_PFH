#!/usr/bin/env python
import utils
import numpy
import time
import random
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import numpy as np
import time
###YOUR IMPORTS HERE###

def add_some_outliers(pc,num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc


def cal_error(inliers, normal, pt):
    # normalization
    normal /= np.linalg.norm(normal)
    d = -pt.T * normal
    # Calculate the residuals for all points
    errors = (inliers.T @ normal) + d
    error = np.linalg.norm(errors)
    return errors, error



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

    best_inliers = best_inliers.T
    best_outliers = best_outliers.T

    return best_plane, best_inliers, best_outliers

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    num_tests = 10
    fig = None
    error_PCA_list = []
    error_RANSAC_list = []
    num_outliers_pca_list = []
    num_outliers_ransac_list = []
    pca_time = []
    ransac_time = []
    for i in range(0,num_tests):
        pc = add_some_outliers(pc,10) #adding 10 new outliers for each test
        # fig = utils.view_pc([pc])

        ###YOUR CODE HERE###
        
        #PCA
        pca_start = time.time()
        threshold = 0.1
        inlier_threshold = 0.225
        #Rotate the points to align with the XY plane
        mean_pt = np.mean(pc, axis=0)
        pc_centered = pc - mean_pt
        pc_matrix = utils.convert_pc_to_matrix(pc)
        X = utils.convert_pc_to_matrix(pc_centered)
        X_T = X.T
        Q = X.T / (X.shape[1] - 1) ** 0.5
        U, S, V_T = np.linalg.svd(Q, full_matrices=True)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(S)[::-1]
        S = S[sorted_indices]
        V_T = V_T[:, sorted_indices]
        s = S ** 2
        # print("s", s)
        num_dim = np.count_nonzero([s>threshold])
        V = V_T.T
        # normal = V[:, -(3-num_dim):]
        normal = V[:, -1]
        pca_end = time.time()

        pca_time.append(pca_end - pca_start)

        # error
        errors, error = cal_error(pc_matrix, normal, mean_pt) # X = 3 * 200
        errors = np.abs(errors)
        # print("errors_pca",errors)
        inliers_PCA = pc_matrix[:,np.where(errors < inlier_threshold)[0]] # 3 * 200
        outliers_PCA  = pc_matrix[:,np.where(errors >= inlier_threshold)[0]] # 3 * 200
        errors_pca, error_pca = cal_error(inliers_PCA, normal, mean_pt)
        # print("outliers_PCA",outliers_PCA.shape[1])
        # print("error_PCA",error_pca)
        error_PCA_list.append(outliers_PCA.shape[1])
        num_outliers_pca_list.append(error_pca)


        # normal = V_T[:,-1]

        #RANSAC
        ransac_start = time.time()
        # Define the number of RANSAC iterations and inlier threshold
        num_iterations = 3000
        inlier_threshold = 0.2  # Adjust this threshold based on your data
        num_inliers = 150
        pc_centered = utils.convert_pc_to_matrix(pc)
        best_plane, best_inliers, best_outliers = fit_plane_ransac(pc_centered, num_iterations, inlier_threshold, num_inliers)
        # print("best_plane", best_plane)
        best_plane[2,:] = -1
        ransac_end = time.time()

        ransac_time.append(ransac_end - ransac_start)

        # error
        errors, error = cal_error(pc_matrix, best_plane, best_inliers[:, 0]) # X = 3 * 200
        errors = np.abs(errors)
        # print("errors_pca",errors)
        inliers_ransac= pc_matrix[:,np.where(errors < inlier_threshold)[0]] # 3 * 200
        outliers_ransac  = pc_matrix[:,np.where(errors >= inlier_threshold)[0]] # 3 * 200
        errors_ransac, error_ransac = cal_error(inliers_ransac, best_plane, best_inliers[:, 0])
        # print("outliers_ransac",outliers_ransac.shape[1])
        # print("error_ransac",error_ransac)
        error_RANSAC_list.append(outliers_ransac.shape[1])
        num_outliers_ransac_list.append(error_ransac)

        # errors_ransac, error_ransac = cal_error(best_inliers, best_plane, best_inliers[:, 0])
        # print("inliers_ransac",best_inliers.shape[1])
        # print("outliers_ransac",best_outliers.shape[1])
        # print("error_ransac",error_ransac)

        #Draw the fitted plane
        # best_outliers = utils.convert_matrix_to_pc(best_outliers)
        # fig = utils.view_pc([best_outliers])
        # utils.draw_plane(fig, best_plane, best_inliers[:, 0], color=(0.1, 0.9, 0.1, 0.5))
        # best_inliers = utils.convert_matrix_to_pc(best_inliers)
        # utils.view_pc([best_inliers],fig=fig, color='r')


        #this code is just for viewing, you can remove or change it
        # input("Press enter for next test:")
        if i == 9:
            outliers_ransac = utils.convert_matrix_to_pc(outliers_ransac)
            inliers_ransac = utils.convert_matrix_to_pc(inliers_ransac)
            fig_1 = utils.view_pc([outliers_ransac], title="RANSAC")
            utils.view_pc([inliers_ransac], fig_1, color='r', title="RANSAC")
            utils.draw_plane(fig_1, best_plane, best_inliers[:, 0], color=(0.1, 0.9, 0.1, 0.5))

            outliers_PCA = utils.convert_matrix_to_pc(outliers_PCA)
            inliers_PCA = utils.convert_matrix_to_pc(inliers_PCA)
            fig_2 = utils.view_pc([outliers_PCA], title="PCA")
            utils.view_pc([inliers_PCA], fig_2, color='r', title="PCA")
            utils.draw_plane(fig_2, normal, mean_pt,color=(0.1, 0.9, 0.1, 0.5))

            
            #visualization 
            fig_3 = plt.figure()
            plt.plot(error_RANSAC_list, num_outliers_ransac_list, color='blue')
            plt.plot(error_PCA_list, num_outliers_pca_list, color='green')
            plt.legend(['RANSAC', 'PCA'])
            plt.xlabel('Number of outliers')  
            plt.ylabel('Error')  
            
            # displaying the title 
            plt.title("Error vs Number of outliers") 
            plt.show()

            fig_4 = plt.figure()
            index = np.arange(1,11,1) 
            plt.plot(index, ransac_time, color='blue')
            plt.plot(index, pca_time, color='green')
            
            plt.legend(['RANSAC', 'PCA'])
            plt.xlabel('Iteration')  
            plt.ylabel('Time')  
            
            # displaying the title 
            plt.title("Time vs iteration") 
            plt.show()   


            # plt.close(fig)
        ###YOUR CODE HERE###

    input("Press enter to end")


if __name__ == '__main__':
    main()
