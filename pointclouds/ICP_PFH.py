#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
from scipy.spatial import cKDTree
# check the dissimilarity
from scipy.stats import wasserstein_distance  # Earth Mover's Distance
def calc_normals(pc):
    """TODO: Docstring for calc_normals.

    :pc: TODO
    :returns: TODO

    """
    print("\tCalculating surface normals. \n")
    normals = []
    ind_of_neighbors = []
    N = len(pc)
    for i in range(N):
        # Get the indices of neighbors, it is a list of tuples (dist, indx)
        indN = getNeighbors(pc[i], pc) #<- old code
        #indN = list((neigh.kneighbors(pc[i].reshape(1, -1), return_distance=False)).flatten())
        #indN.pop(0)

        # Breakout just the indices
        indN = [indN[i][1] for i in range(len(indN))] #<- old code
        ind_of_neighbors.append(indN)
        
        # PCA
        X = utils.convert_pc_to_matrix(pc)[:, indN]
        X = X - np.mean(X, axis=1)
        cov = np.matmul(X, X.T)/(len(indN))
        _, _, Vt = np.linalg.svd(cov)
        normal = Vt[2, :]

        # Re-orient normal vectors
        if np.matmul(normal, -1.*(pc[i])) < 0:
            normal = -1.*normal
        normals.append(normal)

    return normals, ind_of_neighbors
def FPFH(pc, norm, indNeigh):
    """Overriding base PFH to FPFH"""

    print("\tCalculating histograms fast method \n")
    N = len(pc)
    histArray = np.zeros((N, div**3))
    distArray = np.zeros((nneighbors))
    distList = []
    for i in range(N):
        u = np.asarray(norm[i].T).squeeze()
        
        features = np.zeros((len(indNeigh[i]), 3))
        for j in range(len(indNeigh[i])):
            pi = pc[i]
            pj = pc[indNeigh[i][j]]
            if np.arccos(np.dot(norm[i], pj - pi)) <= np.arccos(np.dot(norm[j], pi - pj)):
                ps = pi
                pt = pj
                ns = np.asarray(norm[i]).squeeze()
                nt = np.asarray(norm[indNeigh[i][j]]).squeeze()
            else:
                ps = pj
                pt = pi
                ns = np.asarray(norm[indNeigh[i][j]]).squeeze()
                nt = np.asarray(norm[i]).squeeze()
            
            u = ns
            difV = pt - ps
            dist = np.linalg.norm(difV)
            difV = difV/dist
            difV = np.asarray(difV).squeeze()
            v = np.cross(difV, u)
            w = np.cross(u, v)
            
            alpha = np.dot(v, nt)
            phi = np.dot(u, difV)
            theta = np.arctan(np.dot(w, nt) / np.dot(u, nt))
            
            features[j, 0] = alpha
            features[j, 1] = phi
            features[j, 2] = theta
            distArray[j] = dist

        distList.append(distArray)
        pfh_hist, bin_edges = self.calc_pfh_hist(features)
        histArray[i, :] = pfh_hist / (len(indNeigh[i]))

    fast_histArray = np.zeros_like(histArray)
    for i in range(N):
        k = len(indNeigh[i])
        for j in range(k):
            spfh_sum = histArray[indNeigh[i][j]]*(1/distList[i][j])
        
        fast_histArray[i, :] = histArray[i, :] + (1/k)*spfh_sum
    return fast_histArray
def compute_point_feature_histogram(points):
    # Assuming points is an Nx3 array
    # print("r", r)
    # print("r.shape", r.shape)

    # points = points.T
    points = np.array(points)
    # print("points_T", points)
    # print("points shape", points.shape)
    hist, edges = np.histogramdd(points, bins=(10, 10, 10), range=[[-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]])
    hist /= np.sum(hist)  # Normalize histogram
    return hist.ravel()

def compute_correspondences(pc_source, pc_target):
    hist_source = compute_point_feature_histogram(pc_source[:, :3])
    hist_target = compute_point_feature_histogram(pc_target[:, :3])
    # print("hist_source", hist_source)
    # print("hist_target", hist_target)

    # Compute Earth Mover's Distance (EMD) as a measure of dissimilarity
    emd = wasserstein_distance(hist_source, hist_target)
    print("emd", emd)

    # Return a list of indices with equal length for pc_source
    indices = np.arange(len(pc_target))
    print("indices", indices)

    return indices, emd

def icp(pc_source, pc_target, max_iterations=100, convergence_threshold=1e-4):
    
    T = np.eye(4)
    pc_source = utils.convert_pc_to_matrix(pc_source).T
    pc_target = utils.convert_pc_to_matrix(pc_target).T
    pc_aligned = pc_source.copy()
    errors = []
    error = 0
    
    for iteration in range(max_iterations):
        # Find nearest neighbors between the source and target point clouds
        # #compute correspondence
        # Compute point feature histograms

        # Find correspondences based on the dissimilarity between histograms
        # temp_error = error
        # indices, error = compute_correspondences(pc_aligned, pc_target)
        # errors.append(error)

        tree = cKDTree(pc_target[:,:3])
        distances, indices = tree.query(pc_aligned[:,:3])
        print("indices", indices)
        temp_error = error
        error = np.linalg.norm(distances)
        errors.append(error)
        
        # Extract corresponding points
        source_points = pc_aligned[:,:3]
        target_points = pc_target[indices, :3]

        # Calculate the transformation (Rigid Transformation) using SVD
        avg_p = np.mean (source_points, axis=0)
        avg_q = np.mean (target_points, axis=0)
        source_points = source_points - avg_p
        target_points = target_points - avg_q
        H = np.dot(source_points.T, target_points)
        # H =  target_points @ source_points.T
        U, S, Vt = np.linalg.svd(H)
        # R = np.dot(Vt.T, U.T)
        V = Vt.T
        reflection = np.diag([1, 1, np.linalg.det(V @ U.T)])
        # reflection = np.eye(3)
        R = V @ reflection @ U.T
        t = (avg_q.T - R @ avg_p.T).T    
        
        # Update the aligned point cloud
        pc_aligned_T = pc_aligned.T
        pc_aligned = R @ pc_aligned_T + t.reshape(3,1)
        pc_aligned = pc_aligned.T
        
        # Check for convergence
        if np.abs(temp_error - error) < convergence_threshold:
            break
    return pc_aligned, errors
###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    pc_target = utils.load_pc('cloud_icp_target0.csv') # Change this to load in a different target

    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])

    # Run ICP
    pc_aligned, errors = icp(pc_source, pc_target)
    # print("pc_aligned", pc_aligned.shape)

    pc_aligned = utils.convert_matrix_to_pc(pc_aligned.T)
    # print("pc_aligned", pc_aligned)

    utils.view_pc([pc_aligned, pc_target], None, ['b', 'r'], ['o', '^'])
    # plt.axis([-0.15, 0.15, -0.15, 0.15])
    ###YOUR CODE HERE###

    plt.show()
    #raw_input("Press enter to end:")
    fig_4 = plt.figure()
    index = np.arange(1,len(errors)+1,1) 
    plt.plot(index, errors, color='blue')
    plt.xlabel('Iteration')  
    plt.ylabel('Error')  
    plt.show()


if __name__ == '__main__':
    main()
