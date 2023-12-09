#!/usr/bin/env python
"""
ICP algorithm
"""
import utils
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import time
from method.PFH import PFH
import yaml


def icp(pc_source, pc_target, max_iterations=10, convergence_threshold=1e-4):
    with open("./config/config.yaml", 'r') as stream:
        cfg = yaml.safe_load(stream)

    
    pc_source = utils.convert_pc_to_matrix(pc_source).T
    pc_target = utils.convert_pc_to_matrix(pc_target).T
    pc_aligned = pc_source.copy()
    errors = []
    error = 0
    
    for iterations in range(max_iterations):
        start = time.process_time()
        temp_error = error

        if cfg['Method'] == "PFH":
            # Point Feature Histograms algorithm
            ###################################
            matching = PFH(cfg['Bin'], cfg['N_neighbors'], cfg['Radius']) 
            indices, distances = matching.match(pc_aligned, pc_target)
            ###################################
        else:
            # tranditional mehtod: cal distance
            ###################################
            tree = cKDTree(pc_target)
            distances, indices = tree.query(pc_aligned)
            ###################################
        # calculate error
        error = np.linalg.norm(distances)
        errors.append(error)
                
        # Extract corresponding points
        source_points = pc_aligned
        target_points = pc_target[indices, :]

        # Calculate the transformation (Rigid Transformation) using SVD
        avg_p = np.mean (source_points, axis=0)
        avg_q = np.mean (target_points, axis=0)
        source_points = source_points - avg_p
        target_points = target_points - avg_q
        H =  source_points.T @ target_points
        U, _, Vt = np.linalg.svd(H)
        V = Vt.T
        reflection = np.diag([1, 1, np.linalg.det(V @ U.T)])
        R = V @ reflection @ U.T
        t = (avg_q.T - R @ avg_p.T).T    
        
        # Update the aligned point cloud
        pc_aligned_T = pc_aligned.T
        pc_aligned = R @ pc_aligned_T + t.reshape(3,1)
        pc_aligned = pc_aligned.T

        # Check for convergence
        if np.abs(temp_error - error) < convergence_threshold:
            break
        print("===============================")
        print("iteration: ", iterations + 1)
        print("error: ", error)
        end = time.process_time()
        print("Time per iteration: ", end - start)
        print("===============================\n\n")
    return pc_aligned, errors


def main():
    #Import the cloud
    pc_source = utils.load_pc('data/cloud_icp_source.csv')
    pc_target = utils.load_pc('data/cloud_icp_target3.csv') # Change this to load in a different target

    utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])

    # Run ICP
    pc_aligned, errors = icp(pc_source, pc_target)

    pc_aligned = utils.convert_matrix_to_pc(pc_aligned.T)

    utils.view_pc([pc_aligned, pc_target], None, ['b', 'r'], ['o', '^'])
    plt.show()

    #Plot the result
    index = np.arange(1,len(errors)+1,1) 
    plt.plot(index, errors, color='blue')
    plt.xlabel('Iteration')  
    plt.ylabel('Error')  
    plt.show()


if __name__ == '__main__':
    main()
