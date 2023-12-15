#!/usr/bin/env python
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.special import rel_entr

class PFH(object):
    """Parent class for PFH"""
    def __init__(self, bin, N_neighbors, Radius):
        self.bin = bin
        self.num_neighbors = N_neighbors
        self.radius = Radius

    def get_neighbors(self, pq, pc):
        """Get k nearest neighbors of the query point pq from pc, within the radius
        :pq: the query point pq
        :pc: point cloud
        :returns: k nearest neighbors
        """
        neighbors = []
        dist = np.linalg.norm(pq-pc, axis = 1)
        neighbors_idx = np.where(dist <= self.radius)[0]
        neighbors_val = dist[neighbors_idx]
        neighbors = np.array([neighbors_idx, neighbors_val]).T
        sorted_indices = np.argsort(neighbors[:, 1])
        neighbors = neighbors[sorted_indices][1:self.num_neighbors + 1]
        return neighbors

    def calc_normals(self, pc, curv_thres):
        """ Docstring for calc_normals.
        :pc: point cloud
        :returns: normal vector
        """
        print("\tCalculating surface normals. \n")
        N = pc.shape[0]
        normals = np.zeros((N, 1, 3))
        filtered_pc_list = []
        ind_of_neighbors = []
        # print("pc", pc.shape)
        
        for i in range(N):
            # Get the indices of neighbors, it is a list of tuples (dist, indx)
            indN = self.get_neighbors(pc[i], pc) #<- old code
            ind_of_neighbors.append(indN[:,0].astype(int))
            
            # SVD to get normal
            X = pc[indN[:,0].astype(int), :]
            X = X - np.mean(X, axis=0)
            y = X / np.sqrt(X.shape[0]-1)
            _, S, Vt = np.linalg.svd(y)
            
            # Curvature to decide which points to use in histogram
            if curv_thres != 0.0:
                K = S[0]/np.sum(S)
                if K>curv_thres: 
                    filtered_pc_list.append(i)
            else:
                filtered_pc_list.append(i)
            
            normal = Vt[2,:]
            # Re-orient normal vectors
            if normal @ (-1.*(pc[i])).T < 0:
                normal = -1.*normal
            normals[i, :, :] = normal

        return normals, ind_of_neighbors, filtered_pc_list
    
    def calcHistArray(self, pc, normal, indNeigh, pc_list):
        print("\tCalculating histograms naive method \n")
        N = len(pc)
        histograms = np.zeros((N, self.bin**3))
        for i in pc_list:
            N_features = sp.comb(self.num_neighbors + 1, 2)
            point_ids = np.append(indNeigh[i], [i])
            point_pairs = np.array(np.meshgrid(point_ids, point_ids)).T.reshape(-1, 2)
            point_pairs = point_pairs[point_pairs[:, 0] < point_pairs[:, 1]]
            s = point_pairs[:, 0]
            t = point_pairs[:, 1]
            ps = pc[s]
            pt = pc[t]
            ns = normal[s].squeeze()
            nt = normal[t].squeeze()
            u = ns
            d = np.linalg.norm(np.abs(pt - ps), axis=1, keepdims=True)
            temp = (pt - ps)/d
            v = np.cross(temp, u)
            w = np.cross(u, v)
            alpha = np.sum(v @ nt.T, axis=1)
            phi = np.asarray(np.sum(u @ temp.T, axis=1)).squeeze()
            theta = np.arctan(np.sum(w @ nt.T, axis=1) / np.sum(u @ nt.T, axis=1))
            features = np.array([alpha, phi, theta]).T

            hist, edges = self.calc_hist(features)
            histograms[i, :] = hist / (N_features)
            
        return histograms
    def plot_hist(self, hist_s, hist_t):

        WIDTH = 0.4
        index_s = np.arange(1,(self.bin**3)+1,1)
        index_t = index_s - WIDTH/2
        for i in range(1):
            plt.show()
            plt.bar(index_s, hist_s[i], color='blue' ,width=WIDTH, align='edge')
            plt.bar(index_t, hist_t[i], color='red' ,width=WIDTH)
            plt.xlabel('Bins')  
            plt.ylabel('Ration of point one bin')
        

    def calc_thresholds(self):
        """
        :returns: feature's thresholds (3x(self.bin-1))
        """
        delta_bin = [2./self.bin, 2./self.bin, np.pi/self.bin]
        start_values = [-1, -1, -np.pi / 2]
        thresholds = [[start + i * delta for i in range(1, self.bin)] for start, delta in zip(start_values, delta_bin)]
        return np.array(thresholds)

    def cal_bin(self, si, fi):
        return np.searchsorted(si, fi)

    def calc_hist(self, feature):
        """Calculate histogram and bin edges.

        :f: feature vector of alpha, phi, theta (Nx3)
        :returns:
            hist - array of length div^3, represents number of samples per bin
            bin_edges - range(0, 1, 2, ..., (div^3+1)) 
        """

        # preallocate array sizes, create bin_edges
        # only 3 feature
        # hist, edges = np.zeros(self.bin**3), np.arange(0,self.bin**3+1)
        
        # # find the division thresholds for the histogram
        # threshold = self.calc_thresholds()
        # # Loop for every row in f from 0 to N
        # for j in range(0, feature.shape[0]):
        #     # calculate the bin index to increment
        #     index = 0
        #     for i in range(3):
        #         index += self.cal_bin(threshold[i, :], feature[j, i]) * (self.bin**(i))
        #     # Increment histogram at that index
        #     hist[index] += 1
        # return hist, edges
        # ########
        hist, edges = np.histogramdd(feature, bins=self.bin)
        # print("hist", hist.flatten())
        return hist.flatten(), edges[0]


    def match(self, ps, pt, curv_thres):
        """Find matches from source to target points

        :pcS: Source point cloud
        :pcT: Target point cloud
        :returns: matchInd, distances

        """
        print("...Matching point clouds. \n")
        print("...Processing source point cloud...\n")
        norm_s, ind_nei_s, filtered_ps_list = self.calc_normals(ps, curv_thres)
        hist_s = self.calcHistArray(ps, norm_s, ind_nei_s, filtered_ps_list)
        
        
        print("...Processing target point cloud...\n")
        norm_t, ind_nei_t, filtered_pt_list = self.calc_normals(pt, curv_thres)
        hist_t = self.calcHistArray(pt, norm_t, ind_nei_t, filtered_pt_list)
        # self.plot_hist(hist_s, hist_t)

        distances = np.linalg.norm(hist_s[filtered_ps_list, np.newaxis] - hist_t[filtered_pt_list], axis=2)
        matchInd = np.argmin(distances, axis=1)
        distances = np.min(distances, axis=1)

        # Filter matches based on the distance threshold
        valid_matches = distances < 0.3
        matchInd = matchInd[valid_matches]
        distances = distances[valid_matches]
        filtered_ps_list = np.array(filtered_ps_list)[valid_matches].tolist()

        return matchInd, distances, filtered_ps_list, filtered_pt_list