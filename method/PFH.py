#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import time

# N_neighbors = 8
# Radius = 0.03
# Div = 2

class PFH(object):
    """Parent class for PFH"""
    def __init__(self, bin, N_neighbors, Radius):
        self.bin = bin
        self.num_neighbors = N_neighbors
        self.radius = Radius
    
    def getNeighbors(self, pq, pc):
        """Get k nearest neighbors of the query point pq from pc, within the radius
        :pq: the query point pq
        :pc: point cloud
        :returns: k nearest neighbors
        """
        k = self.num_neighbors
        neighbors = []
        for i in range(len(pc)):
            dist = np.linalg.norm(pq-pc[i])
            if dist <= self.radius:
                neighbors.append((dist, i))
        neighbors.sort(key=lambda x:x[0])
        return neighbors[1:k+1]

    def get_neighbors(self, pq, pc):
        """Get k nearest neighbors of the query point pq from pc, within the radius
        :pq: the query point pq
        :pc: point cloud
        :returns: k nearest neighbors
        """
        k = self.num_neighbors
        neighbors = []
        dist = np.linalg.norm(pq-pc, axis = 1)
        neighbors_idx = np.where(dist <= self.radius)[0]
        neighbors_val = dist[np.where(dist <= self.radius)]
        neighbors = np.array([neighbors_idx, neighbors_val]).T
        sorted_indices = np.argsort(neighbors[:, 1])
        neighbors = neighbors[sorted_indices][1:k+1]
        return neighbors

    def calc_normals(self, pc):
        """ Docstring for calc_normals.
        :pc: point cloud
        :returns: normal vector
        """
        print("\tCalculating surface normals. \n")
        normals = []
        ind_of_neighbors = []
        # print("pc", pc.shape)
        N = pc.shape[0]
        for i in range(N):
            # Get the indices of neighbors, it is a list of tuples (dist, indx)
            indN = self.get_neighbors(pc[i], pc) #<- old code
            ind_of_neighbors.append(indN[:,0].astype(int))
            
            # SVD to get normal
            X = pc[indN[:,0].astype(int), :]
            X = X - np.mean(X, axis=0)
            cov = (X.T @ X)/(len(indN[:,0]))
            _, _, Vt = np.linalg.svd(cov)
            normal = Vt[2,:]
            # Re-orient normal vectors
            if normal @ (-1.*(pc[i])).T < 0:
                normal = -1.*normal
            normals.append(normal)

        return normals, ind_of_neighbors
    def calcHistArray(self, pc, normal, indNeigh):
        print("\tCalculating histograms naive method \n")

        N = len(pc)

        histograms = np.zeros((N, self.bin**3))
        for i in range(N):
            
            N_features = sp.comb(self.num_neighbors + 1, 2)
            features = []
            source_pair = np.append(indNeigh[i], [i])
            target_pair = np.append(indNeigh[i], [i])
            for s in source_pair:
                target_pair = target_pair[1:]
                for t in target_pair:
                    ps = pc[s]
                    pt = pc[t]
                    ns = np.asarray(normal[s]).squeeze()
                    nt = np.asarray(normal[t]).squeeze()
                    u = ns
                    d = np.linalg.norm(np.abs(pt - ps))
                    temp = np.asarray((pt - ps)/d).squeeze()
                    v = np.cross(temp, u)
                    w = np.cross(u, v)
                    alpha = np.dot(v, nt)
                    phi = np.dot(u, temp)
                    theta = np.arctan(np.dot(w, nt) / np.dot(u, nt))
                    features.append(np.array([alpha, phi, theta]))
                
            features = np.asarray(features)


            hist, edges = self.calc_hist(features)
            histograms[i, :] = hist / (N_features)
        return histograms
    def calc_thresholds(self):
        """
        :returns: feature's thresholds (3x(self.bin-1))
        """
        delta_bin = [2./self.bin, 2./self.bin, np.pi/self.bin]
        start_values = [-1, -1, -np.pi / 2]
        thresholds = [[start + i * delta for i in range(1, self.bin)] for start, delta in zip(start_values, delta_bin)]
        return np.array(thresholds)

    def cal_bin(self, si, fi):
        result = 0
        for i, s in enumerate(si):
            if fi >= s:
                result = i+1
        return result

    def calc_hist(self, feature):
        """Calculate histogram and bin edges.

        :f: feature vector of alpha, phi, theta (Nx3)
        :returns:
            hist - array of length div^3, represents number of samples per bin
            bin_edges - range(0, 1, 2, ..., (div^3+1)) 
        """
        # preallocate array sizes, create bin_edges
        # only 3 feature
        hist, edges = np.zeros(self.bin**3), np.arange(0,self.bin**3+1)
        
        # find the division thresholds for the histogram
        threshold = self.calc_thresholds()
        # Loop for every row in f from 0 to N
        for j in range(0, feature.shape[0]):
            # calculate the bin index to increment
            index = 0
            for i in range(3):
                index += self.cal_bin(threshold[i, :], feature[j, i]) * (self.bin**(i))
            # Increment histogram at that index
            hist[index] += 1
        
        return hist, edges

    def match(self, ps, pt):
        """Find matches from source to target points

        :pcS: Source point cloud
        :pcT: Target point cloud
        :returns: matchInd, distances

        """
        print("...Matching point clouds. \n")
        num_s = ps.shape[0]
        num_t = pt.shape[0]
        
        print("...Processing source point cloud...\n")
        norm_s, ind_nei_s = self.calc_normals(ps)
        hist_s = self.calcHistArray(ps, norm_s, ind_nei_s)
        
        print("...Processing target point cloud...\n")
        norm_t,ind_nei_t = self.calc_normals(pt)
        hist_t = self.calcHistArray(pt, norm_t, ind_nei_t)
        
        dist = []
        matchInd = []
        distances = []
        for i in range(num_s):
            for j in range(num_t):
                #appending the l2 norm and j
                dist.append((np.linalg.norm(hist_s[i]-hist_t[j]),j))
            dist.sort(key=lambda x:x[0]) #To sort by first element of the tuple
            matchInd.append(dist[0][1])
            distances.append(dist[0][0])
            dist = []
        
        return matchInd, distances