# This is the from scratch implementation of LOF algorithm.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance


class LOF_alg():
    def __init__(self, k):
        self.k = k
        self.n_features = 0
        self.n_samples = 0
        self.n_samples_pr = 0
        self.n_features_pr = 0
        self.lrd = []
        self.train_data = []
        self.ind_nbr = None
        self.dist_nbr = None

    def sparse_argsort(self, arr):
        indices = np.nonzero(arr)[0]
        return indices[np.argsort(arr[indices])]

    def fit(self, X):
        X = np.array(X)
        self.train_data = X
        self.n_samples, self.n_features = X.shape[0], X.shape[1]  # number of rows and columns

    def predict(self, Y):
        Y = np.array(Y)
        resultArray = np.zeros(shape=Y.shape[0])
        self.n_samples_pr, self.n_features_pr = Y.shape[0], Y.shape[1]  # number of rows and columns
        for i in range(self.n_samples_pr):
            lrd_others = 0
            train_data = self.train_data
            train_data = np.append(train_data, Y[[i]], axis=0)
            self.dist_nbr, self.ind_nbr = self.knn(train_data, Y[i])
            lrd = self.local_reachability_density(train_data)
            ind_nbr = self.ind_nbr
            for item in ind_nbr:
                self.dist_nbr, self.ind_nbr = self.knn(train_data, train_data[item])
                lrd_others = lrd_others + self.local_reachability_density(train_data)
            # print('lof of',i, ': ', lrd_others / (3*lrd))
            if (lrd_others / (self.k * lrd)) > 1.2:
                resultArray[i] = -1
                # resultArray[i] = lrd_others / (self.k * lrd)
            else:
                resultArray[i] = 1
                # resultArray[i] = lrd_others / (self.k * lrd)
        return resultArray

    def knn(self, train_data, x):
        dist_nbr = np.zeros(shape=self.n_samples+1, dtype=float)
        for j in range(self.n_samples+1):
            dist_nbr[j] = np.sqrt(np.sum(np.square(train_data[j] - x)))  # Eucledian Distance
        ind_nbr = self.sparse_argsort(dist_nbr)[:self.k]
        return dist_nbr, ind_nbr

    def local_reachability_density(self, train_data):
        avg_RD = 0
        t = 0
        for j in range(self.k-1):
            r = self.dist_nbr[self.ind_nbr[j]]
            dist_nbr, ind_nbr = self.knn(train_data, train_data[self.ind_nbr[j]])
            t = dist_nbr[ind_nbr[self.k-1]]
            avg_RD = avg_RD + max(r, t)
        return self.k/avg_RD









    # def local_reachability_density(self, i):
    #     # for i in range(self.n_samples):
    #     avg_RD = 0
    #     for j in range(self.k):
    #             avg_RD = avg_RD + max(self.dist_nbr[i][self.ind_nbr[i][j]], self.dist_nbr[self.ind_nbr[i][j]][self.ind_nbr[self.ind_nbr[i][j]][self.k-1]])
    #
    #     return self.k/avg_RD


    # def fit(self, X):
    #     old = 0
    #     X = np.array(X)
    #
    #     self.dist_nbr, self.ind_nbr = self.knn_fit(X)
    #     for i in range(self.n_samples):
    #         self.lrd = np.append(self.lrd, self.local_reachability_density(i))
    #
    #     for i in range(self.n_samples):
    #         sum_lrd = 0
    #         for j in self.ind_nbr[i]:
    #             sum_lrd = sum_lrd + self.lrd[j]
    #         self.lof = sum_lrd / (self.k * self.lrd[i])
    #         if self.lof > old:
    #             self.cutoff = self.lof
    #             old = self.lof
    #     print(self.cutoff)
    #     return self

    # def knn(self, X):
    #     # KNN part
    #     # dist_nbr = [[None] * self.n_samples] * self.n_samples
    #     dist_nbr = np.full(shape=(self.n_samples, self.n_samples), fill_value=100, dtype=float)
    #     for i in range(self.n_samples-1):
    #         for j in range(i+1, self.n_samples):
    #             dist_nbr[j][i] = dist_nbr[i][j] = np.sqrt(np.sum(np.square(X[i] - X[j])))  # Eucledian Distance
    #     # dist_nbr = np.sort(dist)[:, :self.k]
    #     ind_nbr = np.argsort(dist_nbr)[:, :self.k]
    #     return dist_nbr, ind_nbr



