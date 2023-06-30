"""
Local Outlier Factor(LOF) algorithm
-----------------------------------
This module includes the from scratch implementation of the LOF algorithm for anomaly detection. 
Any LOF score below 1.2 is considered as an anomaly. 

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance


class LOF_alg():
    """Local Outlier Factor anomaly detection algorithm with semi-supervised learning format.

    LOF uses k-Nearest Neighbours concept to find the nearest neighbouring points for calculation of LOF scores.
    
    Parameters
    ----------
    k : number of neighbours to consider

    Attributes
    ----------
    n_features : number of features
    n_samples : number of points
    n_samples_pr : number of samples during prediction
    n_features_pr : number of features duting prediction
    lrd : list of Local Reachability Distances
    train_data : list of training data-points
    ind_nbr : index of neighbouring points
    dist_nbr : distance of neighbouring points

    References
    ----------
    -- [1] Aggarwal, Charu C., and Charu C. Aggarwal. "Applications of Outlier Analysis." Outlier Analysis (2013): 373-400.

    """
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
        """Sorting function to sort nearest points to farthest
        Parameters
        ----------
        arr : {n-dimensional array} of shape (n_samples, n_features) to be ascendingly sorted

        Returns
        -------
        indices : ascendingly sorted indeces of arr
        """
        indices = np.nonzero(arr)[0]
        return indices[np.argsort(arr[indices])]

    def fit(self, X):
        """Fit the LOF algorithm with training data or storing the data space on which the new point to be predicted 
        is evaluated against.
        
        Parameters
        ----------
        X : {n-dimensional array} of shape (n_samples, n_features) 

        Returns
        -------
        self : LOF_alg
            The fitted LOF model
        """
        X = np.array(X)
        self.train_data = X
        self.n_samples, self.n_features = X.shape[0], X.shape[1]  # number of rows and columns

    def predict(self, Y):
        """Predict the labels (1 normal , -1 anomalous) of X according to LOF algorithm
        Parameters
        ----------
        Y : {n-dimensional array} of shape (n_samples, n_features) for prediction

        Returns
        ---------
        resultArray : ndarray of shape (n_samples)
            Returns -1 for anomalies and +1 for normal cases.
        """
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
        """Function to find the k-Nearest Neighbours
        Parameters
        ----------
        train_data : {n-dimensional array} of shape (n_samples, n_features) of training data
        x : {n-dimensional array} of shape (n_samples, n_features) of reference data-point whose neighbours are to be found

        Returns
        -------
        dist_nbr : distance each neighbours in ascending order from x
        ind_nbr : index of neighbours in ascending order from x

        Returns the distances and index of the k-Nearest Neighbours 
        """
        dist_nbr = np.zeros(shape=self.n_samples+1, dtype=float)
        for j in range(self.n_samples+1):
            dist_nbr[j] = np.sqrt(np.sum(np.square(train_data[j] - x)))  # Eucledian Distance
        ind_nbr = self.sparse_argsort(dist_nbr)[:self.k]
        return dist_nbr, ind_nbr

    def local_reachability_density(self, train_data):
        """Finds the Local Reachability Density for LOF value calculation 
        
        Parameters
        ----------
        train_data : {n-dimensional array} of shape (n_samples, n_features) of training datas

        Returns
        -------
        Returns the local reachability density
        """
        avg_RD = 0
        t = 0
        for j in range(self.k-1):
            r = self.dist_nbr[self.ind_nbr[j]]
            dist_nbr, ind_nbr = self.knn(train_data, train_data[self.ind_nbr[j]])
            t = dist_nbr[ind_nbr[self.k-1]]
            avg_RD = avg_RD + max(r, t)
        return self.k/avg_RD
