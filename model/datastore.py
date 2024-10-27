import os
import pickle
import numpy as np

import faiss
import torch
import torch.nn as nn

class Datastore(nn.Module):
    def __init__(self, ckptdir, max_k):
        super(Datastore, self).__init__()
        with open(os.path.join(ckptdir, 'emb_array.pkl'), 'rb') as file:
            self.X_train = pickle.load(file)
        with open(os.path.join(ckptdir, 'label_array.pkl'), 'rb') as file:
            self.Y_train = pickle.load(file)
        with open(os.path.join(ckptdir, 'sys_array.pkl'), 'rb') as file:
            self.sys_train = pickle.load(file)

        self.sys_train = np.append(self.sys_train, 'non')
        self.max_k = max_k
        self.index = faiss.IndexFlatL2(768)  
        self.index.add(self.X_train)

        self.local_layer =  nn.Linear(1, 2)


    def knn_regression(self, queries, query_sys):
        D, I = self.index.search(queries, self.max_k)
        knn_values = self.Y_train[I]
        res_sys = self.sys_train[I]

        # print(query_sys, res_sys)
        local_bool = (res_sys == np.array(query_sys)[:, np.newaxis])
        local_bool = torch.BoolTensor(local_bool.astype(int)).to("cuda")

        tensorD = torch.from_numpy(D).requires_grad_(True).unsqueeze(-1).to("cuda")
        loc_dis = self.local_layer(tensorD)
        new_D = torch.where(local_bool, loc_dis[:,:,1], loc_dis[:,:,0])
        knn_values = torch.from_numpy(knn_values).to('cuda:0').requires_grad_(True)
        return new_D, knn_values

    def get_weighted_score_softmax(self, distances, scores):
        # distances: ndarray of shape (n_queries, n_neighbors)
        negative = -distances
        weights = torch.softmax(negative, dim=-1)
        res = torch.sum(torch.mul(weights, scores), axis = -1)
        return res
    
    def forward(self, queries, query_sys):
        # queries:[B, feature_dim]
        dists, scores = self.knn_regression(queries.cpu().detach().numpy(), query_sys)

        knns_scores = torch.zeros((queries.shape[0], self.max_k)).to("cuda")
        for i in range(self.max_k):
            knns_scores[:,i] = self.get_weighted_score_softmax(dists[:,0:i+1], scores[:,0:i+1])

        return dists, knns_scores
