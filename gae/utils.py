import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import coo_matrix
from sklearn import preprocessing
import heapq

def get_data(connpath, scorepath, feature = False):
    all_data = np.load(connpath)
    all_score = np.squeeze(np.load(scorepath))
    
    return all_data, all_score



def create_fusion_end_dataset(data, data2, score,indexx, roi_num, features = None):
    dataset_list = []
    n = data.shape[1]
    kk = 11 #preserve top 10% edges
    for i in range(len(data)):
       
        feature_matrix_ori = np.array(data[i])
        feature_matrix_ori_SC = np.array(data2[i])

        feature_matrix_ori2 =  feature_matrix_ori/np.max(feature_matrix_ori)
        feature_matrix_ori_SC2 =  feature_matrix_ori_SC/np.max(feature_matrix_ori_SC)
        feature_matrix = feature_matrix_ori2[~np.eye(feature_matrix_ori2.shape[0],dtype=bool)].reshape(feature_matrix_ori2.shape[0],-1)
        feature_matrix2 = feature_matrix_ori_SC2[~np.eye(feature_matrix_ori_SC2.shape[0],dtype=bool)].reshape(feature_matrix_ori_SC2.shape[0],-1)
        #print(np.max(node_SC),np.min(node_SC))
        feature_matrix_total = np.concatenate([feature_matrix, feature_matrix2], axis = 0)
        #print(feature_matrix_total.shape)

        edge_index_coo = np.triu_indices(roi_num, k=1)
        edge_index_coo2 = np.triu_indices(roi_num, k=1)
        edge_adj = np.zeros((roi_num, roi_num))
        #print(feature_matrix_ori2.shape)
        #print(edge_adj.shape)

        for ii in range(len(feature_matrix_ori2[1])):
            index_max = heapq.nlargest(kk, range(len(feature_matrix_ori[ii])),feature_matrix_ori[ii].take)
            edge_adj[ii][index_max] = feature_matrix_ori2[ii][index_max]
        edge_weight = edge_adj[edge_index_coo]
        edge_weight2 = feature_matrix_ori_SC2[edge_index_coo2]
        edge_weight2[edge_weight2<0.6] = 0
        edge_weight3 = np.ones(roi_num,int)
        edge_weight_total = np.concatenate([edge_weight, edge_weight2, edge_weight3], axis = 0)

        edge_index_coo = torch.tensor(edge_index_coo)
        edge_index_coo2 = torch.tensor(edge_index_coo2)
        edge_index_coo2 = edge_index_coo2+roi_num
        edge_index_coo4=np.reshape(np.arange(0,roi_num), (1,roi_num))
        edge_index_coo5=np.reshape(np.arange(roi_num,roi_num*2), (1,roi_num))
        edge_index_coo3 = torch.tensor(np.concatenate((edge_index_coo4,edge_index_coo5),axis=0))        
        edge_index_coo_total = torch.cat((edge_index_coo, edge_index_coo2, edge_index_coo3), 1)
         
        graph_data = Data(x = torch.tensor(feature_matrix_total, dtype = torch.float32), edge_index=edge_index_coo_total, edge_weight=torch.tensor(edge_weight_total, dtype = torch.float32),  y = torch.tensor(indexx[i])) 
        dataset_list.append(graph_data)
    return dataset_list
