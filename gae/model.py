"""
Code Reference:

PyTorch Geometric Team
Graph Classification Colab Notebook (PyTorch Geometric)
https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing 

"""

import torch
import numpy as np
import os
from torch.nn import Linear, BatchNorm1d, Conv2d, Softmax
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool, TopKPooling
from torch_sparse import SparseTensor


class GVAE_end_fusion(torch.nn.Module):
    def __init__(self, hidden_channels, hidden_channels2, hc, hc2, hc3,hc4,ratio_hc, roi_num):
        super(GVAE_end_fusion, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(roi_num-1, hidden_channels)
        self.conv11 = GCNConv(roi_num-1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.softmax_func=Softmax(dim=1)

        self.conv4 = Conv2d(1,1, (1,hc4))
        self.lin1 = Linear(hc, hc2)
        self.lin11 = Linear(hc, hc2)
        self.lin2 = Linear(hc2, hc2)
        self.lin3 = Linear(hc2, 2)
        self.lin33 = Linear(hc2, 2)
        self.lin6 = Linear(hc3, hc2)
        self.lin7 = Linear(hc2, 2)

        self.lin4 = Linear(roi_num, ratio_hc)      
        self.lin5 = Linear(ratio_hc, roi_num)

    def flatten(self, x, batch):
        
        seg = int(x.shape[0]/len(torch.unique(batch)))
        flatten_x = torch.empty(len(torch.unique(batch)),seg*x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            flatten_x[i] = torch.reshape(x[i*seg:(i+1)*seg], (1, seg*x.shape[1]))       
        return flatten_x     
    
    def forward(self, x, edge_index, edge_weight, roi_num, batch, device):
      
        lenn = 6670
        seg2 = int(edge_weight.shape[0]/len(torch.unique(batch)))
        seg = seg2-roi_num
        edge_tem = torch.empty(1,seg*len(torch.unique(batch)))    
        edge_weight1 = torch.empty(1,lenn*len(torch.unique(batch))) 
        edge_weight2 = torch.empty(1,lenn*len(torch.unique(batch))) 
        for i in range(0, len(torch.unique(batch))):
            edge_tem[0,i*seg:seg*(i+1)] = torch.reshape(edge_weight[seg2*i:(seg2*i+seg)], (1, seg))
            edge_weight1[0,i*lenn:(i+1)*lenn]=edge_tem[0,i*seg:i*seg+lenn]
            edge_weight2[0,i*lenn:(i+1)*lenn]=edge_tem[0,i*seg+lenn:(i+1)*seg]
        
        edge_weight1 = torch.squeeze(edge_weight1)
        edge_weight2 = torch.squeeze(edge_weight2)
        edge_weight1 = edge_weight1.to(device)
        edge_weight2 = edge_weight2.to(device)

    
        edge_tem = torch.empty(2,seg*len(torch.unique(batch)),dtype=torch.long) 
        edge_index1 = torch.empty(2,lenn*len(torch.unique(batch)),dtype=torch.long) 
        edge_index2 = torch.empty(2,lenn*len(torch.unique(batch)),dtype=torch.long) 
        for i in range(0, len(torch.unique(batch))):    
            edge_tem[0:2,i*seg:seg*(i+1)] = edge_index[0:2,seg2*i:(seg2*i+seg)]            
            edge_index1[0:2,i*lenn:(i+1)*lenn] = edge_tem[0:2,i*seg:i*seg+lenn]-roi_num*i
            edge_index2[0:2,i*lenn:(i+1)*lenn] = edge_tem[0:2,i*seg+lenn:(i+1)*seg]-roi_num*(i+1)
        edge_index1 = edge_index1.to(device)
        edge_index2 = edge_index2.to(device)

        seg2 = int(x.shape[0]/len(torch.unique(batch)))
        seg = int(seg2/2)
        node_fc = torch.empty(len(torch.unique(batch)), seg, x.shape[1])
        node_sc = torch.empty(len(torch.unique(batch)), seg, x.shape[1])
        for i in range(0, len(torch.unique(batch))):
            node_fc[i] = torch.reshape(x[i*seg2:(i*seg2+seg)], (1, seg, x.shape[1]))
            node_sc[i] = torch.reshape(x[(i*seg2+seg):(i+1)*seg2], (1, seg, x.shape[1]))
         
        node_fc = torch.reshape(node_fc,(len(torch.unique(batch))*seg,x.shape[1]))
        node_sc = torch.reshape(node_sc,(len(torch.unique(batch))*seg,x.shape[1]))
        node_fc = node_fc.to(device)
        node_sc = node_sc.to(device) 
        
        ## single branch
        ## for FC
        z1 = self.conv1(node_fc, edge_index1, edge_weight1)
        z1 = F.relu(z1)
        x1 = self.flatten(z1, batch)
        x1 = x1.to(device)   
        x1 = F.dropout(x1, p=0.5, training=self.training)
        x1 = self.lin1(x1)
        x1 = F.relu(x1)
        x1 = self.lin3(x1)
        x1 = self.softmax_func(x1)
       # print(x1)
       ## for SC
        z2 = self.conv11(node_sc, edge_index2, edge_weight2)
        z2 = F.relu(z2)
        x2 = self.flatten(z2, batch)
        x2 = x2.to(device)   
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = self.lin11(x2)
        x2 = F.relu(x2)
        x2 = self.lin33(x2)
        x2 = self.softmax_func(x2)
       # print(x2)
       
        # learning interactive weights
        all = torch.concat((z1,z2), 1)
        all = torch.reshape(all, ((len(torch.unique(batch))), 1, seg, all.shape[1])) ## can be really reshaped?
        all = all.to(device)
        all = self.conv4(all)
        all = torch.reshape(all, ((len(torch.unique(batch))), seg))
        all = self.lin4(all)
        all = F.relu(all)
        all = self.lin5(all)
        all = F.relu(all) ###importance

        ## update edge_weights of new graph with learned interactive weights
        seg2 = int(edge_weight.shape[0]/len(torch.unique(batch)))
        edge_tem = torch.empty(len(torch.unique(batch)),seg2)            
        for i in range(0, len(torch.unique(batch))):
            edge_tem[i] = torch.reshape(edge_weight[i*seg2:(i+1)*seg2], (1, seg2))
            edge_tem[i][-roi_num:] = all[i]          
        edge_tem = edge_tem.to(device)
        edge_tem = torch.reshape(edge_tem,(len(edge_weight), 1))
        edge_tem = torch.squeeze(edge_tem)
        
        # update node features of new graph with learned node features
        seg = int(z1.shape[0]/len(torch.unique(batch)))
        node_dim = z1.shape[1]
        x = torch.empty(len(torch.unique(batch))*seg*2, node_dim)
        for i in range(0, len(torch.unique(batch))):
            x[i*2*seg:i*2*seg+seg,:] = z1[seg*i:seg*(i+1),:]
            x[i*2*seg+seg:(i+1)*2*seg,:] = z2[seg*i:seg*(i+1),:]
        x = x.to(device)
        x = torch.reshape(x, (len(torch.unique(batch))*seg*2, node_dim))
        edge_tem_out = edge_tem
        x_out = x
        x = self.conv2(x, edge_index, edge_tem)
        x = F.relu(x)
        x = self.flatten(x, batch)
        x = x.to(device) 
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin6(x)
        x = F.relu(x)
       
        x = self.lin7(x) 
        x = self.softmax_func(x)

        out = []
        out.append(x)
        out.append(x1)
        out.append(x2)
        out.append(all) 
        return out

