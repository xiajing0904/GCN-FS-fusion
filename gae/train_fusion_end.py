"""
Code Reference:

PyTorch Geometric Team
Graph Classification Colab Notebook (PyTorch Geometric)
https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing 

"""

import torch
import numpy as np
import os
import scipy
import pandas as pd
import csv
import time
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import classification_report

from utils import get_data, create_fusion_end_dataset
from torch_geometric.data import DataLoader
from model import GVAE_end_fusion
import torch.nn
from torch.utils.data.dataloader import default_collate
from torcheval.metrics import R2Score
import torch.nn.functional as F
#os.environ["CUDA_VISIBLE_DEVICES"]="3"


def L2Loss(model, alpha):
    l2_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5*alpha * torch.sum(torch.pow(parma, 2)))
    return l2_loss

def L1Loss(model, alpha):
    l1_loss = torch.tensor(0.0, requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + (0.5*alpha * torch.sum(torch.abs(parma)))
    return l1_loss

def write_attention(filename, train_attention, fold):
    #train_attention = np.float32(train_attention.cpu().detach().numpy())
    filename1 = filename + '/train_attention' + str(fold) + ".npy" 
    np.save(filename1, train_attention)
   
    mean_train = np.mean(train_attention, axis=0)
    #print(mean_train.shape)
    filename2 = filename + '/train_mean_attention' + str(fold) + ".npy"
    np.save(filename2, mean_train) 
   
def write_attention_test(filename, test_all):
    filename1 = filename + '/test_attention.npy'
    np.save(filename1, test_all)
    mean_train = np.mean(test_all, axis=0)
    filename2 = filename + '/test_mean_attention.npy' 
    np.save(filename2, mean_train)
    
def train(model_to_train, train_dataset_loader, model_optimizer,test_score,device,roi_num):
    model_to_train.train()
    for data in train_dataset_loader:  # Iterate in batches over the training dataset.
        data.x, data.edge_index, data.edge_weight, data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device) 
        out_all = model_to_train(data.x.float(), data.edge_index, data.edge_weight, roi_num, data.batch, device)  # Perform a single forward pass.
        out = out_all[0]   
        out1 = out_all[1]
        out2 = out_all[2]
    
        test_tem = test_score[data.y]
        w = torch.Tensor([1, 1])
        w = w.to(device)
        pre_loss = F.cross_entropy(out, test_tem, w) 
        pre_loss1 = F.cross_entropy(out1, test_tem, w)
        pre_loss2 = F.cross_entropy(out2, test_tem, w)
        loss =  pre_loss + L2Loss(model_to_train, 0.001) + pre_loss1 + pre_loss2 
        if loss == 'nan':
            break
        loss.backward()  # Deriving gradients.
        model_optimizer.step()  # Updating parameters based on gradients.
        model_optimizer.zero_grad()  # Clearing gradients.

def test(model, loader, test_score, device, roi_num):
    model.eval()

    pre_loss_sum = 0
    out_sum = torch.tensor(()).to(device)
    true_sum = torch.tensor(()).to(device)
    attention = torch.tensor(()).to(device)
    #correct = 0
    count = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.x, data.edge_index, data.edge_weight, data.batch, data.y = data.x.to(device), data.edge_index.to(device), data.edge_weight.to(device),data.batch.to(device),data.y.to(device) 
        out_all = model(data.x.float(), data.edge_index, data.edge_weight, roi_num, data.batch, device)
        test_tem = test_score[data.y]
        out = out_all[0]   
        A = out_all[3]
        
        w = torch.Tensor([1, 1])
        w = w.to(device)
        pre_loss = F.cross_entropy(out, test_tem, w, size_average=False) 
        
        out = torch.argmax(out, dim = 1)
        
        count = count+1
       
        out_sum = torch.cat((out_sum, out), dim=0) 
    
        pre_loss_sum = pre_loss_sum + pre_loss
        
        true_sum = torch.cat((true_sum, torch.reshape(test_tem.float(), out.shape)), dim=0)
        attention = torch.cat((attention, A), dim=0)
    truev = torch.reshape(true_sum,(len(out_sum),1))
    out = np.squeeze(np.asarray(out_sum.cpu().detach().numpy()))
    truev = np.squeeze(np.asarray(truev.cpu().detach().numpy())) 

    accu =  np.sum(out == truev)/truev.shape[0]
    attention = np.squeeze(np.asarray(attention.cpu().detach().numpy())) 
    pre_loss_sum = pre_loss_sum/count
    
    #print(attention.shape)     
    return pre_loss_sum, accu, out, truev, attention

#parameter setting
hidden_channels=64 #128 #64
hidden_channels2=64 #128 #64
roi_num = 116 
ratio_hc = 58
hc = hidden_channels*roi_num 
hc2 = 256 
hc3 = hidden_channels*roi_num*2
hc4 = 2*hidden_channels
epoch_num = 20
decay_rate = 0.5  
decay_step = 20
lr =0.0005
num_folds = 5
batch_size = 2
runnum = 'e1'
print("\n---------Starting to load Data---------\n")
dir =  os.path.dirname(os.getcwd())
fconnpath = dir + '/dataset_FC/Control/X.npy'  
sconnpath = dir + '/dataset_SC/Control/X.npy'  
fconnpath2 = dir +'/dataset_FC/Disease/X.npy'  
sconnpath2 = dir +'/dataset_SC/Disease/X.npy'  
labelpath1 = dir +'/dataset_SC/Control/Y.npy'  
labelpath2 = dir +'/dataset_SC/Disease/Y.npy'  


timefile = dir + '/results_fusion/GCN-fusion-'+runnum+'-'+str(int(time.time()))
os.mkdir(timefile)
total_score = timefile+'/total_score.csv'
Labelfile   = timefile+'/predict_train.csv'
Labelfile_test   = timefile+'/predict_test.csv'  
finalfile = timefile+'/final_result.csv'
finalfile2 = timefile+'/final_scores.csv'              

con_data_fc, label1 = get_data(fconnpath, labelpath1)
con_data_sc, label1 = get_data(sconnpath, labelpath1)
pd_data_fc, label2 = get_data(fconnpath2, labelpath2)
pd_data_sc, label2 = get_data(sconnpath2, labelpath2)

all_data_fc = np.concatenate((con_data_fc, pd_data_fc),axis=0)
all_data_sc = np.concatenate((con_data_sc, pd_data_sc),axis=0)
all_score= np.concatenate((label1, label2),axis=0)

num = all_data_fc.shape[0]
num1 = con_data_fc.shape[0]
num2 = pd_data_fc.shape[0]
print(num, num1, num2)
stratifiedKFolds = StratifiedKFold(n_splits = num_folds, shuffle = True)
print("\n--------Split and Data loaded-----------\n")
fold = 0
true_out = np.squeeze(np.array([[]]))
true_out_best = np.squeeze(np.array([[]]))
pred_out = np.squeeze(np.array([[]]))
pred_out1 = np.squeeze(np.array([[]]))
pred_out2 = np.squeeze(np.array([[]]))

pred_out_best = np.squeeze(np.array([[]]))
test_att_all = np.empty((1,roi_num))
test_corr_s = []
test_corr_s1 = []
test_corr_s2 = []

test_r2_s = []
test_rmse_s = []
test_mae_s = []


for X_train, X_test in stratifiedKFolds.split(all_data_fc, all_score): #kf.split(list(range(0,num))): #
    fold = fold+1
    model = GVAE_end_fusion(hidden_channels, hidden_channels2, hc, hc2, hc3, hc4, ratio_hc, roi_num)
    print("Model:\n\t",model)
    print(torch.cuda.is_available())
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(fold)
    model.to(device)
    
    train_data = all_data_fc[X_train]
    test_data = all_data_fc[X_test]
    train_data2 = all_data_sc[X_train]
    test_data2 = all_data_sc[X_test]
    train_score = all_score[X_train]
    test_score = all_score[X_test]
    #print(all_data.shape, all_score.shape)
    index_test = np.reshape(np.arange(0,len(X_test)),(len(X_test),1))
    index_train = np.reshape(np.arange(0,len(X_train)),(len(X_train),1))
    training_dataset = create_fusion_end_dataset(train_data, train_data2, train_score,index_train, roi_num)
    testing_dataset = create_fusion_end_dataset(test_data,test_data2, test_score,index_test, roi_num)
    train_score_input = torch.tensor(train_score).to(device)
    test_score_input = torch.tensor(test_score).to(device)
    
    train_loader = DataLoader(training_dataset, batch_size, shuffle = True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_loader = DataLoader(testing_dataset, batch_size, shuffle= True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(1, epoch_num):
        if epoch % decay_step == 0:
            for p in optimizer.param_groups:
                p['lr'] *= decay_rate
        train(model, train_loader, optimizer,train_score_input, device, roi_num)
        pre_loss_sum, accu, train_out, train_true, train_att  = test(model, train_loader,train_score_input, device, roi_num)
        pre_loss_sum2, accu2, test_out, test_true, test_att  = test(model, test_loader,test_score_input, device, roi_num)
        if epoch % 1 == 0:
            print(epoch)
            print(f'Epoch: {epoch:03d}, Train_loss: {pre_loss_sum:.4f}, Train_accu: {accu:.4f}')
            print(f'Epoch: {epoch:03d}, Test_loss: {pre_loss_sum2:.4f}, Test_accu: {accu2:.4f}')
            accu2 = torch.tensor(np.float32(accu2))
            
    write_attention(timefile, train_att, fold)
    df_predict = {'Predicted':train_out, 'Actual':train_true}
    df_predict = pd.DataFrame(data=df_predict, dtype=np.float32)
    df_predict.to_csv(Labelfile, mode='a+', header=True) 
    df_predict2 = {'Predicted':test_out, 'Actual':test_true}
    df_predict2 = pd.DataFrame(data=df_predict2, dtype=np.float32)
    df_predict2.to_csv(Labelfile_test, mode='a+', header=True)
    test_corr_s.append(accu2)
    pred_out_best = np.concatenate((pred_out_best, test_out), axis=0)
    true_out_best = np.concatenate((true_out_best, test_true), axis=0)
    test_att_all = np.concatenate((test_att_all,test_att), axis = 0)
test_att_all = test_att_all[1:(test_att_all.shape[0])]
write_attention_test(timefile, test_att_all)   
accu_best =  np.sum(pred_out_best == true_out_best)/true_out_best.shape[0]
report_best = classification_report(pred_out_best, true_out_best)
df_predict3 = {'Predicted':torch.tensor(pred_out_best), 'Actual':torch.tensor(true_out_best)}
df_predict3 = pd.DataFrame(data=df_predict3, dtype=np.float32)
df_predict3.to_csv(total_score, mode='a+', header=True)
print(report_best)
torch.cuda.empty_cache()


