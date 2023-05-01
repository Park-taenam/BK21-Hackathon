# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

from DNN_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import warnings
warnings.filterwarnings('ignore')
# %%
class Dataset(data_utils.Dataset):
    # for dictionary batch
    def __init__(self, X, y):
        self.X = X
        self.y = y
   
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'y': self.y[idx]}
   
    def __len__(self):
        return len(self.X)

class MLPRegressor(nn.Module):
    def __init__(self):
        super(MLPRegressor, self).__init__()
        h1 = nn.Linear(len(X_train.columns), 64)
        h2 = nn.Linear(64, 32)
        h3 = nn.Linear(32, 1)
        self.hidden = nn.Sequential(
            h1,
            nn.ReLU(),
            h2,
            nn.ReLU(),
            h3,
        )
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()
        
    def forward(self, x):
        o = self.hidden(x)
        return o.view(-1)

def training(plant_name, X_train, y_train, X_test, y_test):
    trn_X = torch.from_numpy(X_train.astype(float).values).float()
    trn_y = torch.from_numpy(y_train.astype(float).values).float()

    val_X = torch.from_numpy(X_test.astype(float).values).float()
    val_y = torch.from_numpy(y_test.astype(float).values).float()

    batch_size = len(val_y)

    trn = Dataset(trn_X, trn_y)
    trn_loader = data_utils.DataLoader(trn, batch_size=batch_size, shuffle=True)

    val = Dataset(val_X, val_y)
    val_loader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=False)

    ## Train model
    model = MLPRegressor().cuda()

    criterion = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 100
    num_batches = 120

    trn_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        trn_loss_summary = 0.0
        for i, trn in enumerate(trn_loader):
            trn_X, trn_y = trn['X'], trn['y']
            if torch.cuda.is_available():
                trn_X, trn_y = trn_X.cuda(), trn_y.cuda()
            optimizer.zero_grad()
            trn_pred = model(trn_X)
            trn_loss = criterion(trn_pred, trn_y)
            trn_loss.backward()
            optimizer.step()
            
            trn_loss_summary += trn_loss
            
            if (i+1) % 15 == 0:
                with torch.no_grad():
                    val_loss_summary = 0.0
                    for j, val in enumerate(val_loader):
                        val_X, val_y = val['X'], val['y']
                        if torch.cuda.is_available():
                            val_X, val_y = val_X.cuda(), val_y.cuda()
                        val_pred = model(val_X)
                        val_loss = criterion(val_pred, val_y)
                        val_loss_summary += val_loss
                    
                # print("epoch: {}/{} | step: {}/{} | trn_loss: {:.4f} | val_loss: {:.4f}".format(
                #     epoch + 1, num_epochs, i+1, num_batches, (trn_loss_summary/15)**(1/2), (val_loss_summary/len(val_loader))**(1/2)
                # ))
                    
                trn_loss_list.append((trn_loss_summary/15)**(1/2))
                val_loss_list.append((val_loss_summary/len(val_loader))**(1/2))
                trn_loss_summary = 0.0
    print("{} - finish Training".format(plant_name))

    return model, val_X, val_y, trn_loss_list, val_loss_list

def prediction(inverter_list, model, val_X, val_y):
    pred_dict = {}

    for i, inverter in enumerate(inverter_list):
        val_pred = model(val_X.cuda()[120*(i):120*(i+1)])
        val_y_120 = val_y[120*(i):120*(i+1)]

        val_pred = [x.cpu().detach().numpy() for x in val_pred]
        val_pred = [max(0, x) for x in val_pred]
        val_y_120 = [x.cpu().detach().numpy() for x in val_y_120]

        pred_df = pd.DataFrame({'y_test':val_y_120, 'y_pred':val_pred})
        pred_dict[inverter] = pred_df
    
    return pred_dict

def evaluate(evaluate_df, plant_name, pred_dict):
    for key, value in pred_dict.items():
        plant_inverter = plant_name+ '-' + key
        hourly_comparison_plot(value, plant_inverter)
        error_smape , error_rmse, error_r2 = evaluation(value['y_test'], value['y_pred'])

        evaluate_df.loc[evaluate_df.shape[0]] = [plant_inverter, error_smape , error_rmse, error_r2]
    
    return evaluate_df
    
# %%
if __name__=="__main__":
    # Dataset
    data_dict = pd.read_pickle('./Data/final_data.pkl')

    evaluate_df = pd.DataFrame(columns = ['plant_inverter','SMAPE' , 'RMSE', 'R2'])
    for key, value in data_dict.items():
        plant_name = key
        df = value

        df = create_features(df) # sine encoding
        inverter_list, df_train, df_test = split_data_time(df, num=120) # Inverter 별로 120개
        X_train, y_train, X_test, y_test = feature_selection(df_train, df_test)
        X_train, X_test = one_hot_encoding(X_train, X_test)
        X_train, X_test = min_max_scaling(X_train, X_test)

        model, val_X, val_y, trn_loss_list, val_loss_list= training(plant_name, X_train, y_train, X_test, y_test)
        Loss_plot(plant_name, trn_loss_list, val_loss_list)

        pred_dict = prediction(inverter_list, model, val_X, val_y)
        evaluate_df = evaluate(evaluate_df, plant_name, pred_dict)

        evaluate_df.to_csv('./result/DNN_result.csv')
    print('Model Training Done')
# %%