import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import os
import numpy as np
from timeseries_CNN import TimeSeriesCNN
import pandas as pd

class P2G_Module:
    def __init__(self,checkpoint_path, db_path):
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')
        
        if db_path is not None:
            self.feature_db = pd.read_csv(db_path)
        else:
            self.feature_db = pd.DataFrame()

        self.model = TimeSeriesCNN(input_channels=76,
                                   num_filters=64, 
                                   kernel_size=3, 
                                   embedding_dim=256)
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, file_name):
        ckpt = torch.load(file_name)
        self.model.load_state_dict(ckpt['model'])
        print(f'load model from {file_name}')
    
    @torch.no_grad()
    def predict(self,pose):
        pose = torch.tensor(pose)
        seq_len, n_feat, n_dim = pose.shape
        pose = pose.reshape(seq_len, n_feat*n_dim)
        pose = pose.permute(1,0).unsqueeze(0).to(self.device).to(torch.float32)

        feat = self.model(pose)

        return feat

        

if __name__ == '__main__':


    p2g = P2G_Module(checkpoint_path=None,
                     db_path=None)

    pose = np.zeros((45,38,2))
    p2g.predict(pose)

