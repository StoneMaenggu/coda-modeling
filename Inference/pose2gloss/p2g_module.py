import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import os
import numpy as np
from .timeseries_GCN import TimeSeriesGCN
import pandas as pd

class P2G_Module:
    def __init__(self,checkpoint_path, db_path, n=5, stride=3, time_window=45):
        self.n = n
        self.stride = stride
        self.time_window = time_window
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')

        print(self.device)
        
        # self.device = 'cpu'

        if db_path is not None:
            self.feature_db = pd.read_csv(db_path)
            self.feature_vectors = torch.tensor(self.feature_db.iloc[:,2:].values)
            self.gloss2id = {g:i for i, g in enumerate(self.feature_db.gloss.unique())}
            self.id2gloss = {v:k for k,v in self.gloss2id.items()}

            self.feature_label = self.feature_db.gloss.apply(lambda x: self.gloss2id[x]).values
        else:
            
            self.feature_db = pd.DataFrame(np.eye(256))

        # self.model = TimeSeriesCNN(input_channels=68,
        #                            num_filters=256, 
        #                            kernel_size=3, 
        #                            embedding_dim=32)
        self.model = TimeSeriesGCN(input_channels=68,
                                   num_filters=128, 
                                   kernel_size=3, 
                                   embedding_dim=32)
        self.hand_adj = np.array([[4,5],
                                  [4,7],
                                  [4,10],
                                  [4,13],
                                  [4,16],
                                  [5,6],
                                  [7,8],
                                  [8,9],
                                  [10,11],
                                  [11,12],
                                  [13,14],
                                  [14,15],
                                  [16,17],
                                  [17,18]])-4
        self.pose_matrix = [[0,2],
                            [0,1],
                            [2,3],]
        self.adjacent_matrix = np.concatenate([self.pose_matrix, self.hand_adj+len(self.pose_matrix),self.hand_adj+len(self.pose_matrix)+len(self.hand_adj)],0).T
        self.adjacent_matrix = torch.tensor(self.adjacent_matrix).to(self.device)
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, file_name):
        ckpt = torch.load(file_name)
        self.model.load_state_dict(ckpt['model'])
        print(f'load model from {file_name}')
    
    @torch.no_grad()
    def get_features(self,pose):
        pose = self.split_pose_seq2batch(pose, self.stride, self.time_window)
        pose = self.preprocessing(pose)
        pose = torch.tensor(pose).to(self.device)
        batch_size, seq_len, n_feat, n_dim = pose.shape
        pose = pose.reshape(batch_size, seq_len, n_feat*n_dim)
        pose = pose.permute(0,2,1).to(self.device).to(torch.float32)

        x, feat = self.model(pose,self.adjacent_matrix)
        return feat.to('cpu')
    def predict(self, pose):
        feat = self.get_features(pose)
        return self.predict_class_based_on_knn(feat, self.n)
    
    def preprocessing(self, pose):
        return pose
        
    # def feat2gloss(self, feat, n):
        
    #     batch_size = self.feature_vectors.shape[0]
        
    #     # Step 1: Normalize the feature vectors
    #     normalized_features = F.normalize(self.feature_vectors, p=2, dim=1)
        
    #     # Step 2: Compute pairwise cosine similarity between feature vectors
    #     similarity_matrix = torch.matmul(normalized_features, normalized_features.T)
        
    #     # Step 3: Get the indices of the n most similar neighbors for each feature vector
    #     _, nearest_indices = torch.topk(similarity_matrix, k=n+1, largest=True)
    #     # Exclude the first index since it is the feature vector itself
    #     nearest_indices = nearest_indices[:, 1:]
        
    #     # Step 4: Get the classes of the n most similar neighbors
    #     nearest_classes = self.feature_idx[nearest_indices]
        
    #     # Step 5: Determine the most common class among the n nearest neighbors
    #     predicted_classes = torch.zeros(batch_size, dtype=torch.int64)
    #     for i in range(batch_size):
    #         # Get the most common class
    #         predicted_classes[i] = torch.mode(nearest_classes[i])[0]
        
    #     return predicted_classes
        
    def predict_class_based_on_knn(self, feature_vectors, n):
        """
        Predict the class of each feature vector based on the majority class of its n nearest neighbors.
        
        :param feature_vectors: Tensor of shape (batch_size, feature_dim)
        :param feature_idx: Tensor of shape (batch_size,) containing class indices for each feature vector
        :param n: Number of nearest neighbors to consider
        :return: predicted_classes: Tensor of shape (batch_size,) containing predicted classes
        """
        batch_size = feature_vectors.shape[0]

        distances = torch.cdist(feature_vectors, self.feature_vectors.to(torch.float32), p=2)
        _, nearest_indices = torch.topk(distances, k=n+1, largest=False)
        nearest_indices = nearest_indices[:, 1:]
        
        # Step 3: Get the classes of the n nearest neighbors
        nearest_classes = torch.tensor(self.feature_label[nearest_indices])
        
        # Step 4: Determine the most common class among the n nearest neighbors
        predicted_classes = torch.zeros(batch_size, dtype=torch.int64)
        ret = []
        for i in range(batch_size):
            # Get the most common class
            gloss_id = torch.mode(nearest_classes[i])[0]
            predicted_classes[i] = gloss_id
            ret.append(self.id2gloss[gloss_id.item()])
        return ' '.join(ret)
    
    def split_pose_seq2batch(self, pose_seq, stride, time_window):
        seq_len, n_channel, n_dim = pose_seq.shape
        last_id = seq_len-time_window
        start_idx = np.arange(0,last_id+1,stride)
        pose_batch = np.stack([pose_seq[s_id:s_id+time_window] for s_id in start_idx])
        return pose_batch

if __name__ == '__main__':
    pose = np.random.randn(100,34,2)

    # init
    p2g = P2G_Module(checkpoint_path='/home/horang1804/Dolmaggu/coda-modeling/Inference/pose2gloss/checkpoints/epoch_400.pt',
                     db_path='/home/horang1804/Dolmaggu/coda-modeling/Inference/gloss_db.csv')

    # predict
    gloss = p2g.predict(pose)

    # check
    '''
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    feat = p2g.feature_vectors
    pca = PCA(n_components=2) # 주성분을 몇개로 할지 결정
    pc = pca.fit_transform(feat)

    for i,gloss in p2g.id2gloss.items():
        idx = p2g.feature_label==i
        plt.scatter(pc[idx,0],pc[idx,1],s=100,label=gloss)
    plt.legend()
    plt.show()
    '''


    print('='*10,'gloss','='*10)
    print(gloss)


