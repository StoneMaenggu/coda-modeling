import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from collections import defaultdict
import os
import numpy as np
from tqdm import tqdm
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'
### i want to create function of triplet loss 


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Compute the pairwise distance between anchor-positive and anchor-negative
        positive_distance = F.pairwise_distance(anchor, positive, p=2)
        negative_distance = F.pairwise_distance(anchor, negative, p=2)

        # Compute the triplet loss
        loss = F.relu(positive_distance - negative_distance + self.margin)

        # Calculate the mean of the loss
        loss = loss.mean()

        # Return the loss and the individual distances for debugging purposes
        return loss, {'pos_dist': positive_distance.mean().item(), 
                      'neg_dist': negative_distance.mean().item()}
# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.5):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, pred, label):
#         pred = F.normalize(pred, p=2, dim=1)
#         cosine_similarity = torch.mm(pred, pred.t())
#         angular_distance = torch.acos(torch.clamp(cosine_similarity, -1.0, 1.0))
#         label_matrix = label.unsqueeze(0) == label.unsqueeze(1)
#         margin = 1.0
#         # positive_mask = label_matrix.float()
#         # negative_mask = 1.0 - positive_mask
#         positive_loss = label_matrix * angular_distance
#         negative_loss = (~label_matrix) * torch.relu(margin - angular_distance)
#         positive_loss = positive_loss.sum()
#         negative_loss = negative_loss.sum()
        
#         loss = (positive_loss + negative_loss)/(label_matrix.shape[0]**2)

#         # # Triplet loss
#         # loss = torch.relu(positive_distance - negative_distance + self.margin)
#         # loss = F.triplet_margin_loss(anchor,pos_neg[label==1],pos_neg[label==0] )
#         return loss, {'pos_loss':positive_loss.item(),
#                       'neg_loss':negative_loss.item()}

# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.5):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, pred, label):
#         # Normalize the embeddings to make it easier to calculate the distances
#         pred = F.normalize(pred, p=2, dim=1)

#         # Compute the pairwise distance matrix between all embeddings
#         distance_matrix = torch.cdist(pred, pred, p=2)

#         # Mask to filter out invalid triplet combinations
#         label_matrix = label.unsqueeze(1) == label.unsqueeze(0)  # Create a boolean mask for positive pairs
#         positive_mask = label_matrix.float()
#         negative_mask = 1.0 - label_matrix.float()  # Negative mask is the inverse of the positive mask

#         # For each anchor, we calculate the hardest positive and hardest negative
#         hardest_positive_dist = (distance_matrix * positive_mask).max(dim=1, keepdim=True)[0]
#         hardest_negative_dist = (distance_matrix + 1e6 * positive_mask).min(dim=1, keepdim=True)[0]

#         # Compute the triplet loss
#         loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

#         # Calculate the mean of the losses for the batch
#         loss = loss.mean()

#         # Calculate and return the average positive and negative distances as well
#         pos_loss = hardest_positive_dist.mean().item()
#         neg_loss = hardest_negative_dist.mean().item()

#         return loss, {'pos_loss': pos_loss, 'neg_loss': neg_loss}

# class TripletLoss(nn.Module):
#     def __init__(self, margin=0.5):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, pred, label):
#         # Generate triplets based on labels
#         anchor, positive, negative = self._get_triplets(pred, label)

#         # Calculate the Euclidean distances between anchor-positive and anchor-negative
#         positive_distance = F.pairwise_distance(anchor, positive, p=2)
#         negative_distance = F.pairwise_distance(anchor, negative, p=2)

#         # Calculate the triplet loss
#         loss = F.relu(positive_distance - negative_distance + self.margin)

#         # Calculate the mean of the losses for the batch
#         loss = loss.mean()

#         # Return the loss and individual distances
#         return loss, {'pos_loss': positive_distance.mean().item(),
#                       'neg_loss': negative_distance.mean().item()}

#     def _get_triplets(self, pred, label):
#         """
#         This function generates triplets (anchor, positive, negative) from the predictions and labels.
#         """
#         triplets = []
#         unique_labels = torch.unique(label)

#         for lbl in unique_labels:
#             # Get all indices where label == lbl (positive examples)
#             positive_indices = torch.where(label == lbl)[0]
#             # Get all indices where label != lbl (negative examples)
#             negative_indices = torch.where(label != lbl)[0]

#             # Generate all possible anchor-positive pairs
#             for anchor_idx in positive_indices:
#                 for positive_idx in positive_indices:
#                     if anchor_idx == positive_idx:
#                         continue

#                     # Randomly select a negative example
#                     negative_idx = negative_indices[torch.randint(len(negative_indices), (1,)).item()]

#                     # Append the triplet to the list
#                     triplets.append((pred[anchor_idx], pred[positive_idx], pred[negative_idx]))

#         # Stack the triplets to return them as tensors
#         anchors = torch.stack([triplet[0] for triplet in triplets])
#         positives = torch.stack([triplet[1] for triplet in triplets])
#         negatives = torch.stack([triplet[2] for triplet in triplets])

#         return anchors, positives, negatives

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        pos_euclidean_distance = F.pairwise_distance(anchor, pos)
        neg_euclidean_distance = F.pairwise_distance(anchor, neg)
        pos_loss = torch.mean(torch.pow(pos_euclidean_distance, 2)) 
        neg_loss = torch.mean(torch.pow(torch.clamp(self.margin - neg_euclidean_distance, min=0.0), 2))
        loss_contrastive = (pos_loss + neg_loss)/2
        return loss_contrastive, {'pos_dis':pos_euclidean_distance,
                                  'neg_dis':neg_euclidean_distance,
                                  'pos_loss':pos_loss.item(),
                                  'neg_loss':neg_loss.item()}

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features, features.t())
        logits = similarity_matrix / self.temperature
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.t()).float().to(logits.device)
        mask = mask - torch.eye(mask.size(0)).to(mask.device)
        log_prob = F.log_softmax(logits, dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()

        return loss

class SignLang_contrastive_trainer:
    def __init__(self, config, model):
        self.task = config.task
        self.train_id = config.train_id
        self.EPOCHS = config.epochs
        self.lr = config.lr
        self.base_path = config.base_path
        self.use_wandb = config.use_wandb
        self.save_freq = config.save_freq
        self.resume = config.resume
        self.device = config.device
        self.batch_size = config.batch_size
        self.w_con = config.w_con
        self.w_ce = config.w_ce
        self.num_pos_samples = config.num_pos_samples 
        self.num_neg_samples = config.num_neg_samples 
        # self.num_gloss = config.num_gloss

        self.criterion = TripletLoss()

        self.model = model

        self.logger = defaultdict(list)
        
        import pandas as pd
        self.test_db = pd.read_csv('/home/horang1804/Dolmaggu/coda-modeling/Inference/gloss_db.csv')

        self.test_db = self.test_db[['pose_path','gloss']]
        self.word2id = {g:i for i, g in enumerate(self.test_db.gloss.unique())}
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.test_db.loc[:,'id'] = self.test_db['gloss'].apply(lambda x: self.word2id[x])
    def save_model(self, file_name, epoch):
        ckpt = {
            'epoch':epoch,
            'model':self.model.state_dict(),
            'optimizer':self.optim.state_dict()
                }
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        torch.save(ckpt, file_name)
        print(f'save checkpoint to {file_name}')

    def load_model(self, file_name, use_optim=True):
        ckpt = torch.load(file_name)
        self.model.load_state_dict(ckpt['model'])
        if use_optim:
            self.optim.load_state_dict(ckpt['optimizer'])
        print(f'load model from {file_name}')
        return ckpt['epoch']

    def train(self,train_loader, val_loader=None):

        ##### init Training #####
        self.optim = optim.Adam(self.model.parameters(),
                                lr=self.lr,
                                betas=(0.9,0.999))
        start_epoch = 1
        
        if self.resume is None:
            self.model.init_weights()
        else:
            start_epoch = self.load_model(self.resume)
                
        self.model.to(self.device)
        self.adjacent_matrix = torch.tensor(train_loader.dataset.adjacent_matrix).to(self.device)
        ##### Training loop #####
        for e in range(start_epoch, self.EPOCHS+1):
            ### Training ###
            losses, train_metric = self.train_one_epoch(train_loader)

            # log training result
            self.logger['epoch'].append(e)
            self.logger['learning_rate'].append(self.optim.state_dict()['param_groups'][0]['lr'])
            for k, v in losses.items():
                self.logger['train_'+k].append(v.item())
            for k, v in train_metric.items():
                self.logger['train_'+k].append(v.item())

            ### Validation ###
            # if val_loader is not None:
            val_loss, val_metric = self.test(val_loader)
            # log validation result
            for k, v in val_loss.items():
                self.logger['val_'+k].append(v)
            for k, v in val_metric.items():
                self.logger['val_'+k].append(v)
            
            print(f'Epoch: {e:04d}', end='')
            for k,v in self.logger.items():
                vv = v[-1]
                if type(vv)==int and k!='epoch':
                    print(f' | {k}:{vv:03d}', end='')
                else:
                    print(f' | {k}:{vv:00.4f}', end='')
            print()
            
            if self.use_wandb:
                wandb.log({k+'_':float(v[-1]) for i, (k,v) in enumerate(self.logger.items()) if i>1})
            # save model
            if e%self.save_freq == 0:
                file_name = os.path.join(self.base_path, 'checkpoint',self.task,str(self.train_id),f'epoch_{e}.pt')
                self.save_model(file_name, e)
            file_name = os.path.join(self.base_path, 'checkpoint',self.task,str(self.train_id),f'epoch_last.pt')
            self.save_model(file_name, e)

        return self.model, self.logger
    
    def train_one_epoch(self, train_loader):
        self.model.train()
        losses = defaultdict(list)
        metrics = defaultdict(list)
        # self.adjacent_label = train_loader.dataset.adjacent_label
        for iter, (anchor_batch, pos_batch, neg_batch) in enumerate(tqdm(train_loader),1):
            # pose_batch = pose_batch.to(self.device)
            # label = label.to(self.device)
            
            anchor_batch = anchor_batch.to(self.device)
            pos_batch = pos_batch.to(self.device)
            neg_batch = neg_batch.to(self.device)

            num_batch, seq_len, n_channel, n_dim = anchor_batch.shape

            anchor_batch = anchor_batch.reshape(num_batch, seq_len, n_channel*n_dim).permute(0,2,1)
            pos_batch = pos_batch.reshape(num_batch, seq_len, n_channel*n_dim).permute(0,2,1)
            neg_batch = neg_batch.reshape(num_batch, seq_len, n_channel*n_dim).permute(0,2,1)

            pose_batch = torch.cat([anchor_batch,pos_batch,neg_batch])

            pred, feature_vector = self.model(pose_batch, self.adjacent_matrix)
            
            anchor_feat = feature_vector[:num_batch]
            pos_feat = feature_vector[num_batch:num_batch*2]
            neg_feat = feature_vector[num_batch*2:]

            if self.w_con != 0:
                loss_contrastive,loss_info = self.criterion(anchor_feat, pos_feat, neg_feat)
            # loss_ce = F.cross_entropy(pred, label)

            '''
            #---------------------------------------
            # ref_feature = torch.cat([feature_vector[:batch_size],feature_vector[:batch_size]],0)
            anc_feature = feature_vector[:num_anc]
            pos_neg_feature = feature_vector[num_anc:]
            # pos_feature = con_feature[pos_neg_label==1]
            # neg_feature = con_feature[pos_neg_label==0]
            # label = torch.cat([torch.ones((batch_size,),dtype=torch.float32),torch.zeros((batch_size,),dtype=torch.float32)],0).to(self.device)

            # loss_contrastive = self.criterion(ref_feature,con_feature,label)
            loss_contrastive = self.criterion(anc_feature,pos_neg_feature,pos_neg_label)
            '''

            if self.w_con != 0:
                loss = self.w_con * loss_contrastive#+ self.w_ce * loss_ce
            else:
                loss = self.w_ce #* loss_ce
            # update weights
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            losses['loss'].append(loss.item())
            if self.w_con != 0:
                losses['loss_con'].append(loss_contrastive.item())
            # losses['loss_ce'].append(loss_ce.item())
            losses['pos_dist'].append(loss_info['pos_dist'])
            losses['neg_dist'].append(loss_info['neg_dist'])

            # # calc metric
            # with torch.no_grad():
                # metrics['acc'].append(torch.mean((pred.argmax(1)==label)*1.).item())
                # metrics['pos_dist'].append(_)
        for k,v in losses.items():
            losses[k] = np.mean(v)

        for k,v in metrics.items():
            metrics[k] = np.mean(v)

        return losses, metrics
    
    
    @torch.no_grad()
    def test(self,test_loader, resume=None, verbose=0):
        if resume is not None:
            epoch = self.load_model(resume,use_optim=False)
        self.model.eval()
        self.model.to(self.device)
        losses = defaultdict(list)
        metrics = defaultdict(list)
        valid_features = []
        valid_cls = []

        pose_batch = []
        for idx, (pose_path, gloss, g_id) in self.test_db.iterrows():
            pose = np.load(pose_path)
            pose_batch.append(pose)

        pose_batch = np.stack(pose_batch)
        label  = self.test_db.id.values

        pose_batch = torch.tensor(pose_batch, dtype=torch.float32, device=self.device)
        label = torch.tensor(label, dtype=torch.float32)

            
        batch_size, seq_len, n_channel, n_dim = pose_batch.shape
        
        pose_batch = pose_batch.reshape((batch_size,seq_len,n_dim*n_channel)).permute(0,2,1)
        
        pred, feature_vector = self.model(pose_batch,self.adjacent_matrix)
        
        # loss_ce = F.cross_entropy(pred, label)
        # losses['loss_ce'].append(loss_ce.item())

        # with torch.no_grad():
        #     metrics['cls_acc'].append(torch.mean((pred.argmax(1)==label)*1.).item())

        feature_vector = feature_vector.to('cpu')
        pred_cls = self.predict_class_based_on_knn(feature_vector, label, 5)
        #     # calc metric
        #     gloss_mask = (gloss_batch.argmax(2)>2).sum(0)
        #     if gloss_mask.min() ==0:
        #         gloss_mask[gloss_mask==0]=1
        #     metrics['acc'] += torch.sum(((pred_gloss_seq.argmax(2) ==gloss_batch.argmax(2))*(gloss_batch.argmax(2)!=0)).sum(0)/(gloss_batch.argmax(2)!=0).sum(0)).item()
        #     metrics['g_acc'] += torch.sum(((pred_gloss_seq.argmax(2) ==gloss_batch.argmax(2))*(gloss_batch.argmax(2)>2)).sum(0)/gloss_mask).item()

        metrics['acc']=torch.sum(pred_cls==label)/len(pred_cls)
        if verbose:
            print(f'Epoch: {epoch:04d}', end='')
            for k,v in losses.items():
                if type(v)==float:
                    print(f' | {k}:{v:00.4f}', end='')
            for k,v in metrics.items():
                if type(v)==float:
                    print(f' | {k}:{v:00.4f}', end='')
            print()

        return losses, metrics
    '''@torch.no_grad()
    def test(self,test_loader, resume=None, verbose=0):
        if resume is not None:
            epoch = self.load_model(resume,use_optim=False)
        self.model.eval()
        losses = defaultdict(list)
        metrics = defaultdict(list)
        valid_features = []
        valid_cls = []
        for iter, (pose_batch, label) in enumerate(tqdm(test_loader),1):
            pose_batch = pose_batch.to(self.device)
            label  = label.to(self.device)
            
            batch_size, seq_len, n_channel, n_dim = pose_batch.shape
            
            pose_batch = pose_batch.reshape((batch_size,seq_len,n_dim*n_channel)).permute(0,2,1)
            
            pred, feature_vector = self.model(pose_batch)
            
            # loss_ce = F.cross_entropy(pred, label)
            # losses['loss_ce'].append(loss_ce.item())

            # with torch.no_grad():
            #     metrics['cls_acc'].append(torch.mean((pred.argmax(1)==label)*1.).item())


            valid_features.append(feature_vector.to('cpu'))
            valid_cls.append(label)
        valid_features = torch.cat(valid_features, 0)
        valid_cls = torch.cat(valid_cls, 0)
        valid_cls = valid_cls.to('cpu')
        pred_cls = self.predict_class_based_on_knn(valid_features, valid_cls, 5)
        #     # calc metric
        #     gloss_mask = (gloss_batch.argmax(2)>2).sum(0)
        #     if gloss_mask.min() ==0:
        #         gloss_mask[gloss_mask==0]=1
        #     metrics['acc'] += torch.sum(((pred_gloss_seq.argmax(2) ==gloss_batch.argmax(2))*(gloss_batch.argmax(2)!=0)).sum(0)/(gloss_batch.argmax(2)!=0).sum(0)).item()
        #     metrics['g_acc'] += torch.sum(((pred_gloss_seq.argmax(2) ==gloss_batch.argmax(2))*(gloss_batch.argmax(2)>2)).sum(0)/gloss_mask).item()

        metrics['acc']=torch.sum(pred_cls==valid_cls)/len(pred_cls)
        # metrics['cls_acc'] = np.mean(metrics['cls_acc'])
        losses = {k:np.mean(v) for k,v in losses.items()}
        if verbose:
            print(f'Epoch: {epoch:04d}', end='')
            for k,v in losses.items():
                if type(v)==float:
                    print(f' | {k}:{v:00.4f}', end='')
            for k,v in metrics.items():
                if type(v)==float:
                    print(f' | {k}:{v:00.4f}', end='')
            print()

        return losses, metrics'''
    def predict_class_based_on_knn(self, feature_vectors, feature_idx, n):
        """
        Predict the class of each feature vector based on the majority class of its n nearest neighbors.
        
        :param feature_vectors: Tensor of shape (batch_size, feature_dim)
        :param feature_idx: Tensor of shape (batch_size,) containing class indices for each feature vector
        :param n: Number of nearest neighbors to consider
        :return: predicted_classes: Tensor of shape (batch_size,) containing predicted classes
        """
        batch_size = feature_vectors.shape[0]
        
        # Step 1: Normalize the feature vectors
        normalized_features = F.normalize(feature_vectors, p=2, dim=1)
        
        # Step 2: Compute pairwise cosine similarity between feature vectors
        similarity_matrix = torch.matmul(normalized_features, normalized_features.T)
        
        # Step 3: Get the indices of the n most similar neighbors for each feature vector
        _, nearest_indices = torch.topk(similarity_matrix, k=n+1, largest=True)
        # Exclude the first index since it is the feature vector itself
        nearest_indices = nearest_indices[:, 1:]
        
        # Step 4: Get the classes of the n most similar neighbors
        nearest_classes = feature_idx[nearest_indices]
        
        # Step 5: Determine the most common class among the n nearest neighbors
        predicted_classes = torch.zeros(batch_size, dtype=torch.int64)
        for i in range(batch_size):
            # Get the most common class
            predicted_classes[i] = torch.mode(nearest_classes[i])[0]
        
        return predicted_classes
    # def predict_class_based_on_knn(self, feature_vectors, feature_idx, n):
    #     """
    #     Predict the class of each feature vector based on the majority class of its n nearest neighbors.
        
    #     :param feature_vectors: Tensor of shape (batch_size, feature_dim)
    #     :param feature_idx: Tensor of shape (batch_size,) containing class indices for each feature vector
    #     :param n: Number of nearest neighbors to consider
    #     :return: predicted_classes: Tensor of shape (batch_size,) containing predicted classes
    #     """
    #     batch_size = feature_vectors.shape[0]
        
    #     # Step 1: Compute pairwise Euclidean distance between feature vectors
    #     distances = torch.cdist(feature_vectors, feature_vectors, p=2)
        
    #     # Step 2: Get the indices of the n nearest neighbors for each feature vector
    #     _, nearest_indices = torch.topk(distances, k=n+1, largest=False)
    #     # Exclude the first index since it is the feature vector itself
    #     nearest_indices = nearest_indices[:, 1:]
        
    #     # Step 3: Get the classes of the n nearest neighbors
    #     nearest_classes = feature_idx[nearest_indices]
        
    #     # Step 4: Determine the most common class among the n nearest neighbors
    #     predicted_classes = torch.zeros(batch_size, dtype=torch.int64)
    #     for i in range(batch_size):
    #         # Get the most common class
    #         predicted_classes[i] = torch.mode(nearest_classes[i])[0]
        
    #     return predicted_classes

if __name__ == '__main__':

    class CONFIG:
        task = 'pose2gloss'
        group = 'Classification + Contrastive Learning'
        wandb_name = 'GCN 32'
        train_id = 6
        epochs = 1000
        lr = 0.0001
        base_path = './'
        use_wandb = True
        save_freq = 10
        resume = None
        # resume = '/home/horang1804/Dolmaggu/coda-modeling/Pose2Gloss/checkpoint/pose2gloss/4/epoch_last.pt'
        phase = 'train'
        device = 'cuda:0'
        batch_size = 16
        num_worker = 6
        # num_gloss = 427
        # data 관련 parameter
        time_window = 1.5
        freq = 30
        num_pos_samples = 8
        num_neg_samples = 8
        input_channels = 68
        num_filters = 128
        kernel_size = 3
        embedding_dim = 32
        seq_len = 45
        #loss 관련 parameter
        w_con = 1
        w_ce = 0


    from signlang_cl_dataloader_v3 import AIHUB_SIGNLANG_DATASET, collate_fn
    from torch.utils.data import DataLoader
    train_dataset = AIHUB_SIGNLANG_DATASET(base_path='/home/horang1804/HDD1/dataset/aihub_sign',
                                           phase='train',
                                           modality=['pose','gloss'],
                                           gloss_type=['SEN','WORD'],                          
                                           time_window=CONFIG.time_window, # sec
                                           freq=CONFIG.freq, # hz
                                           num_pos_samples=CONFIG.num_pos_samples,
                                           num_neg_samples=CONFIG.num_neg_samples,
                                           )
    
    train_loader = DataLoader(train_dataset,CONFIG.batch_size,shuffle=True, collate_fn=collate_fn)
    # valid_dataset = AIHUB_SIGNLANG_DATASET(base_path='/home/horang1804/HDD1/dataset/aihub_sign',
    #                                        phase='valid',
    #                                        modality=['pose','gloss'],
    #                                        gloss_type=['SEN','WORD'],                          
    #                                        time_window=CONFIG.time_window, # sec
    #                                        freq=CONFIG.freq, # hz
    #                                        num_pos_samples=CONFIG.num_pos_samples,
    #                                        num_neg_samples=CONFIG.num_neg_samples, )
    
    # valid_loader = DataLoader(valid_dataset,CONFIG.batch_size*4,shuffle=True, collate_fn=collate_fn)


    from timeseries_GCN import TimeSeriesGCN
    
    model = TimeSeriesGCN(input_channels = CONFIG.input_channels,
                          num_filters = CONFIG.num_filters, 
                          kernel_size  = CONFIG.kernel_size, 
                          embedding_dim = CONFIG.embedding_dim)
    



    trainer = SignLang_contrastive_trainer(CONFIG, model)
    
    if CONFIG.phase =='train':

        ##### init wandb #####
        if CONFIG.use_wandb:
            wandb.init(project=f'Dolmaenggu-{CONFIG.task}',
                       group = CONFIG.group,
                       name = CONFIG.wandb_name,
                       config = {k:v for k,v in CONFIG.__dict__.items() if k[:2]!='__'})
        trainer.train(train_loader,None)
    else:
        import glob
        ckpt_file = glob.glob(os.path.join('./checkpoint/pose2gloss/4/*'))
        for cf in ckpt_file:
            trainer.test(None, resume=cf, verbose=1)

    print('end')