import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import glob
import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd

class AIHUB_SIGNLANG_DATASET(Dataset):
    def __init__(self, 
                 base_path:str = './dataset',
                 phase:str = 'train',
                 modality:list = ['video', 'pose', 'gloss', 'text'],
                 gloss_type:list = ['word','sen'],
                 view_point:list = ['F'],
                 time_window:float = 2, # sec
                 freq:float = 10, # hz
                 num_pos_samples:int=4,
                 num_neg_samples:int=4,
                 ):
                
                
        self.base_path = base_path
        self.phase = phase
        self.modality = modality
        self.gloss_type = [gt.upper() for gt in gloss_type]
        self.view_point = view_point
        self.time_window = time_window
        self.freq = freq
        self.num_pos_samples = num_pos_samples
        self.num_neg_samples = num_neg_samples
        self.use_pose = [5,6,2,3]
        self.hand_pose = [0,2,4,5,6,8,9,10,12,13,14,16,17,18,20]
        self.use_features = self.use_pose+[i+25 for i in self.hand_pose]+[i+46 for i in self.hand_pose]

        data_list = None
        if 'video' in modality: 
            self.use_video=True 
            
        else: self.use_video=False

        if 'pose' in modality: 
            self.use_pose=True
            new_data = []
            for g_type in self.gloss_type:
                for v_point in self.view_point:
                    new_data += [os.path.split(file_name)[-1][:-4] for file_name in glob.glob(os.path.join(self.base_path,f'pose/{self.phase}',f'*/{g_type}/*/*_*_*_*_{v_point}*'))]
            if data_list is None:
                data_list = new_data
            else:
                data_list = list(set(data_list).intersection(new_data))
        else: self.use_pose=False
    
        if 'gloss' in modality: 
            self.use_gloss=True
            
            new_data = []

            with open('./word2id.json', "r") as st_json: word2id = json.load(st_json)
            self.word2id = word2id

            with open('./id2word.json', "r") as st_json: id2word = json.load(st_json)
            self.id2word = id2word

            self.gloss_df = pd.read_csv(f'./gloss_label_{self.phase}_cl.csv')
            self.gloss_df = self.gloss_df[np.abs(self.gloss_df.duration-self.gloss_df.seq_len/30)<0.01]
            self.gloss_df = self.gloss_df.dropna()
            gloss_dic = word2id.keys()
            def exist(x):
                ret = True
                for g in x.split('|'):
                    if g not in gloss_dic:
                        ret = False
                return ret
            self.gloss_df.exist = self.gloss_df.gloss.apply(exist)
            self.gloss_df = self.gloss_df[self.gloss_df.exist]
            new_data = list(self.gloss_df.file_id)
            if data_list is None:
                data_list = new_data
            else:
                data_list = list(set(data_list).intersection(new_data))

            # gloss, start, end를 분리하여 각각의 행으로 확장
            df_expanded = self.gloss_df.apply(lambda x: pd.Series(x['gloss'].split('|')), axis=1).stack().reset_index(level=1, drop=True).to_frame('gloss')
            df_expanded['start'] = self.gloss_df.apply(lambda x: pd.Series(x['start'].split('|')), axis=1).stack().reset_index(level=1, drop=True).astype(float)
            df_expanded['end'] = self.gloss_df.apply(lambda x: pd.Series(x['end'].split('|')), axis=1).stack().reset_index(level=1, drop=True).astype(float)
            df_expanded['file_id'] = self.gloss_df['file_id'].repeat(df_expanded.groupby(level=0).size())
            df_expanded['duration'] = self.gloss_df['duration'].repeat(df_expanded.groupby(level=0).size())
            df_expanded['seq_len'] = self.gloss_df['seq_len'].repeat(df_expanded.groupby(level=0).size())

            # 인덱스 리셋
            df_expanded = df_expanded.reset_index(drop=True)
            self.gloss_df = df_expanded

            self.gloss_df = self.gloss_df[self.gloss_df.duration>self.gloss_df.end]
            
            avail_id = (self.gloss_df.end-self.gloss_df.start)
            avail_id = avail_id<1.5
            avail_id = avail_id>0.8
            self.gloss_df2 = self.gloss_df[avail_id]

            avail_dic = self.gloss_df2.gloss.value_counts()>30

            def avail(x):
                return avail_dic[x]
            
            self.gloss_df = self.gloss_df2[self.gloss_df2.gloss.apply(avail)]

        else: self.use_gloss=False

        if 'text' in modality: self.use_text=True
        else: self.use_text=False

        self.data_list = data_list 
        
    def __len__(self):
        return len(self.gloss_df)
    
    def __getitem__(self, idx):
        pose = []
        label = []

        gloss, start, end, file_id, duration, seq_len = self.gloss_df.iloc[idx]
        anchor_sample = self.get_samples(gloss, start, end, file_id, duration, seq_len)
        pose.append(anchor_sample)
        label.append(self.word2id[gloss])
        if self.phase != 'train':
            return torch.tensor(pose, dtype=torch.float32), label

        pos_file = self.gloss_df[np.logical_and(self.gloss_df.gloss==gloss, self.gloss_df.index != idx)]
        pos_file = pos_file.iloc[np.random.randint(0,len(pos_file),(self.num_pos_samples,))].values
        for gloss, start, end, file_id, duration, seq_len in pos_file:
            pose.append(self.get_samples(gloss, start, end, file_id, duration, seq_len))
            label.append(self.word2id[gloss])
        return torch.tensor(pose, dtype=torch.float32), label
    
        # -------------------------------------------
        # negative_sample = []
        # neg_file = self.gloss_df[np.logical_and(self.gloss_df.gloss!=gloss, self.gloss_df.index != idx)]
        # neg_file = neg_file.iloc[np.random.randint(0,len(neg_file),(self.num_neg_samples,))].values
        # for gloss, start, end, file_id, duration, seq_len in neg_file:
        #     negative_sample.append(self.get_samples(gloss, start, end, file_id, duration, seq_len))
        
        # label = torch.zeros((self.num_pos_samples+self.num_neg_samples,))
        # label[:self.num_pos_samples] = 1
        # -------------------------------------------

        # return torch.tensor(np.tile(anchor_sample,(self.num_neg_samples,1,1,1))), torch.cat([torch.tensor(np.stack(positive_sample)), torch.tensor(np.stack(negative_sample))],0), label
        # return torch.tensor(np.tile(anchor_sample,(self.num_samples,1,1,1))), torch.tensor(np.stack(positive_sample)), torch.tensor(np.stack(negative_sample))
    

    def get_samples(self, gloss, start, end, file_id, duration, seq_len):
        _,_,gtid,ctid,view_point = file_id.split('_')

        gloss_type, data_id = gtid[:-4],gtid[-4:]
        collect_type, dir_id = ctid[:-2],ctid[-2:]

        pose2d = None
        if self.use_pose:
            keypoints_seq = np.load(os.path.join(self.base_path,'pose',self.phase,collect_type,gloss_type,dir_id,f'{file_id}.npy'))
            # 0~24: pose, 25~45: left hand, 46~66: right hand, 67~136: face

            # Slicing
            seq_len = keypoints_seq.shape[0]
            if (end-start)>self.time_window: # time window 보다 길 경우
                can_start = max(0, start-self.time_window*0.1)
                can_end = min(seq_len/30, end+self.time_window*0.1)
            else:   # time window 보다 짧은 경우
                can_start = max(0, min(end-self.time_window, start-self.time_window*0.1))
                can_end = min(seq_len/30, max(start+self.time_window,end+self.time_window*0.1))

            
            can_start_init = can_start
            can_start_fin = can_end-self.time_window

            start_id = np.random.randint(int(can_start_init*30),int(can_start_fin*30))
            
            end_id = start_id+int(self.time_window*30)
            
            seq_idx = np.arange(start_id, end_id, 30//self.freq)
            pose = keypoints_seq[seq_idx,:,:][:,self.use_features,:]
            

            # Augementation
            # white noise 
            if self.phase == 'train':
                pose += np.random.randn(*pose.shape)*3

                # bias
                pose += + np.random.randn(2)*2

            # Normalization
            pose = (pose - np.array([1920/2,0]))/1080
        
        # if self.use_gloss:
        #     gloss = self.gloss_df[self.gloss_df.file_id == file_id].gloss.values[0].split('|')
        #     gloss_id = [self.word2id[g] for g in gloss]

        # a = np.sum(keypoints_seq==0)
        # if a !=0:
        #     print(a)
        return pose

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # batch는 (sequence, label)의 리스트입니다.
    ref = []
    ref_cls = []
    for item in batch:
        ref+=torch.tensor(item[0])
        ref_cls+=item[1]

    ref = torch.stack(ref).to(torch.float32)
    ref_cls = torch.tensor(ref_cls).to(torch.int64)
    
    return ref, ref_cls


    # if len(batch[0])==3:
    #     ref = torch.cat([item[0] for item in batch],0).to(torch.float32)
    #     pos = torch.cat([item[1] for item in batch],0).to(torch.float32)
    #     neg = torch.cat([item[2] for item in batch],0).to(torch.float32)
        
    #     return ref, pos, neg
    # else:
    #     ref = torch.cat([item[0] for item in batch],0).to(torch.float32)
    #     ref_cls = torch.tensor([item[1] for item in batch]).to(torch.int64)
        
    #     return ref, ref_cls


# from torch.nn.utils.rnn import pad_sequence
# def collate_fn(batch):
#     # batch는 (sequence, label)의 리스트입니다.
#     sequences = [item['pose'] for item in batch]
#     labels = [item['gloss'] for item in batch]
    
#     # 시퀀스를 패딩하여 동일한 길이로 맞춥니다.
#     padded_sequences = pad_sequence(sequences, batch_first=False, padding_value=0)
#     padded_labels = pad_sequence(labels, batch_first=False, padding_value=0)
    
    
#     return padded_sequences, padded_labels
if __name__ == '__main__':
    train_dataset = AIHUB_SIGNLANG_DATASET(base_path='/home/horang1804/HDD1/dataset/aihub_sign',
                                           phase='train',
                                           modality=['pose','gloss'],
                                           gloss_type=['SEN'],                          
                                           time_window=1.5, # sec
                                           freq=10, # hz
                                           num_pos_samples=1,
                                           num_neg_samples=4,
                 )
    
    pose, label = train_dataset.__getitem__(0)

    train_loader = DataLoader(train_dataset,4,shuffle=True, num_workers=0, collate_fn=collate_fn)
    pose_batch, label = next(iter(train_loader))
    from tqdm import tqdm
    for _ in tqdm(train_loader):
        pass
    print('end')

    # import time 

    # start_time = time.time()
    # for i, batch in enumerate(train_loader):
    #     end_time = time.time()
    #     print(f'\r{(end_time-start_time)/(i+1)}')