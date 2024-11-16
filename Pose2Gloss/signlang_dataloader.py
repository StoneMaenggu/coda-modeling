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
                 ):
                
        self.base_path = base_path
        self.phase = phase
        self.modality = modality
        self.gloss_type = [gt.upper() for gt in gloss_type]
        self.view_point = view_point

        with open('./word2id.json', "r") as st_json: word2id = json.load(st_json)
        self.word2id = word2id

        with open('./id2word.json', "r") as st_json: id2word = json.load(st_json)
        self.id2word = id2word

        self.token_len = len(id2word)

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
            self.gloss_df = pd.read_csv(f'./gloss_label_{self.phase}.csv')
            new_data = list(self.gloss_df.file_id)
            if data_list is None:
                data_list = new_data
            else:
                data_list = list(set(data_list).intersection(new_data))
        else: self.use_gloss=False

        if 'text' in modality: self.use_text=True
        else: self.use_text=False

        self.data_list = data_list 
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        file_id = self.data_list[idx]
        
        _,_,gtid,ctid,view_point = file_id.split('_')

        gloss_type, data_id = gtid[:-4],gtid[-4:]
        collect_type, dir_id = ctid[:-2],ctid[-2:]

        pose2d = None
        if self.use_pose:
            keypoints_seq = np.load(os.path.join(self.base_path,'pose',self.phase,collect_type,gloss_type,dir_id,f'{file_id}.npy'))
            # 0~24: pose, 25~45: left hand, 46~66: right hand, 67~136: face
            pose = keypoints_seq[:,:67,:]
            pose = (pose - np.array([1920/2,0]))/1080
            seq_len = pose.shape[0]
            
        if self.use_gloss:
            gloss = self.gloss_df[self.gloss_df.file_id == file_id].gloss.values[0].split('|')
            gloss_id = [self.word2id[g] for g in gloss]

        return {
            'modality':self.modality,
            'pose': torch.tensor(pose, dtype=torch.float32),
            'gloss':F.one_hot(torch.tensor(gloss_id, dtype=torch.int64), self.token_len),
            }
    
import torch
from torch.nn.utils.rnn import pad_sequence

# def collate_fn(batch):
#     # batch는 (sequence, label)의 리스트입니다.
#     sequences = [item[0] for item in batch]
#     labels = [item[1] for item in batch]
    
#     return sequences, labels


from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    # batch는 (sequence, label)의 리스트입니다.
    sequences = [item['pose'] for item in batch]
    labels = [item['gloss'] for item in batch]
    
    # 시퀀스를 패딩하여 동일한 길이로 맞춥니다.
    padded_sequences = pad_sequence(sequences, batch_first=False, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=False, padding_value=0)
    
    
    return padded_sequences, padded_labels
if __name__ == '__main__':
    train_dataset = AIHUB_SIGNLANG_DATASET(base_path='/home/horang1804/HDD1/dataset/aihub_sign',
                                           phase='train',
                                           modality=['pose','gloss'],
                                           gloss_type=['SEN'])
    
    batch = train_dataset.__getitem__(0)

    # train_loader = DataLoader(train_dataset,32,shuffle=True, num_workers=4, collate_fn=collate_fn)
    # pose_batch, gloss_batch = next(iter(train_loader))
    
    print('end')

    # import time 

    # start_time = time.time()
    # for i, batch in enumerate(train_loader):
    #     end_time = time.time()
    #     print(f'\r{(end_time-start_time)/(i+1)}')