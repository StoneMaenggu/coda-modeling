import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import glob
import os
import json
from collections import defaultdict
import numpy as np
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
                    new_data += [os.path.split(file_name)[-1] for file_name in glob.glob(os.path.join(self.base_path,'004.수어영상/1.Training/라벨링데이터/REAL',f'{g_type}/*/*_*_*_*_{v_point}'))]
            if data_list is None:
                data_list = new_data
            else:
                data_list = list(set(data_list).intersection(new_data))
        else: self.use_pose=False
    
        if 'gloss' in modality: 
            self.use_gloss=True
            
            new_data = []
            for g_type in self.gloss_type:
                for v_point in view_point:
                    new_data += [os.path.split(file_name)[-1].split('_morpheme')[0] for file_name in glob.glob(os.path.join(self.base_path,'004.수어영상/1.Training/라벨링데이터/REAL',f'{g_type}/morpheme/*/*_*_*_*_{v_point}_*'))]
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
            flist = glob.glob(os.path.join(self.base_path,'004.수어영상/1.Training/라벨링데이터/REAL',f'{gloss_type}/{dir_id}/{file_id}/*'))
            # flist = glob.glob(os.path.join(self.base_path,'004.수어영상/1.Training/라벨링데이터/REAL',f'{gloss_type}/{dir_id}_{collect_type.lower()}_{gloss_type.lower()}_keypoint/{dir_id}/{file_id}/*'))
            keypoints_info = defaultdict(list)
            for fname in flist:
                with open(fname, "r") as st_json: keypoint_json = json.load(st_json)
                for k, v in keypoint_json['people'].items():
                    if k[-2:]=='2d':
                        keypoints_info[k].append(np.array(v).reshape(-1,3))

            pose2d = np.stack(keypoints_info['pose_keypoints_2d'])
            seq_len = pose2d.shape[0]

        gloss_seq = None
        if self.use_gloss:
            gloss_seq = np.ones(seq_len)*self.word2id['<sos>']
            fname = os.path.join(self.base_path,'004.수어영상/1.Training/라벨링데이터/REAL',f'{gloss_type}/morpheme/{dir_id}/{file_id}_morpheme.json')
            with open(fname, "r") as st_json: gloss_json = json.load(st_json)
            gloss_info = defaultdict(list)
            gloss_info['start_time'].append(0)
            gloss_info['gloss'].append('<sos>')
            start_time=0
            end_time=0
            start_idx=0
            end_idx=-1
            for d in gloss_json['data']:
                start_time = d['start']
                end_time = d['end']
                start_idx = int(start_time*30)
                end_idx = int(end_time*30)
                gloss = d['attributes'][0]['name']
                gloss_info['start_time'].append(start_time)
                gloss_info['end_time'].append(end_time)
                gloss_info['gloss'].append(gloss)
                gloss_seq[start_idx:end_idx]=self.word2id[gloss]
            gloss_info['end_time'].append(end_time) #real end 계산필요
            gloss_info['gloss'].append('<eos>')
            gloss_seq[end_idx:]=self.word2id['<eos>']
            gloss_info['gloss_id'] = np.array([self.word2id[g] for g in gloss_info['gloss']])

        return {
            'modality':self.modality,
            'pose': torch.tensor(pose2d, dtype=torch.float32),
            'gloss_seq': torch.tensor(gloss_seq, dtype=torch.int64),
            'gloss':F.one_hot(torch.tensor(gloss_info['gloss_id'], dtype=torch.int64), self.token_len),
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
    
    # batch = train_dataset.__getitem__(0)

    train_loader = DataLoader(train_dataset,32,shuffle=True,collate_fn=collate_fn)
    pose_batch, gloss_batch = next(iter(train_loader))

    print('end')