import json
import glob
import os
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
base_path = '/home/horang1804/HDD1/dataset/aihub_sign'
gloss_type = ['WORD','SEN']
view_point = ['F']
phase = 'train'
if phase=='train':
    phase_dir = '1.Training'
elif phase=='valid':
    phase_dir='2.Validation'

with open('./word2id.json', "r") as st_json: word2id = json.load(st_json)
with open('./id2word.json', "r") as st_json: id2word = json.load(st_json)

token_len = len(id2word)

data_list = None 

new_data = []
for g_type in gloss_type:
    for v_point in view_point:
        if phase =='valid':
            new_data += [os.path.split(file_name)[-1] for file_name in glob.glob(os.path.join(base_path,f'004.수어영상/{phase_dir}/라벨링데이터/REAL',f'{g_type}/*/*/*_*_*_*_{v_point}'))]
        else:
            new_data += [os.path.split(file_name)[-1] for file_name in glob.glob(os.path.join(base_path,f'004.수어영상/{phase_dir}/라벨링데이터/REAL',f'{g_type}/*/*_*_*_*_{v_point}'))]
if data_list is None:
    data_list = new_data
else:
    data_list = list(set(data_list).intersection(new_data))



new_data = []
for g_type in gloss_type:
    for v_point in view_point:
        new_data += [os.path.split(file_name)[-1].split('_morpheme')[0] for file_name in glob.glob(os.path.join(base_path,f'004.수어영상/{phase_dir}/라벨링데이터/REAL',f'{g_type}/morpheme/*/*_*_*_*_{v_point}_*'))]
if data_list is None:
    data_list = new_data
else:
    data_list = list(set(data_list).intersection(new_data))

df = pd.DataFrame(columns=['file_id','gloss', 'start', 'end','duration','seq_len'])
idx_df = 0
for file_id in tqdm(data_list):
    _,_,gtid,ctid,view_point = file_id.split('_')

    gloss_type, data_id = gtid[:-4],gtid[-4:]
    collect_type, dir_id = ctid[:-2],ctid[-2:]

    # pose2d = None
    if phase =='valid':
        flist = glob.glob(os.path.join(base_path,f'004.수어영상/{phase_dir}/라벨링데이터/REAL',f'{gloss_type}/keypoint/{dir_id}/{file_id}/*'))
    else:
        flist = glob.glob(os.path.join(base_path,f'004.수어영상/{phase_dir}/라벨링데이터/REAL',f'{gloss_type}/{dir_id}/{file_id}/*'))
    # flist = glob.glob(os.path.join(self.base_path,'004.수어영상/1.Training/라벨링데이터/REAL',f'{gloss_type}/{dir_id}_{collect_type.lower()}_{gloss_type.lower()}_keypoint/{dir_id}/{file_id}/*'))
    keypoints_info = defaultdict(list)
    for fname in flist:
        with open(fname, "r") as st_json: keypoint_json = json.load(st_json)
        for k, v in keypoint_json['people'].items():
            if k[-2:]=='2d':
                keypoints_info[k].append(np.array(v).reshape(-1,3))

    pose2d = np.stack(keypoints_info['pose_keypoints_2d']) # seq, 25, 3
    handleft2d = np.stack(keypoints_info['hand_left_keypoints_2d']) # seq, 21, 3
    handright2d = np.stack(keypoints_info['hand_right_keypoints_2d']) # seq, 21 ,3
    face2d = np.stack(keypoints_info['face_keypoints_2d']) # seq, 70, 3
    pose = np.concatenate([pose2d[:,:,:2], handleft2d[:,:,:2], handright2d[:,:,:2], face2d[:,:,:2]],1)
    # pose = pose.astype(np.float32)
    seq_len = pose2d.shape[0]
    np.save(os.path.join(base_path,'pose',phase,collect_type,gloss_type,dir_id,f'{file_id}.npy'),pose)
    del pose
    # continue
    # gloss_seq = None
    # gloss_seq = np.ones(seq_len)*word2id['<sos>']
    fname = os.path.join(base_path,f'004.수어영상/{phase_dir}/라벨링데이터/REAL',f'{gloss_type}/morpheme/{dir_id}/{file_id}_morpheme.json')
    with open(fname, "r") as st_json: gloss_json = json.load(st_json)
    gloss_info = defaultdict(list)
    # gloss_info['start_time'].append(0)
    # gloss_info['gloss'].append('<sos>')
    # start_time=0
    # end_time=0
    # start_idx=0
    # end_idx=-1
    for d in gloss_json['data']:
        start_time = d['start']
        end_time = d['end']
        # start_idx = int(start_time*30)
        # end_idx = int(end_time*30)
        gloss = d['attributes'][0]['name']
        gloss_info['start_time'].append(str(start_time))
        gloss_info['end_time'].append(str(end_time))
        gloss_info['gloss'].append(gloss)
        # gloss_seq[start_idx:end_idx]=word2id[gloss]
    # gloss_info['end_time'].append(end_time) #real end 계산필요
    # gloss_info['gloss'].append('<eos>')
    # gloss_seq[end_idx:]=word2id['<eos>']
    # gloss_info['gloss_id'] = np.array([word2id[g] for g in gloss_info['gloss']])

    # pose2d
    df.loc[idx_df] = [file_id, 
                      '|'.join(gloss_info['gloss']), 
                      '|'.join(gloss_info['start_time']), 
                      '|'.join(gloss_info['end_time']),
                      gloss_json['metaData']['duration'],
                      seq_len]
    # print(seq_len,gloss_json['metaData']['duration']*30)
    idx_df +=1
    # break

df.to_csv(f'./gloss_label_{phase}_cl.csv', index=False)