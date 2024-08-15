import glob
import json
import pandas as pd
from tqdm import tqdm
data_type = ['WORD','SEN']

df = pd.DataFrame(columns=['file_path', 'g_type', 'start_time', 'end_time', 'gloss'])
idx_df = 0
for dt in data_type:
    print('='*10,dt,'='*10)
    for i, file in enumerate(tqdm(glob.glob(f"/home/horang1804/HDD1/dataset/aihub_sign/004.수어영상/1.Training/라벨링데이터/REAL/{dt}/morpheme/*/*_*_*_*_F_*"))):
        with open(file, "r") as st_json: 
            st_python = json.load(st_json)
            ### print ###
            # print(len(st_python['data']), end='  ')
            # print(st_python['data'][0]['start'], end='  ')
            # print(st_python['data'][-1]['end'], end='  ')
            # for a in st_python['data']:
            #     print(a['attributes'][0]['name'], end=' ')
            # print()
            #############
            # obj = {'file_path':file,
            #     'start_end_time':[[d['start'], d['end']] for d in st_python['data']],
            #     'gloss':[d['attributes'][0]['name'] for d in st_python['data']]}
            df.loc[idx_df] = [file,
                              dt,
                              '|'.join([str(d['start']) for d in st_python['data']]),
                              '|'.join([str(d['end']) for d in st_python['data']]),
                              '|'.join([d['attributes'][0]['name'] for d in st_python['data']])]
            
            idx_df+=1

df.to_csv('./gloss_info.csv', index=False)