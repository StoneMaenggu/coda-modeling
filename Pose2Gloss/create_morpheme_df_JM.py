import glob
import json
import pandas as pd
from tqdm import tqdm

# /1.Training/라벨링데이터/2.형태소_비수지(json)_TL/03_JSON_TrL/1.tact_morpheme/1.자연재난/COLDWAVE/1_1

df = pd.DataFrame(columns=['file_name', 'phase', 'situation', 'detail', 'gloss', 'text'])
idx_df = 0
for i, file in enumerate(tqdm(glob.glob(f"/home/horang1804/HDD1/dataset/aihub_sign/additional/114.재난_안전_정보_전달을_위한_수어영상_데이터/01.데이터/**/*.json", recursive=True))):
    try:
        with open(file, "r") as st_json: 
            st_python = json.load(st_json)
            
        dirs = file.split('/')
        phase = dirs[0].split('.')[-1]
        situation = dirs[14].split('.')[-1]
        detail = dirs[15]
        fname = dirs[17].split('.')[0]

        text = st_python['korean_text']
        gloss = []
        for g in st_python['sign_script']['sign_gestures_both']:
            gloss.append(g['gloss_id'])

        df.loc[idx_df] = [
            fname,
            phase,
            situation,
            detail,
            '|'.join(gloss),
            text
        ]
        
        idx_df+=1
    except:
        print('error')
df.to_csv('./gloss_info_JM.csv', index=False)