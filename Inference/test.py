import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'  # 또는 'Noto Sans CJK' 등 설치된 한글 폰트 이름

# '-' 기호가 깨지는 경우 방지
plt.rcParams['axes.unicode_minus'] = False


df = pd.read_csv('gloss_db.csv')

X = df.drop(['pose_path','gloss'],axis=1)

g = df.gloss
le = LabelEncoder()
g_i = le.fit_transform(g)
pca = PCA(n_components=2)
X_low = pca.fit_transform(X)
plt.figure(1)
for i in range(np.max(g_i)):
    idx = g_i==i
    plt.scatter(X_low[idx,0],X_low[idx,1],s=100,label=le.inverse_transform(np.array([i]))[0])
plt.legend()
plt.title('PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
print('end')

