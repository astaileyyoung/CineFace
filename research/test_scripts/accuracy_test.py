import os
import subprocess as sp
from pathlib import Path 

import pandas as pd
from tqdm import tqdm 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def measure_diff(face, row):
    face['name'] = row['name']
    face['face_num'] = row['face_num']
    face['x1_ground'] = row['x1']
    face['y1_ground'] = row['y1']
    face['x2_ground'] = row['x2']
    face['y2_ground'] = row['y2']
    face['x1_diff'] = face['x1_ground'] - face['x1']
    face['y1_diff'] = face['x1_ground'] - face['y1']
    face['x2_diff'] = face['x1_ground'] - face['x2']
    face['y2_diff'] = face['x1_ground'] - face['y2']
    face['area_diff'] = row['area'] - face['area']
    face['pct_diff'] = row['pct_of_frame'] - face['pct_of_frame']
    return face


def empty_df(face):
    face = row[['name', 'face_num', 'img_width', 'img_height']].to_frame().transpose()
    # face['name'] = row['name']
    # face['face_num'] = row['face_num']
    face['x1'] = None 
    face['y1'] = None 
    face['x2'] = None 
    face['y2'] = None
    face['width'] = None
    face['height'] = None
    face['area'] = None
    face['pct_of_frame'] = None 
    face['confidence'] = None
    face['x1_ground'] = row['x1']
    face['y1_ground'] = row['y1']
    face['x2_ground'] = row['x2']
    face['y2_ground'] = row['y2']
    face['x1_diff'] = None
    face['y1_diff'] = None
    face['x2_diff'] = None
    face['y2_diff'] = None
    face['area_diff'] = None
    face['pct_diff'] = None
    return face


funcs = [('dlib', 'dlib_accuracy.py'),
         ('torch', 'mtcnn_accuracy.py'),
         ('cv', 'opencv_accuracy.py'),
         ('retina', 'retina_accuracy.py')]
base = Path('/home/amos/programs/CineFace/research/test_scripts')
df = pd.read_csv('/home/amos/programs/CineFace/research/data/faces.csv', index_col=0)
df['path'] = df['name'].map(lambda x: str(Path('/home/amos/programs/CineFace/research/test_images').joinpath(x)))
df.to_csv('./images.csv')
for env, f in funcs:
    dst = Path('/home/amos/programs/CineFace/research/test_results').joinpath(f'{env}.csv')
    command = ['conda', 'run', '-n', env, 'python', str(base.joinpath(f)), './images.csv', str(dst)]
    sp.run(command)
    face = pd.read_csv(str(dst), index_col=0)
    # data = []
    # for idx, row in face.iterrows():
    #     temp = df.iloc[idx]
    #     eph = measure_diff(row, temp)
    #     data.append(eph)
    # face_df = pd.concat(data)
    face.to_csv(f'/home/amos/programs/CineFace/research/test_results/{env}.csv')

    # if face.shape[0] < 1:
    #     face = empty_df(face) 
    # else:
    #     face = measure_diff(face, row)
    # face['id'] = idx
    # face['detector'] = f
    # dfs.append(face)
# for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
#     dfs = []
#     for env, f in tqdm(funcs, leave=False):
#         fp = Path('/home/amos/programs/CineFace/research/test_images').joinpath(row['name'])
#         dst = Path('/home/amos/programs/CineFace/research/test_scripts/temp').joinpath(f'{Path(row["name"]).stem}.csv')
#         command = ['conda', 'run', '-n', env, 'python', str(base.joinpath(f)), str(fp), str(dst)]
#         sp.run(command)
#         face = pd.read_csv(str(dst), index_col=0)
#         if face.shape[0] < 1:
#             face = empty_df(face) 
#         else:
#             face = measure_diff(face, row)
#         face['id'] = idx
#         face['detector'] = f
#         dfs.append(face)
#     face_df = pd.concat(dfs)
#     face_df.reset_index(inplace=True)
#     face_df.to_csv(Path('./results/').joinpath(f'{Path(row["name"]).stem}.csv'))


