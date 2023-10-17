import ast 
import shutil
from pathlib import Path 
from argparse import ArgumentParser

import cv2
import dlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm


import sklearn.cluster as cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_images(files):
    data = []
    for file in files:
        try:
            temp, frame_num, face_num = file.stem.split('_')
        except:
            print(file)
            exit()
        s = int(temp[1:3])
        e = int(temp[4:6])
        datum = {'fp': str(file.absolute().resolve()),
                 'season': s,
                 'episode': e,
                 'frame_num': int(frame_num),
                 'face_num': int(face_num)
                }
        data.append(datum)
    fp_df = pd.DataFrame(data)
    return fp_df


def save_images(df,
                dst):
    
    if not dst.exists():
        Path.mkdir(dst, parents=True)
        
    for idx, row in df.iterrows():
        fp = dst.joinpath(f'{row["label"]}/{Path(row["fp"]).name}')
        if not fp.parent.exists():
            Path.mkdir(fp.parent)
        shutil.copy(row["fp"], fp)


def chinese_whisper(df,
                    dst):
    encodings = [dlib.vector(ast.literal_eval(x)) for x in df['encoding']]
    labels = dlib.chinese_whispers_clustering(encodings, 0.5)
    df = df.assign(label=labels)
    g = df[['label', 'fp']].groupby('label').count()
    top = g[g['fp'] > 15]
    df_top = df.merge(top.rename({'fp': 'count'}, axis=1),
                   how='inner',
                   right_index=True,
                   left_on='label')
    save_images(df_top, dst)


def cluster_series(df_fp, 
                   dst):
    seasons = df_fp['season'].unique()
    episodes = df_fp['episode'].unique()
    for season in tqdm(seasons):
        for episode in tqdm(episodes):
            name = f'S{str(season).zfill(2)}E{str(episode).zfill(2)}'
            dst = Path(dst).joinpath(name)
            episode_fp = df_fp[(df_fp['season'] == season) & (df_fp['episode'] == episode)]
            chinese_whisper(episode_fp,
                            dst)
    
    
def main(args):
    dst = Path(args.cluster_dir).joinpath(args.dst)
    
    df = pd.read_csv(args.src, index_col=0)
    episode_df = pd.read_csv(args.episode_df, index_col=0)
    episode_df = episode_df[episode_df['series_id'] == int(args.series_id)]
    if args.episode_id is not None:
        episode_df = episode_df[episode_df['episode_id'] == int(args.episode_id)]
    df_ep = df.merge(episode_df[['episode_id', 'cast']],
                     on='episode_id',
                     how='inner'
                     )
    files = [x for x in Path(args.image_dir).iterdir()]
    fp_df = get_images(files)
    df_fp = df_ep.merge(fp_df,
                        on=['frame_num', 'face_num', 'season', 'episode'],
                        how='inner')
    cluster_series(df_fp,
                   dst)    
    

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('series_id')
    ap.add_argument('dst')
    ap.add_argument('--cluster_dir', default='./data/clustering')
    ap.add_argument('--src', default='./data/faces.csv')
    ap.add_argument('--episode_id', default=None)
    ap.add_argument('--episode_df', default='./data/episodes.csv')
    ap.add_argument('--image_dir', default='./data/images')
    args = ap.parse_args()
    main(args)
