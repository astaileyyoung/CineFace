import ast 
import shutil
from pathlib import Path 
from argparse import ArgumentParser

import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm


def join_episodes(src,
                  series_id,
                  episode_id=None,
                  episode_path='./data/episodes.csv'):
    if isinstance(src, str):
        df = pd.read_csv(src, index_col=0)
    else:
        df = src
    episode_df = pd.read_csv(episode_path, index_col=0)
    episode_df = episode_df[episode_df['series_id'] == int(series_id)]
    if episode_id is not None:
        episode_df = episode_df[episode_df['episode_id'] == int(episode_id)]
    df_ep = df.merge(episode_df[['episode_id', 'cast']],
                 on='episode_id',
                 how='inner'
                 )
    return df_ep
    

def chinese_whisper(df):
    encodings = [dlib.vector(ast.literal_eval(x)) for x in df['encoding']]
    labels = dlib.chinese_whispers_clustering(encodings, 0.5)
    df = df.assign(label=labels)
    g = df[['label', 'fp']].groupby('label').count()
    top = g[g['fp'] > 15]
    df = df.merge(top.rename({'fp': 'count'}, axis=1),
                  how='left',
                  right_index=True,
                  left_index=True)
    return df


def cluster_series(df):
    seasons = df['season'].unique()
    episodes = df['episode'].unique()
    for season in tqdm(seasons):
        for episode in tqdm(episodes):
            name = f'S{str(season).zfill(2)}E{str(episode).zfill(2)}'
            dst = Path(dst).joinpath(name)
            episode_fp = df_fp[(df_fp['season'] == season) & (df_fp['episode'] == episode)]
            df_top = chinese_whisper(episode_fp)
            df = df.merge(df_top[['season', 'episode', 'frame_num', 'face_num', 'label']],
                          how='left',
                          on=['season', 'episode', 'frame_num', 'face_num'])


def main(args):
    dst = Path(args.cluster_dir).joinpath(args.dst)
    df = join_episodes(args.src,
                       args.series_id,
                       image_dir=args.image_dir)
    df['fp'] = df['filename'].map(lambda x: Path(args.image_dir).joinpath(x))
    df = cluster_series(df)   
    df = df.drop(['fp', 'cast'], axis=1)
    df.to_csv(args.dst)
    

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
