import ast 
import shutil
from pathlib import Path 
from argparse import ArgumentParser

import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from imdb import Cinemagoer

from utils import format_series_name


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
    top = g[g['fp'] > 30]
    df = df.merge(top.rename({'fp': 'count'}, axis=1),
                  how='left',
                  left_on='label',
                  right_index=True)
    return df


# def merge_dfs(df,
#               df_top):
#     df.loc[df_top.index, 'label'] = df_top['label']
#     # for idx, row in df_top.iterrows():
#     #     df.loc[idx, 'label'] = row['label']
#     return df


def cluster_series(df):
    df['label'] = np.nan
    seasons = df['season'].unique()
    episodes = df['episode'].unique()

    for season in tqdm(seasons):
        for episode in tqdm(episodes):
            episode_fp = df[(df['season'] == season) & (df['episode'] == episode)]
            df_top = chinese_whisper(episode_fp)
            df_top = df_top[df_top['count'].notna()]
            df.loc[df_top.index, 'label'] = df_top['label']
    return df


def main(args):
    ia = Cinemagoer()
    series = ia.get_movie(args.series_id)
    name = format_series_name(series)
    dst = Path(args.cluster_dir).joinpath(name)
    image_dir = Path(args.image_dir).joinpath(name)
    if not dst.exists():
        Path.mkdir(dst)
        
    df = join_episodes(args.faces,
                       args.series_id)
    df['fp'] = df['filename'].map(lambda x: Path(image_dir).joinpath(x))
    df = cluster_series(df)   
    df = df.drop(['fp', 'cast'], axis=1)
    df.to_csv(args.dst)
    

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('series_id')
    ap.add_argument('dst')
    ap.add_argument('--cluster_dir', default='./data/clusters')
    ap.add_argument('--faces', default='./data/faces.csv')
    ap.add_argument('--episode_id', default=None)
    ap.add_argument('--episode_df', default='./data/episodes.csv')
    ap.add_argument('--image_dir', default='./data/images')
    args = ap.parse_args()
    main(args)
