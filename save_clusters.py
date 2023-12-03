import shutil
from pathlib import Path
from functools import partial
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm 
from imdb import Cinemagoer 

from utils import format_series_name, format_episode_name


def save_cluster(row,
                 series_name,
                 label_column='label',
                 dst='./data/clusters',
                 images='./data/images'):
    dst_dir = Path(dst).joinpath(series_name)
    if not dst_dir.exists():
        Path.mkdir(dst_dir, parents=True)
    
    name = format_episode_name(row)
    episode_dir = dst_dir.joinpath(name)
    if not episode_dir.exists():
        Path.mkdir(episode_dir, parents=True)

    cluster_dir = episode_dir.joinpath(str(row[label_column]))
    if not cluster_dir.exists():
        Path.mkdir(cluster_dir, parents=True)
 
    image_dir = Path(images).joinpath(series_name)
    src_fp = image_dir.joinpath(row['filename'])
    dst_fp = cluster_dir.joinpath(row['filename'])
    shutil.copy(str(src_fp), str(dst_fp))
    

def main(args):
    df = pd.read_csv(args.src, index_col=0, dtype={'label': 'Int64'})
    df = df[df['series_id'] == int(args.series_id)]
    df = df[df[args.label_column].notna()]
    series = ia.get_movie(args.series_id)
    series_name = format_series_name(series)
    f = partial(save_cluster,
                series_name=series_name,
                label_column=args.label_column,
                dst=args.cluster_dir,
                images=args.image_dir)
    tqdm(total=df.shape[0]).pandas()
    df.progress_apply(f, axis=1)   


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('series_id')
    ap.add_argument('--label_column', default='label')
    ap.add_argument('--cluster_dir', default='./data/clusters')
    ap.add_argument('--image_dir', default='./data/images')
    args = ap.parse_args()

    ia = Cinemagoer()
    
    main(args)
