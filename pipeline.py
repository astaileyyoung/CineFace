import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"


import logging
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd 
from tqdm import tqdm
from imdb import Cinemagoer, IMDbError

from find_faces import find_faces
from match_faces import match_faces
from metadata import parse_path
from utils import get_id, get_id_sparse, gather_files


def get_metadata(df):
    filepath = df.at[0, 'filepath']
    datum = parse_path(filepath)
    if not pd.isnull(datum['year']):
        try:
            imdb_id = get_id(title=datum['title'],
                             year=datum['year'],
                             kind='tv')
        except IndexError:
            imdb_id = get_id_sparse(title=datum['title'],
                                    kind='tv')
    else:
        imdb_id = get_id_sparse(title=datum['title'],
                                kind='tv')
    ia = Cinemagoer()
    cnt = 0
    while cnt < 5:
        try:
            info = ia.get_movie(imdb_id)
            break
        except IMDbError:
            cnt += 1
    df['title'] = info.data['title']
    df['year'] = info.data['year']
    df['imdb_id'] = imdb_id
    df['season'] = datum['season']
    df['episode'] = datum['episode']
    return df


def add_metadata(df, metadata):
    for k, v in metadata.items():
        if k in ['title', 'year', 'imdb_id', 'season', 'episode']:
            df = df.assign(k=v)
    return df


def pipeline(file,
             num_threads=4,
             metadata=None):
    df = find_faces(file, num_threads=num_threads)
    if not metadata:
        df = get_metadata(df)
    else:
        df = add_metadata(df, metadata)
    df = match_faces(df)
    return df


def main(args):
    if Path(args.src).is_dir():
        if not Path(args.dst).exists():
            Path.mkdir(Path(args.dst), parents=True)

        files = gather_files(args.src, ext=('.mp4', '.mkv', '.m4v', '.avi'))
        for file in tqdm(files, leave=True):
            dst = Path(args.dst).joinpath(f'{file.stem}.csv')
            if dst.exists():
                continue

            df = pipeline(file, num_threads=args.num_threads)
            df.to_csv(dst)

    else:
        df = pipeline(args.src, num_threads=args.num_threads)
        df.to_csv(args.dst)  


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--num_threads', '-n', default=4, type=int)
    ap.add_argument('--host', default='localhost', type=str)
    ap.add_argument('--username', default='amos')
    ap.add_argument('--password', default='M0$hicat')
    ap.add_argument('--port', default='3306')
    ap.add_argument('--database', default='film')
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        filename='./logs/pipeline.log',
                        format='%(asctime)s %(levelname)s-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    main(args)
