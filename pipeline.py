from argparse import ArgumentParser

import numpy as np
import pandas as pd
from imdb import Cinemagoer

from find_faces import find_faces
from save_faces import save_faces
from encode_faces import encode_faces
from get_episode_info import get_episode_info
from cluster_series import cluster_series, join_episodes


def get_info(series_id):
    ia = Cinemagoer()
    series = ia.get_movie(series_id)
    name = f'{series.data["title"].replace(' ', '-')}_{series.data["year"]}_{series_id}'
    return name


def join_encodings(face_dir,
                   encoding_dir):
    face_df = pd.concat([x for x in Path(face_dir).iterdir()], axis=0)
    encodings = [x for x in Path(encoding_dir).iterdir()]

    data = []
    for encoding in encodings:
        info, frame_num, face_num = encoding.stem.split('_')
        season = int(info[1:3])
        episode = int(info[4:6])
        datum = {'encoding': np.load(encoding),
                 'season': season,
                 'episode': episode,
                 'frame_num': frame_num,
                 'episode': episode
                }
        data.append(datum)
    encoding_df = pd.DataFrame(data)
    df = face_df.merge(encoding_df,
                       how='left',
                       on=['season', 'episode', 'frame_num', 'face_num'])
    return df


def join_images(df,
                image_dir):
    data = []
    for file in tqdm(files):
        info, frame_num, face_num = file.stem.split('_')
        season = int(info[1:3])
        episode = int(info[4:6])
        datum = {'filename': file.name,
                 'season': season,
                 'episode': episode,
                 'frame_num': int(frame_num),
                 'face_num': int(face_num)
                }
        data.append(datum)
    fp_df = pd.DataFrame(data)
    df = df.merge(fp_df,
                  how='left',
                  on=['season', 'episode', 'frame_num', 'face_num'])
    return df
    
    
def main(args):
    name = get_info(args.series_id)
    face_dir = Path(args.face_dir)
    face_dst = face_dir.joinpath(name)
    if not face_dst.exists():
        Path.mkdir(face_dst, parents=True)

    find_faces(args.video_dir,
               face_dst,
               ext=args.ext)

    image_dst = Path(args.image_dir).joinpath(name)
    if not image_dst.exists():
        Path.mkdir(image_dst, parents=True)
        
    save_faces(face_dst,
               image_dst)

    encoding_dir = Path(args.encoding_dir).joinpath(name)
    if not encoding_dir.exists():
        Path.mkdir(encoding_dir, parents=True)

    encode_faces(image_dst,
                 encoding_dir)

    episodes = get_episode_info(args.episodes,
                                args.url,
                                args.series_id)

    df = join_encodings(args.face_dir,
                        args.encoding_dir)

    df = join_images(df,
                     args.image_dir)

    columns = [
        'series_id',
        'episode_id',
        'season',
        'episode',
        'frame_num',
        'face_num',
        'x1',
        'y1',
        'x2',
        'y2',
        'img_height',
        'img_width',
        'area',
        'pct_of_frame',
        'encoding',
        'filename',
        'character',
        'cast_id',
        'cluster'
    ]

    df = df[[x for x in columns if x in df.columns.tolist()]]

    df_ep = join_episodes(df, 
                          args.series_id)
    df_clustered = cluster_series(df_ep)

    headshot_dir = Path(args.headshot_dir).joinpath(name)
    episode_ids = df_clustered['episode_id'].unique()
    dfs = [match_clusters(args.series_id,
                          x,
                          args.episodes,
                          headshot_dir
                          ) for x in episode_ids]
    df = pd.concat(dfs)
    df.to_csv(args.dst)
    

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('series_id')
    ap.add_argument('video_dir')
    ap.add_argument('dst')
    ap.add_argument('--face_dir', default='./data/faces')
    ap.add_argument('--image_dir', default='/home/amos/storage/datasets/CineFace/images')
    ap.add_argument('--encoding_dir', default='./data/encodings')
    ap.add_argument('--cluster_dir', default='./data/clusters'
    ap.add_argument('--headshot_dir', default='./data/headshots')
    ap.add_argument('--episodes', default='./data/episodes.csv')
    ap.add_argument('--ext', default=('.mp4', '.avi', '.m4v', '.mkv')
    ap.add_argument('--url', default=None)
    args = ap.parse_args()

    if args.url is None:
        args.url = f'https://www.imdb.com/search/title/?series=tt{args.series_id}&sort=user_rating,desc&count=250&view=simple'

    ia = Cinemagoer(loggingLevel=50)
    
    main(args)
