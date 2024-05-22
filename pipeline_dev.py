from argparse import ArgumentParser

import pandas as pd 

from find_faces_dev import detect_faces
from encode_faces_dev import encode_faces
from add_to_server import add_to_server


def main(args):
    df = detect_faces(args.src)
    df['series_id'] = 129134
    df['episode_id'] = 437913
    df = encode_faces(df)
    df = df.drop('filepath', axis=1)
    add_to_server(df, 'faces')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--faces_dir', default='/home/amos/datasets/CineFace/faces')
    ap.add_argument('--host', default='localhost', type=str)
    ap.add_argument('--username', default='amos')
    ap.add_argument('--password', default='M0$hicat')
    ap.add_argument('--port', default='3306')
    ap.add_argument('--database', default='film')
    args = ap.parse_args()
    main(args)
