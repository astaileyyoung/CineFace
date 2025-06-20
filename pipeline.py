import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path
from argparse import ArgumentParser

import docker
import pandas as pd
from qdrant_client import QdrantClient 

from metadata import get_metadata
from find_faces_dev import find_faces
from match_faces import match_faces
from save_faces import save_faces

from Visage.visage import run_visage


def pull_if_not_exists(image_name):
    client = docker.from_env()
    try:
        client.images.get(image_name)
    except docker.errors.ImageNotFound:
        client.images.pull(image_name)
    except docker.errors.APIError as e:
        print("Docker error: ", e)
        

def gather_files(d,
                 ext=None):
    import os 

    paths = []
    for root, dirs, files in os.walk(d):
        for name in files:
            path = Path(root).joinpath(name)
            paths.append(path)
    paths = paths if ext is None else [x for x in paths if x.suffix in ext]
    return list(sorted(paths))


def add_metadata(df, metadata):
    for k, v in metadata.items():
        if k in ['title', 'year', 'imdb_id', 'season', 'episode']:
            df[k] = v
    return df


def pipeline(file,
             client,
             frameskip=24,
             encoding_col='encoding',
             image='astaileyyoung/visage',
             log_level='info',
             show=False,
             recognition_model='Facenet',
             threshold=0.5,
             timeout=60,
             batch_size=256,
             metadata=None):
    if not metadata:
        metadata = get_metadata(file)
    
    run_visage(file, 'temp.csv', image, frameskip, log_level, show)
    df = pd.read_csv('temp.csv')
    
    df = add_metadata(df, metadata)
    df = match_faces(df, 
                     client, 
                     encoding_col=encoding_col,
                     recognition_model=recognition_model, 
                     threshold=threshold, 
                     batch_size=batch_size,
                     timeout=timeout)
    Path('temp.csv').unlink()
    return df


def main(args):
    client = QdrantClient(host=args.qdrant_client, port=args.qdrant_port)

    metadata = {
        'imdb_id': args.imdb_id,
        'season': args.season,
        'episode': args.episode
    } if args.imdb_id or args.season or args.episode else None
    df = pipeline(
            args.src, 
            client,
            encoding_col=args.encoding_col,
            image=args.image,
            frameskip=args.frameskip,
            log_level=args.log_level,
            show=args.show,
            threshold=args.threshold,
            timeout=args.timeout,
            batch_size=args.batch_size,
            metadata=metadata)

    if args.faces_dir:
        save_faces(args.dst, args.faces_dir, label='predicted_name')

    if 'filepath' in df.columns:
        df = df.drop('filepath', axis=1)
    df.to_csv(args.dst)  


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--faces_dir', default=None)
    ap.add_argument('--encoding_col', default='embedding')
    ap.add_argument('--image', default='astaileyyoung/visage', type=str)
    ap.add_argument('--frameskip', default=24, type=int)
    ap.add_argument('--log_level', default='info', type=str)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--threshold', '-t', default=0.5, type=float)
    ap.add_argument('--timeout', default=60, type=int)
    ap.add_argument('--batch_size', default=256, type=int)
    ap.add_argument('--imdb_id', default=None, type=int)
    ap.add_argument('--season', default=None, type=int)
    ap.add_argument('--episode', default=None, type=int)
    ap.add_argument('--qdrant_client', default='192.168.0.131')
    ap.add_argument('--qdrant_port', default=6333, type=int)
    args = ap.parse_args()
    main(args)
