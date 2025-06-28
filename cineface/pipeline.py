import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import sys
import signal
import logging
import subprocess as sp
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import docker
import pandas as pd
from tensorflow.keras import backend as K
from qdrant_client import QdrantClient 

from cineface.metadata import get_metadata
from cineface.match_faces import match_faces
from cineface.save_faces import save_faces

from visage import run_visage


logger = logging.getLogger("pipeline")


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


class Pipeline(object):
    def __init__(self, client_info):
        self.client_info = client_info

        self.docker_client = docker.from_env()
        self.container = None

    def handle_sigint(self, sig, frame):
        print("\nCtrl+C detected! Attempting to stop Docker container...")
        if self.container_name:
            try:
                sp.run(['docker', 'stop', '-t', '1', self.container_name], stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=30)
                print(f"Container '{self.container_name}' stopped.")
            except Exception as e:
                print(f"Failed to stop container: {e}")
        else:
            print("No container to stop.")
        sys.exit(1)

    def run(self,
            file,
            frameskip=24,
            encoding_col='embedding',
            image='astaileyyoung/visage',
            model_dir=None,
            log_level='info',
            show=False,
            recognition_model='Facenet',
            threshold=0.5,
            timeout=60,
            batch_size=256,
            metadata=None):
        if model_dir is None:
            model_dir = Path.home() / '.visage/models'

        if not metadata:
            metadata = get_metadata(file)
        
        data, self.container_name = run_visage(file, 'temp.csv', image, frameskip, log_level, show, model_dir)
        if not data:
            return 
        
        df = pd.DataFrame(data)
        
        df = add_metadata(df, metadata)
        logger.info('Matching faces ...')
        
        df = match_faces(df, 
                         self.client_info, 
                         encoding_col=encoding_col,
                         recognition_model=recognition_model,
                         threshold=threshold,
                         batch_size=batch_size,
                         timeout=timeout)
        logger.info('Finished matching.')

        K.clear_session()
        gc.collect()

        return df


def run_pipeline_worker(src, 
                        client_info,
                        queue,
                        frameskip=1,
                        encoding_col='embedding',
                        image='astaileyyoung/visage',
                        model_dir=Path.home() / '.visage/models',
                        log_level='info',
                        show=False,
                        recognition_model='Facenet',
                        threshold=0.5,
                        timeout=60,
                        batch_size=256,
                        metadata=None
                        ):
    pipe = Pipeline(client_info)
    df = pipe.run(src,
                  encoding_col=encoding_col,
                  image=image,
                  frameskip=frameskip,
                  model_dir=model_dir,
                  recognition_model=recognition_model,
                  log_level=log_level,
                  show=show,
                  threshold=threshold,
                  timeout=timeout,
                  batch_size=batch_size,
                  metadata=metadata
    )
    queue.put(df)


def run_pipeline(src,
                 client_info,
                 frameskip=1,
                 encoding_col='embedding',
                 image='astaileyyoung/visage',
                 model_dir=None,
                 log_level='info',
                 show=False,
                 recognition_model='Facenet',
                 threshold=0.5,
                 timeout=60,
                 batch_size=256,
                 metadata=None
):
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    proc = ctx.Process(
        target=run_pipeline_worker,
        args=(
            src,
            client_info,
            queue,
            frameskip,
            encoding_col,
            image,
            model_dir,
            log_level,
            show,
            recognition_model,
            threshold,
            timeout,
            batch_size,
            metadata
        )
    )
    proc.start()
    df = queue.get()
    proc.join()
    return df 


def main():
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('imdb_id')
    ap.add_argument('--faces_dir', default=None)
    ap.add_argument('--encoding_col', default='embedding')
    ap.add_argument('--image', default='astaileyyoung/visage', type=str)
    ap.add_argument('--frameskip', default=24, type=int)
    ap.add_argument('--log_level', default='info', type=str)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--threshold', '-t', default=0.5, type=float)
    ap.add_argument('--timeout', default=60, type=int)
    ap.add_argument('--batch_size', default=256, type=int)
    ap.add_argument('--season', default=None, type=int)
    ap.add_argument('--episode', default=None, type=int)
    ap.add_argument('--qdrant_client', default='localhost')
    ap.add_argument('--qdrant_port', default=6333, type=int)
    ap.add_argument('--model_dir', default=None)
    args = ap.parse_args()

    levels = {
        'debug': 10,
        'info': 20,
        'warning': 30,
        'error': 40
    }
    level = levels[args.log_level]

    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('[%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)    

    metadata = {}
    keys = ('imdb_id', 'season', 'episode')
    for num, arg in enumerate((args.imdb_id, args.season, args.episode)):
        if arg is not None:
            metadata[keys[num]] = arg 

    logger.info(f'Running detection on {args.src}')
    logger.info(f'Saving results to {args.dst}')

    client_info = (args.qdrant_client, args.qdrant_port)
    df = run_pipeline(
        args.src,
        client_info,
        frameskip=args.frameskip,
        encoding_col=args.encoding_col,
        image=args.image,
        model_dir=args.model_dir,
        log_level=args.log_level,
        show=args.show,
        recognition_model='Facenet',
        threshold=args.threshold,
        timeout=args.timeout,
        batch_size=args.batch_size,
        metadata=metadata
    )

    if df is None:
        exit() 
    
    if args.faces_dir:
        save_faces(args.dst, args.faces_dir, label='predicted_name')

    if 'filepath' in df.columns:
        df = df.drop('filepath', axis=1)
    df.to_csv(args.dst)  
    logger.info(f'Results from {args.src} saved to {args.dst}')


if __name__ == '__main__':
    main()
