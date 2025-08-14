import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gc
import sys
import json
import shutil
import logging
import traceback
import subprocess as sp
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import docker
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K

from cineface.metadata import get_metadata
from cineface.match_faces import match_faces
from cineface.save_faces import save_faces

from visage import run_visage


def setup_logging(level):
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s] [cineface] [%(levelname)s]: %(message)s',
                                  datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger = logging.getLogger("pipeline")
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


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


def drop_metadata(df):
    for field in ['title', 'year', 'imdb_id', 'season', 'episode', 'match_threshold']:
        if field in df.columns:
            df = df.drop(field, axis=1)
    return df
       

def add_embeddings(df, embeddings):
    df['embedding'] = embeddings.tolist()
    return df


def drop_embeddings(df):
    df = df.drop('embedding', axis=1)
    return df


class Pipeline(object):
    def __init__(self, client_info, log_level="info"):
        self.client_info = client_info

        self.docker_client = docker.from_env()
        self.container = None

        levels = {
            'debug': 10,
            'info': 20,
            'warning': 30,
            'error': 40
        }
        self.logger = setup_logging(levels[log_level])

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
        
        try:
            data, detection_metadata, embedding_data, self.container_name = run_visage(file, 'temp', image, frameskip, log_level, show, model_dir)
        except RuntimeError:
            raise
        
        frame_nums, face_nums, embeddings = embedding_data

        self.logger.debug(f"CSV rows: {len(data)}")
        self.logger.debug(f"Frame nums: {len(frame_nums)}")
        self.logger.debug(f"Face nums: {len(face_nums)}")
        self.logger.debug(f"Embeddings: {len(embeddings)}")

        detection_metadata['match_threshold'] = threshold
        detection_metadata['imdb_id'] = metadata['imdb_id']
        detection_metadata['season'] = metadata['season']
        detection_metadata['episode'] = metadata['episode']
        
        df = pd.DataFrame(data)
        df = add_metadata(df, metadata)
        df = add_embeddings(df, embeddings)

        self.logger.info('Matching faces ...')
        
        df = match_faces(df,
                         metadata, 
                         self.client_info, 
                         encoding_col=encoding_col,
                         recognition_model=recognition_model,
                         threshold=threshold,
                         batch_size=batch_size,
                         timeout=timeout)
        df = drop_metadata(df)    
        df = drop_embeddings(df)    

        self.logger.info('Finished matching.')

        K.clear_session()
        gc.collect()

        shutil.rmtree('temp')

        return df, detection_metadata, embedding_data 
    

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
    try:
        pipe = Pipeline(client_info, log_level=log_level)
        df, detection_metadata, embedding_data = pipe.run(src,
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
        queue.put((df, detection_metadata, embedding_data))
    except:
        tb = traceback.format_exc()
        queue.put(tb)


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
    result = queue.get()
    proc.join()
    if isinstance(result, tuple):
        return result
    else:
        raise RuntimeError(f"Pipeline worker failed:\n{result}")


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
    logger = setup_logging(level)
    logger.debug(f"Log level: {level}")

    dst = Path(args.dst)
    if dst.is_file():
        logger.error("Destination must be directory, not file. Exiting.")
        exit()
    else:
        Path.mkdir(dst, exist_ok=True)
        logger.debug(f"Created destination directory at {str(dst)}")

    metadata = {}
    keys = ('imdb_id', 'season', 'episode')
    for num, arg in enumerate((args.imdb_id, args.season, args.episode)):
        if arg is not None:
            metadata[keys[num]] = arg 

    logger.info(f'Running detection on {args.src}')
    logger.info(f'Saving results to {str(dst)}')

    client_info = (args.qdrant_client, args.qdrant_port)
    try:
        df, metadata, embedding_data = run_pipeline(args.src,
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
    except RuntimeError as e:
        logger.error(f"Pipeline failed to process {args.src}: {e}")
        exit()

    if df is None:
        logger.error("DataFrame is empty. Exiting.")

    elif isinstance(df, str):
        logger.error("Returned error: ", df)
    else:
        if args.faces_dir:
            save_faces(args.dst, args.faces_dir, label='predicted_name')
        
        detection_path = dst / "detections.csv"
        metadata_path = dst / "metadata.json"
        embedding_path = dst / "embeddings.npz"

        logger.debug(f"Detection path: {str(detection_path)}")
        logger.debug(f"Metadata path: {str(metadata_path)}")
        logger.debug(f"Embedding path: {str(embedding_path)}")

        df.to_csv(detection_path)  
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        frame_nums, face_nums, embeddings = embedding_data
        np.savez(embedding_path, frame_nums=frame_nums, face_nums=face_nums, embeddings=embeddings)

        logger.info(f'Results from {args.src} saved to {str(dst)}')


if __name__ == '__main__':
    main()
