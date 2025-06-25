import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
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


def match_faces_worker(src, client_info, encoding_col, recognition_model, threshold, batch_size, timeout):
    import gc
    from tensorflow.keras import backend as K

    df = pd.read_csv(src)
    qdrant_client, qdrant_port = client_info
    client = QdrantClient(host=qdrant_client, port=qdrant_port)
    logger.debug(f'Successfully connected to Qdrant database at {qdrant_client}: {qdrant_port}')

    result_df = match_faces(df, 
                            client, 
                            encoding_col=encoding_col,
                            recognition_model=recognition_model,
                            threshold=threshold,
                            batch_size=batch_size,
                            timeout=timeout)
    # result_df.to_csv('temp_matched.csv', index=False)
    K.clear_session()
    gc.collect()
    return result_df


def pipeline(file,
             client_info,
             frameskip=24,
             encoding_col='encoding',
             image='astaileyyoung/visage',
             model_dir=Path.home() / '.visage/models',
             log_level='info',
             show=False,
             recognition_model='Facenet',
             threshold=0.5,
             timeout=60,
             batch_size=256,
             metadata=None):
    if not metadata:
        metadata = get_metadata(file)
    
    run_visage(file, 'temp.csv', image, frameskip, log_level, show, model_dir)
    df = pd.read_csv('temp.csv')
    
    df = add_metadata(df, metadata)
    logger.info('Matching faces ...')
    df.to_csv('temp_meta.csv', index=False)

    with mp.get_context('spawn').Pool(1) as pool:
        async_result = pool.apply_async(
            match_faces_worker,
            ('temp_meta.csv', client_info, encoding_col, recognition_model, threshold, batch_size, timeout)
        )
        df = async_result.get()
    logger.info('Finished matching.')
    
    # df = pd.read_csv('temp_matched.csv')
    # df = match_faces(df, 
    #                  client,
    #                  encoding_col=encoding_col,
    #                  recognition_model=recognition_model,
    #                  threshold=threshold,
    #                  batch_size=batch_size,
    #                  timeout=timeout)
    Path('temp.csv').unlink()
    Path('temp_meta.csv').unlink()
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
    df = pipeline(
            args.src, 
            (args.qdrant_client, args.qdrant_port),
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
    logger.info(f'Results from {args.src} saved to {args.dst}')


if __name__ == '__main__':
    main()
