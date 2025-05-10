import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"


from pathlib import Path
from argparse import ArgumentParser

from qdrant_client import QdrantClient 

from metadata import get_metadata
from find_faces import find_faces
from match_faces import match_faces
from save_faces import save_faces


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
             num_threads=4,
             detection_backend='SCRFD',
             recognition_model='Facenet',
             threshold=0.5,
             metadata=None):
    if not metadata:
        metadata = get_metadata(file)
    
    df = find_faces(
            file, 
            num_threads=num_threads,
            detection_backend=detection_backend,
            recognition_model=recognition_model)
    if df is None:
        return 
    
    df = add_metadata(df, metadata)
    df = match_faces(df, client, recognition_model=recognition_model, threshold=threshold)
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
            num_threads=args.num_threads,
            detection_backend=args.detection_backend,
            recognition_model=args.recognition_model,
            threshold=args.threshold,
            metadata=metadata)
    df.to_csv(args.dst)  

    if args.faces_dir:
        save_faces(args.dst, args.faces_dir, label='predicted_name')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--faces_dir', default=None)
    ap.add_argument('--num_threads', '-n', default=4, type=int)
    ap.add_argument('--detection_backend', '-db', default='SCRFD')
    ap.add_argument('--recognition_model', '-rm', default='Facenet')
    ap.add_argument('--threshold', '-t', default=0.5, type=float)
    ap.add_argument('--imdb_id', default=None, type=int)
    ap.add_argument('--season', default=None, type=int)
    ap.add_argument('--episode', default=None, type=int)
    ap.add_argument('--qdrant_client', default='192.168.0.131')
    ap.add_argument('--qdrant_port', default=6333, type=int)
    args = ap.parse_args()
    main(args)
