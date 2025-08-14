import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import ast
import time
from argparse import ArgumentParser

import pandas as pd

from qdrant_client import QdrantClient 
from qdrant_client.models import QueryRequest, Filter, FieldCondition, MatchValue

from tmdbv3api import TMDb

from cineface.metadata import get_cast, get_headshot


tmdb = TMDb()
tmdb.api_key = '64a6b6f9419ae4cba5b9a5f1c9e87401'


def add_headshots(cast, 
                  client,
                  collection_name=None,
                  detector_backend='retinaface',
                  recognition_model='Facenet'):
    if not collection_name:
        collection_name = f'Headshots_{recognition_model}'

    for c in cast:
        response = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key='tmdb_id',
                                     match=MatchValue(value=c['id']))]
            ),
            limit=1
        )
        if not response[0]:
            get_headshot(c['id'], 
                         client,
                         collection_name=collection_name,
                         detector_backend=detector_backend,
                         recognition_model=recognition_model)


def parse_response(response,
                   cast_ids):
    r = [{**x.payload, 'score': x.score} for x in response.points if x.payload['tmdb_id'] in cast_ids]
    if r:
        name = r[0]['name']
        cast_id = r[0]['tmdb_id']
        confidence = round(r[0]['score'], 3)
    else:
        name = None
        cast_id = None
        confidence = None
    return name, cast_id, confidence


def match_faces(df, 
                metadata,
                client_info, 
                threshold=0.5, 
                timeout=60,
                recognition_model='Facenet', 
                encoding_col='encoding',
                batch_size=1024):
    host, port = client_info
    client = QdrantClient(host=host, port=port)

    imdb_id = metadata['imdb_id']
    season = metadata['season'] if 'season' in metadata else None
    episode = metadata['episode'] if 'episode' in metadata else None
   
    try:
        cast = get_cast(imdb_id, season_num=season, episode_num=episode)
        if not cast:
            return df
    except TypeError:
        raise

    characters = {x['id']: x['character'] for x in cast}
    collection_name = f'Headshots_{recognition_model}'
    cast_ids = [x['id'] for x in cast]
    add_headshots(cast, client, collection_name=collection_name, recognition_model=recognition_model)
    if isinstance(df.at[0, encoding_col], str):
        encodings = df[encoding_col].map(ast.literal_eval).tolist()
    else:
        encodings = df[encoding_col].tolist()

    response = []
    requests = []
    for num, encoding in enumerate(encodings):
        requests.append(QueryRequest(
            query=encoding, 
            with_payload=True, 
            score_threshold=threshold, 
            limit=100
            ))
        if len(requests) == batch_size or len(encodings) == num + 1:
            r = client.query_batch_points(
                collection_name=collection_name,
                requests=requests,
                timeout=timeout
            )
            response.extend(r)
            requests = []

    # requests = [QueryRequest(
    #                     query=encoding, 
    #                     with_payload=True, 
    #                     score_threshold=threshold, 
    #                     limit=100
    #                     ) for encoding in encodings]
    # response = client.query_batch_points(
    #     collection_name=collection_name,
    #     requests=requests,
    #     timeout=60
    # )

    names, cast_ids, confidence = zip(*[parse_response(x, cast_ids) for x in response])
    df = df.assign(
        predicted_name=names, 
        predicted_character=[characters[cast_id] if cast_id else None for cast_id in cast_ids],
        predicted_tmdb_id=cast_ids, 
        predicted_confidence=confidence,
        match_threshold=threshold
        )
    return df


def main(args):
    t = time.time()
    client = QdrantClient(host=args.qdrant_client, port=args.qdrant_port)
    df = pd.read_csv(args.src, index_col=0)
    if args.imdb_id or args.season or args.episode:
        df = df.assign(imdb_id=args.imdb_id, season=args.season, episode=args.episode)
    df = df.reset_index(drop=True)
    df = match_faces(df, 
                     client, 
                     threshold=args.threshold, 
                     recognition_model=args.recognition_model, 
                     batch_size=args.batch_size)
    df.to_csv(args.dst)
    print(f'Successfully matched faces for {args.src} ({time.time() - t}) and saved to {args.dst}')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--recognition_model', default='Facenet')
    ap.add_argument('--threshold',
                    '-t', 
                    default=0.5,
                    type=float)
    ap.add_argument('--timeout', default=60, type=int)
    ap.add_argument('--batch_size', default=1024, type=int)
    ap.add_argument('--imdb_id', default=None, type=int)
    ap.add_argument('--season', default=None, type=int)
    ap.add_argument('--episode', default=None, type=int)
    ap.add_argument('--qdrant_client', default='192.168.0.131')
    ap.add_argument('--qdrant_port', default=6333, type=int)
    args = ap.parse_args()
    main(args)
