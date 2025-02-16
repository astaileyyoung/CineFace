import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import ast
import uuid
import urllib
import logging
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from deepface.modules import preprocessing
from deepface import DeepFace
from deepface.models.facial_recognition.Facenet import load_facenet128d_model

from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, PointStruct, QueryRequest, Filter, FieldCondition, MatchValue

from tmdbv3api import TMDb, Person

from metadata import get_cast 


tmdb = TMDb()
tmdb.api_key = '64a6b6f9419ae4cba5b9a5f1c9e87401'

CLIENT = QdrantClient(host='192.168.0.131', port=6333)
collections = [x.name for x in CLIENT.get_collections().collections]
if 'Headshots' not in collections:
    CLIENT.recreate_collection(collection_name='Headshots',
                            vectors_config=VectorParams(size=128, distance=Distance.COSINE))

model = load_facenet128d_model()
    

def get_headshot(tmdb_id, 
                 detector_backend='retinaface',
                 recognition_model='Facenet', 
                 collection_name='Headshots'):
    normalization = {
                "VGG-Face": "VGGFace2", 
                "Facenet": "Facenet", 
                "Facenet512": "Facenet", 
                "OpenFace": "base", 
                "DeepID": "base", 
                "ArcFace": "ArcFace", 
                "Dlib": "base", 
                "SFace": "base",
                "GhostFaceNet": "base"
                }
    
    person = Person()
    p = person.details(tmdb_id)
    for num, image in enumerate(p['images']['profiles']):
        u = image['file_path']
        cnt = 0
        while cnt < 5:
            url = f'http://image.tmdb.org/t/p/w500{u}'

            r = urllib.request.urlopen(url)
            img_array = np.array(bytearray(r.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            if img is None:
                cnt += 1
            else:
                break

        if img is None:
            continue 
        # DeepFace requires a 3-dimensional image. 
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        try:
            faces = DeepFace.represent(img,
                                       detector_backend=detector_backend,
                                       model_name=recognition_model,
                                       enforce_detection=True,
                                       align=True,
                                       normalization=normalization[recognition_model]
                                       )
        except ValueError:
            continue 

        x1 = faces[0]['facial_area']['x']
        y1 = faces[0]['facial_area']['y']
        x2 = x1 + faces[0]['facial_area']['w']
        y2 = y1 + faces[0]['facial_area']['h']
        face = img[y1:y2, x1:x2]
        fp = Path('./data/headshots').joinpath(f'{p["name"]}_{p["id"]}_{num}.png')
        cv2.imwrite(str(fp), face)

        # f = preprocessing.normalize_input(preprocessing.resize_image(face, (160, 160)), normalization='Facenet')
        # encoding = model(f)[0]
        encoding = faces[0]['embedding']

        point = PointStruct(
            id=str(uuid.uuid4()),
            payload={
                'name': p['name'],
                'tmdb_id': p['id']
                },
            vector=encoding)
        CLIENT.upsert(collection_name=collection_name,
                    points=[point])


def add_headshots(cast, 
                  collection_name='Headshots',
                  detector_backend='retinaface',
                  recognition_model='Facenet'):
    for c in cast:
        response = CLIENT.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key='tmdb_id',
                                     match=MatchValue(value=c['id']))]
            ),
            limit=1
        )
        if not response[0]:
            get_headshot(c['id'], 
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


def match_faces(df, threshold=0.5):
    imdb_id = df.at[0, 'imdb_id']
    season = df.at[0, 'season']
    episode = df.at[0, 'episode']

    cast = get_cast(imdb_id, season, episode)
    if not cast:
        return df

    cast_ids = [x['id'] for x in cast]
    add_headshots(cast)
    if isinstance(df.at[0, 'encoding'], str):
        encodings = df['encoding'].map(ast.literal_eval).tolist()
    else:
        encodings = df['encoding'].tolist()

    response = CLIENT.query_batch_points(
        collection_name='Headshots',
        requests=[QueryRequest(
            query=encoding, 
            with_payload=True, 
            score_threshold=threshold, 
            limit=100
            ) for encoding in encodings],
        timeout=60
    )

    names, cast_ids, confidence = zip(*[parse_response(x, cast_ids) for x in response])
    df = df.assign(
        predicted_name=names, 
        predicted_tmdb_id=cast_ids, 
        predicted_confidence=confidence
        )
    return df


def main(args):
    df = pd.read_csv(args.src, index_col=0)
    df = match_faces(df, threshold=args.threshold)
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--threshold',
                    '-t', 
                    default=0.5,
                    type=float)
    ap.add_argument('--host',
                    default='192.168.0.131',
                    type=str)
    ap.add_argument('--username',
                    default='amos')
    ap.add_argument('--password',
                    default='M0$hicat')
    ap.add_argument('--port',
                    default='3306')
    ap.add_argument('--database',
                    default='CineFace')
    ap.add_argument('--series_id', default=None)
    ap.add_argument('--episode_id', default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                    filename='./logs/match_faces.log',
                    format='%(asctime)s %(levelname)s-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

    main(args)
