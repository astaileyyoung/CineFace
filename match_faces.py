import ast
import uuid
import urllib
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
import sqlalchemy as db
from tqdm import tqdm
from deepface import DeepFace
from deepface.modules import preprocessing
from retinaface import RetinaFace
from deepface.models.facial_recognition.Facenet import load_facenet128d_model

from qdrant_client import QdrantClient 
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct

from tmdbv3api import TMDb, TV, Find, Season, Episode, exceptions, Person

from utils import tmdb_from_imdb


tmdb = TMDb()
tmdb.api_key = '64a6b6f9419ae4cba5b9a5f1c9e87401'

CLIENT = QdrantClient(host='192.168.0.131', port=6333)


def cast_from_season(imdb_id, season_num):
    tmdb_id = tmdb_from_imdb(imdb_id)
    season = Season()
    s = season.details(tmdb_id, season_num)
    cast = [{k: v for k,v in x.items() if k in ['name', 'id']} for x in s['credits']['cast']]
    return cast


def cast_from_episode(imdb_id, season_num, episode_num):
    tmdb_id = tmdb_from_imdb(imdb_id)
    episode = Episode()
    e = episode.details(tmdb_id, season_num, episode_num)
    cast = [{k: v for k,v in x.items() if k in ['name', 'id']} for x in e['guest_stars']]
    return cast


def get_cast(imdb_id, season_num, episode_num=None):
    cast = cast_from_season(imdb_id, season_num)
    if episode_num:
        guest_stars = cast_from_episode(imdb_id, season_num, episode_num)
        cast.extend(guest_stars)
    return cast
    

def get_headshot(tmdb_id):
    model = load_facenet128d_model()
    person = Person()
    p = person.details(tmdb_id)
    for num, image in enumerate(p['images']['profiles']):
        url = image['file_path']
        r = urllib.request.urlopen(f'http://image.tmdb.org/t/p/w500{url}')
        img_array = np.array(bytearray(r.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)
        faces = RetinaFace.detect_faces(img)
        if not faces:
            continue

        x1, y1, x2, y2 = faces['face_1']['facial_area']
        face = img[y1:y2, x1:x2]
        fp = Path('./data/headshots').joinpath(f'{p["name"]}_{p["id"]}_{num}.png')
        cv2.imwrite(str(fp), face)
        f = preprocessing.normalize_input(preprocessing.resize_image(face, (160, 160)), normalization='Facenet')
        encoding = model(f)[0]

        # encoding = DeepFace.represent(face, 
        #                             model_name='Facenet', 
        #                             detector_backend='skip', 
        #                             enforce_detection=False,
        #                             normalization='Facenet',
        #                             max_faces=1,
        #                             align=True)[0]['embedding']

        point = models.PointStruct(id=str(uuid.uuid4()),
                                payload={'name': p['name'],
                                         'tmdb_id': p['id']},
                                vector=encoding)
        CLIENT.upsert(collection_name='Headshots',
                    points=[point])


def add_headshots(cast):
    for c in cast:
        response = CLIENT.scroll(
            collection_name='Headshots',
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key='tmdb_id',
                                            match=models.MatchValue(value=c['id']))]
            ),
            limit=1
        )
        if not response[0]:
            get_headshot(c['id'])


def main(args):
    collections = [x.name for x in CLIENT.get_collections().collections]
    if 'Headshots' not in collections:
        CLIENT.recreate_collection(collection_name='Headshots',
                                vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    if args.src:
        df = pd.read_csv(args.src, index_col=0)
    else:
        connection_string = f'mysql+pymysql://{args.username}:{args.password}@{args.host}:{args.port}/{args.database}'
        engine = db.create_engine(connection_string)
        with engine.connect() as conn:
            query = 'SELECT * FROM faces WHERE TRUE'
            if args.series_id:
                query += f' AND series_id = {args.series_id}'
            elif args.episode_id:
                query += f' AND episode_id = {args.episode_id}'
            df = pd.read_sql_query(query, conn)

    # seasons = df['season'].unique().tolist()
    # episodes = df['episode'].unique().tolist()
    # dfs = []
    # for season in seasons:
    #     for episode in episodes:
    #         episode_df = df[(df['season'] == season) &
    #                         (df['episode'] == episode)]
    #         if episode_df.shape[0] == 0:
    #             continue

    #         season = episode_df.at[0, 'season']
    #         series_id = episode_df.at[0, 'series_id']


    cast = get_cast(412142, 1, 1)
    add_headshots(cast)

    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        encoding = np.array(ast.literal_eval(row['encoding']))
        response = CLIENT.query_points(collection_name='Headshots',
                                query=encoding,
                                limit=1)
        if response.points and response.points[0].score > args.threshold:
            name = response.points[0].payload['name']
            cast_id = response.points[0].payload['tmdb_id']
        else:
            name = None
            cast_id = None
        
        df.at[idx, 'predicted_name'] = name
        df.at[idx, 'predicted_tmdb_id'] = cast_id
    df.to_csv(args.dst)
    #     dfs.append(episode_df)
    
    # if args.dst:
    #     df = pd.concat(dfs, axis=0)
    #     df.to_csv(args.dst)          


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
    main(args)
