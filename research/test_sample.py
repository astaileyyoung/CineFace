import ast
from pathlib import Path 

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
# from deepface import DeepFace
from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, PointStruct, QueryRequest, Filter, FieldCondition, MatchValue
from tmdbv3api import TMDb, TV, Find, Season, Episode, exceptions, Person


CLIENT = QdrantClient(host='192.168.0.131', port=6333)


backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yolov11s',
  'yolov11n',
  'yolov11m',
  'yunet',
  'centerface',
]

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet"
]

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



def tmdb_from_imdb(imdb_id):
    from tmdbv3api import TMDb, Find

    tmdb = TMDb()
    tmdb.api_key = '64a6b6f9419ae4cba5b9a5f1c9e87401'

    imdb_id = f'tt{str(imdb_id).zfill(7)}'    # The imdb_id is stored as an integer in the database. Convert to formatted string.
    search = Find()
    results = search.find_by_imdb_id(imdb_id)
    tmdb_id = results['tv_results'][0]['id']  
    return tmdb_id


def cast_from_season(imdb_id, season_num):
    tmdb_id = tmdb_from_imdb(imdb_id)
    season = Season()
    s = season.details(tmdb_id, season_num)
    cast = [{k: v for k,v in x.items() if k in ['name', 'id']} for x in s['credits']['cast']]
    return cast


def cast_from_episode(imdb_id, season_num, episode_num):
    tmdb_id = tmdb_from_imdb(imdb_id)
    episode = Episode()
    try:
        e = episode.details(tmdb_id, season_num, episode_num)
        cast = [{k: v for k,v in x.items() if k in ['name', 'id']} for x in e['guest_stars']]
        return cast
    except exceptions.TMDbException:
        logging.error(f'Episode not found for imdb_id = {imdb_id}, season = {season_num}, episode = {episode_num}. Unable to match faces.')
        return []


def get_cast(imdb_id, season_num, episode_num=None):
    cast = cast_from_season(imdb_id, season_num)
    if episode_num:
        guest_stars = cast_from_episode(imdb_id, season_num, episode_num)
        cast.extend(guest_stars)
    return cast


data = []
sample = pd.read_csv('/home/amos/programs/CineFace/data/label_df.csv', index_col=0)
for idx, row in tqdm(sample.iterrows(), total=sample.shape[0]):
    imdb_id = row['series_id']
    season = row['season']
    episode = row['episode']

    cast = get_cast(imdb_id, season, episode)
    cast_ids = [x['id'] for x in cast]

    # cap = cv2.VideoCapture(row['filepath'])
    # cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_num'])
    # ret, frame = cap.read()
    # face = frame[row['y1']:row['y2'], row['x1']:row['x2']]

    # for model in models:
    #     encoding = DeepFace.represent(
    #         face,
    #         detector_backend='skip',
    #         model=model,
    #         align=True,
    #         enforce_detection=False,
    #         max_faces=1,
    #         normalization=normalization[models],
    #                                 )
    encoding = ast.literal_eval(row['encoding'])

    for threshold in np.arange(0, 1, 0.05):
        response = CLIENT.query_points(collection_name='Headshots',
                                        with_payload=True,
                                        query=encoding,
                                        limit=100)
        r = [x for x in response.points if x.payload['tmdb_id'] in cast_ids and x.score > threshold]
        if pd.isnull(row['tmdb_id']):
            if r:
                result = 0
            else:
                result = 1
        else:
            if r and int(row['tmdb_id']) == r[0].payload['tmdb_id']:
                result = 1
            else:
                result = 0

        datum = {**row.to_dict(), 'result': result, 'threshold': threshold}
        data.append(datum)    
df = pd.DataFrame(data)
df.to_csv('/home/amos/programs/CineFace/data/label_df_results.csv')
            