import sys 

sys.path.append('/home/amos/programs/CineFace')

import logging
from pathlib import Path 
from collections import namedtuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, PointStruct, QueryRequest, Filter, FieldCondition, MatchValue
from tmdbv3api import TMDb, TV, Find, Season, Episode, exceptions, Person

from match_faces import add_headshots

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
  'SCRFD'
]

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
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


class Detector(object):
    def __init__(self):
        self.app = FaceAnalysis(allowed_modules=['detection'], name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect(self, img, backend, model):
        if backend != 'SCRFD':
            try:
                faces = DeepFace.represent(
                            img,
                            detector_backend=backend,
                            model_name=model,
                            align=True,
                            enforce_detection=True,
                            normalization=normalization[model],
                                                    )
                return faces
            except ValueError:
                return []
        else:
            faces = self.app.get(img)
            new = []
            for face in faces:
                x1, y1, x2, y2 = [max(0, int(x)) for x in face['bbox']]
                f = img[y1:y2, x1:x2]
                encoding = DeepFace.represent(
                    f,
                    detector_backend='skip',
                    model_name=model,
                    align=True,
                    enforce_detection=False,
                    normalization=normalization[model],
                                            )
                datum = {
                    'facial_area': {
                        'x': x1,
                        'y': y1,
                        'w': x2 - x1,
                        'h': y2 - y1},
                    'embedding': encoding[0]['embedding'],
                    'confidence': face['det_score']}
                new.append(datum)
            return new


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


def calc_overlap(a, b):
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    aa = Rectangle(a['x1'], a['y1'], a['x2'], a['y2'])
    bb = Rectangle(b['x1'], b['y1'], b['x2'], b['y2'])
    dx = min(aa.xmax, bb.xmax) - max(aa.xmin, bb.xmin)
    dy = min(aa.ymax, bb.ymax) - max(aa.ymin, bb.ymin)
    if (dx>0) and (dy>0):
        area = (aa.xmax - aa.xmin) * (aa.ymax - aa.ymin)
        return (dx * dy) / area
    else:
        return 0


def validate(row, r):
    tmdb_id = int(row['tmdb_id']) if not pd.isnull(row['tmdb_id']) else None
    if pd.isnull(tmdb_id) and not r:
        status = 'tn'
    elif not pd.isnull(tmdb_id) and not r:
        status = 'fn'
    elif pd.isnull(tmdb_id) and r:
        status = 'fp'
    elif not pd.isnull(tmdb_id) and r and r[0].payload['tmdb_id'] == row['tmdb_id']:
        status = 'tp' 
    elif not pd.isnull(tmdb_id) and r and r[0].payload['tmdb_id'] != row['tmdb_id']:
        status = 'mp'
    return status
 

detector = Detector()

data = []
existing = pd.read_csv('/home/amos/programs/CineFace/research/data/label_df_results.csv', index_col=0)
sample = pd.read_csv('/home/amos/programs/CineFace/research/data/label_df.csv', index_col=0)
sample = sample[sample['frame_num'].notna()]
for idx, row in tqdm(sample.iterrows(), total=sample.shape[0], desc='Images'):
    imdb_id = row['series_id']
    season = row['season']
    episode = row['episode']

    cast = get_cast(imdb_id, season, episode)
    cast_ids = [x['id'] for x in cast]

    cap = cv2.VideoCapture(row['filepath'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_num'])
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    for backend in tqdm(backends, desc='Backends', leave=False):
        for model in tqdm(models, desc='Models', leave=False):
            temp = existing[(existing['filepath'] == row['filepath']) &
                            (existing['detector'] == backend) &
                            (existing['recognition_model'] == model)]
            if temp.shape[0] > 0:
                d = temp.to_dict(orient='records')
                data.extend(d)
                continue
            
            faces = detector.detect(frame, backend, model)
            if not faces:
                datum = {
                    **row.to_dict(), 
                    'detector': backend,
                    'recognition_model': np.nan,
                    'status': np.nan, 
                    'threshold': np.nan
                    }
                data.append(datum)
                break
     
            encoding = None
            for f in faces:
                d = {
                    'x1': f['facial_area']['x'],
                    'y1': f['facial_area']['y'],
                    'x2': f['facial_area']['x'] + f['facial_area']['w'],
                    'y2': f['facial_area']['y'] + f['facial_area']['h']
                    }
                overlap = calc_overlap(row, d)
                if overlap >= 0.5:
                    encoding = f['embedding']
                    break
        
            # The value for encoding is only set if the overlap is greater than 0.5
            if not encoding:
                datum = {
                    **row.to_dict(), 
                    'detector': backend,
                    'recognition_model': np.nan,
                    'status': np.nan, 
                    'threshold': np.nan}
                break
            else:
                encoding = np.array(encoding)

            collections = [x.name for x in CLIENT.get_collections().collections]
            collection_name = f'Headshots_{model}'
            size = encoding.size
            if collection_name not in collections:
                CLIENT.recreate_collection(collection_name=collection_name,
                                        vectors_config=VectorParams(size=size, distance=Distance.COSINE))
            
            add_headshots(cast, collection_name=collection_name, recognition_model=model)
    
            for threshold in np.arange(0, 1, 0.05):
                response = CLIENT.query_points(collection_name=collection_name,
                                                with_payload=True,
                                                query=encoding,
                                                limit=100,
                                                timeout=1000)
                r = [x for x in response.points if x.payload['tmdb_id'] in cast_ids and x.score > threshold]
                status = validate(row, r)

                datum = {
                    **row.to_dict(), 
                    'detector': backend,
                    'recognition_model': model,
                    'status': status, 
                    'threshold': threshold
                    }
                data.append(datum)    
            df = pd.DataFrame(data)
            df.to_csv('/home/amos/programs/CineFace/research/data/label_df_results.csv')
            