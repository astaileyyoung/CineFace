import random
import urllib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import sqlalchemy as db
from tqdm import tqdm
from deepface import DeepFace
from tmdbv3api import TMDb, TV, Find, Season, Episode, exceptions, Person

from match_faces import get_cast


tmdb = TMDb()
tmdb.api_key = '64a6b6f9419ae4cba5b9a5f1c9e87401'


def get_headshot(tmdb_id):
    person = Person()
    p = person.details(tmdb_id)
    headshots = list(p['images']['profiles'])
    n = 1 if len(headshots) == 1 else 2
    for num, image in enumerate(headshots[:n]):
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
            faces = DeepFace.extract_faces(img,
                                        detector_backend='retinaface',
                                        enforce_detection=True,
                                        align=True,
                                        normalize_face=True)
        except ValueError:
            continue 

        x1 = faces[0]['facial_area']['x']
        y1 = faces[0]['facial_area']['y']
        x2 = x1 + faces[0]['facial_area']['w']
        y2 = y1 + faces[0]['facial_area']['h']
        face = img[y1:y2, x1:x2]
        fp = Path('./data/temp').joinpath(f'{p["name"]}_{p["id"]}_{num}.png').absolute().resolve()
        cv2.imwrite(str(fp), face)


username = 'amos'
password = 'M0$hicat'
host = '192.168.0.131'
port = '3306'
database = 'CineFace'

connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
engine = db.create_engine(connection_string)
conn = engine.connect()

df = pd.read_sql_query('SELECT * FROM queue WHERE to_analyze = 1 OR analyzed = 1', conn)

if not Path('./data/temp').exists():
    Path.mkdir(Path('./data/temp'))

if Path('./data/label_df.csv').exists():
    label_df = pd.read_csv('./data/label_df.csv', index_col=0)
    labeled = label_df.values.tolist()
else:
    label_df = pd.DataFrame([], columns=['x1', 'y1', 'x2', 'y2', 'encoding', 'tmdb_id', 'filepath', 'series_id', 'season', 'episode', 'pct_of_frame'])

pb = tqdm(total=385)
if label_df is not None:
    pb.update(n=label_df.shape[0])

while label_df.shape[0] < 385:
    try:
        row = df.sample(n=1).iloc[0]
        if pd.isnull(row['imdb_id']):
            continue

        filepath = row['filepath']
        imdb_id = int(row['imdb_id'])
        season = int(row['season'])
        episode = int(row['episode'])

        [Path.unlink(x) for x in Path('./data/temp').iterdir()]
        try:
            cast = get_cast(imdb_id, season, episode_num=episode)
        except:
            continue
        
        names = {x['id']: x['name'] for x in cast}
        for c in cast:
            get_headshot(c['id'])
        
        cap = cv2.VideoCapture(filepath)
        framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            frame_num = random.randint(0, framecount - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            try:
                faces = DeepFace.represent(
                    frame,
                    detector_backend='yunet',
                    model_name='Facenet',
                    normalization='Facenet',
                    enforce_detection=True,
                    align=True
                )
            except ValueError:
                continue

            face = faces[0]
            x1 = face['facial_area']['x']
            y1 = face['facial_area']['y']
            x2 = x1 + face['facial_area']['w']
            y2 = y1 + face['facial_area']['h']
            f = frame[y1:y2, x1:x2]
            cv2.imshow('frame', f)
            k = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if k == 27: # escape
                continue
            elif k == 13: # enter
                break
            elif k == 32: # space
                while True:
                    cast_id = input('Enter cast id: ')
                    if int(cast_id) not in names.keys():
                        print(f'{cast_id} not in {", ".join([str(x) for x in names.keys()])}')
                        continue
                    answer = input(f'Cast id = {cast_id} ({names[int(cast_id)]}). Is this correct (y/n): ')
                    if answer.lower() == 'y':
                        break
            elif k == 116:
                cast_id = None
            else:
                continue
            
            label = {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'encoding': face['embedding'],
                'tmdb_id': cast_id,
                'filepath': filepath,
                'series_id': imdb_id,
                'season': season,
                'episode': episode,
                'pct_of_frame': ((x2 - x1   ) * (y2 - y1))/(h * w)
            }
            label_df = pd.concat([label_df, pd.DataFrame([label])], axis=0)
            label_df.to_csv('./data/label_df.csv')
            pb.update()
            break
    except KeyboardInterrupt:
        exit()

