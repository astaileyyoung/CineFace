import uuid
import shutil
from pathlib import Path
from argparse import ArgumentParser

import cv2
import dlib
import numpy as np
import pandas as pd
import sqlalchemy as db
from tqdm import tqdm
from retinaface import RetinaFace
from tmdbv3api import Person, TMDb
from icrawler.builtin import GoogleImageCrawler

from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, PointStruct, FieldCondition, Filter, MatchValue, SearchParams

from utils import create_table


username = 'amos' 
password = 'M0$hicat' 
host = '192.168.0.131' 
port = '3306' 
database = 'CineFace'  

connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
ENGINE = db.create_engine(connection_string) 
CONN = ENGINE.connect()

model_path = './data/dlib_face_recognition_resnet_model_v1.dat'
ENCODER = dlib.face_recognition_model_v1(str(model_path))

tmdb = TMDb()
tmdb.api_key = '64a6b6f9419ae4cba5b9a5f1c9e87401'

CLIENT = QdrantClient(host='192.168.0.131', port=6333)   


def download_cast(cast_id):
    person = Person()
    details = person.details(cast_id)
    name = details['name']
    dst = Path('./tempdir')
    if not dst.exists():
        Path.mkdir(dst)
    crawler = GoogleImageCrawler(storage={'root_dir': str(dst)})
    crawler.crawl(keyword=f'{name}', max_num=20)
    return details


def upload_cast(cast_id, save_dir=None):
    details = download_cast(cast_id)

    paths = [x for x in Path('./tempdir').iterdir()]
    data = []
    for path in paths:
        id_ = str(uuid.uuid4())
        img = cv2.imread(str(path))
        faces = RetinaFace.detect_faces(img)

        # If there is more than one face in the image, then we can't know which encoding is the correct one.
        if len(faces) > 1:
            continue

        d = list(faces.values())[0]
        x1, y1, x2, y2 = d['facial_area']
        img_h, img_w = img.shape[:2]
        w = x2 - x1
        h = y2 - y1
        area = w * h
        img_area = img_h * img_w
        pct_of_frame = area/img_area

        # Face needs to be resized to 150x150 for the dlib encoder.
        face = img[y1:y2, x1:x2]
        resized = cv2.resize(img, (150, 150))
        encoding = ENCODER.compute_face_descriptor(resized)

        datum = {'cast_id': cast_id,
                 'headshot_id': id_,
                 'name': details['name'],
                 'x1': round(x1/img_w, 3),
                 'y1': round(y1/img_h, 3),
                 'x2': round(x2/img_w, 3),
                 'y2': round(y2/img_h, 3),
                 'img_height': img_w,
                 'img_width': img_h,
                 'pct_of_frame': round(pct_of_frame, 3),
                 'encoding': encoding,
                 'face': face,
                 'fp': str(path)}
        data.append(datum)
    best = list(sorted(data, key=lambda x: x['pct_of_frame'], reverse=True))[:5]
    for datum in best:
        # Save the image if a destination is provided.
        if save_dir is not None:
            dst = Path(save_dir).joinpath(details['name'])
            if not dst.exists():
                Path.mkdir(dst, parents=True)
            fp = dst.joinpath(f'{datum["headshot_id"]}.png')
            cv2.imwrite(str(fp), datum['face'])
        
        CLIENT.upsert(collection_name='Headshots',
              points=[
                  PointStruct(id=datum['headshot_id'],
                              payload={'cast_id': str(cast_id),
                                       'name': details['name']},
                              vector=np.array(encoding).tolist())
              ])

        df = pd.DataFrame([datum])
        df = df.drop(['encoding', 'fp', 'face'], axis=1)
        df.to_sql('headshots', CONN, if_exists='append', index=False)
        CONN.commit()
    [x.unlink() for x in paths]
    Path('./tempdir').rmdir()


def headshots_from_episode(episode_id, save_dir=None):
    cast = CONN.execute(db.text(f'SELECT cast FROM episodes WHERE episode_id = {args.episode_id}')).fetchone()[0]
    cast_ids = cast.split(',')

    for cast_id in cast_ids:
        # Only download headshots if actor is not already in database.
        existing = pd.read_sql_query(f'SELECT * FROM headshots WHERE cast_id = {cast_id}', CONN)
        if existing.shape[0] == 0:
            upload_cast(cast_id, save_dir=save_dir)


def main(args):
    if not db.inspect(ENGINE).has_table('headshots'):
        create_table('./sql/tables/headshots.sql', CONN)

    collections = [x.name for x in CLIENT.get_collections().collections]
    if 'Headshots' not in collections:
        CLIENT.recreate_collection(collection_name='Headshots',
                                    vectors_config=VectorParams(size=128, distance=Distance.COSINE))
    

    headshots_from_episode(args.episode_id, save_dir=args.save_dir)
    

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('episode_id')
    ap.add_argument('--save_dir', default='/home/amos/datasets/CineFace/headshots')
    args = ap.parse_args()
    main(args)
