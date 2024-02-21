import re
from pathlib import Path 
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import sqlalchemy as db
from tqdm.auto import tqdm

from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

from utils import parse_filename


def string_to_array(encoding):
    t = re.sub(r'[^-.0-9\s]', '', encoding)
    t = re.sub(r' +', ',', t)
    t = t.replace('\n', '')
    a = np.fromstring(t, sep=',', dtype=float)
    if a.shape[0] != 128:
        b = 1
    # a = np.array([float(x) for x in t.split(',')])
    return a 


def inject_encodings(encodings):
    batch = []
    batch_size = 1024
    for idx, vector in tqdm(encodings):
        batch.append((idx, vector))
        if idx % batch_size == 0:
            CLIENT.upsert(
                collection_name='FacialEmbeddings',
                points=[
                    PointStruct(
                        id=i,
                        vector=v.tolist()
                    ) for i, v in batch 
                ]
            )
            batch = []


def add_encodings(d,
                  cnt=0):
    base_dir = Path('/home/amos/programs/CineFace/data/faces_new')
    subdirs = [x for x in base_dir.iterdir()]
    for subdir in tqdm(subdirs, leave=True):
        files = [x for x in subdir.iterdir()]
        for file in tqdm(files, leave=False):
            face_df = pd.read_csv(str(file), index_col=0) 
            if face_df[face_df['encoding'].isna()].shape[0] > 0:
                continue
            encodings = [(idx, np.load(row['encoding_path'])) \
                            for idx, row in tqdm(face_df.iterrows(), total=face_df.shape[0], leave=False)]
            assert len(encodings) == face_df.shape[0]
            inject_encodings(encodings)
               

def main(args):
    username = 'amos'
    password = 'M0$hicat'
    host = '192.168.0.131'
    port = '3306'
    database = 'CineFace'

    connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
    engine = db.create_engine(connection_string)
    conn = engine.connect()

    cnt = conn.execute(db.text('SELECT MAX(uid)'))

    CLIENT.recreate_collection(collection_name='FacialEmbeddings',
                           vectors_config=VectorParams(size=128, distance=Distance.COSINE))

    add_encodings(args.d)
    
         


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--d', default='./data/faces_new')
    args = ap.parse_args()

    CLIENT = QdrantClient(host='localhost', port=6333)

    main(args)
