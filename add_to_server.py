import urllib
from argparse import ArgumentParser 

import numpy as np
import pandas as pd
import sqlalchemy as db 
from tqdm import tqdm 

from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams, PointStruct


CLIENT = QdrantClient(host='192.168.0.131', port=6333)    


def parse_vector(vector):
    return np.array([float(x) for x in vector.split('\n')])


def inject_encodings(df, batch_size=1024):
    batch = []
    batch_size = min(batch_size, df.shape[0])
    for idx, row in df.iterrows():
        d = {'point': {'idx': idx,
                       'vector': parse_vector(row['encoding'])},
             'payload': {'filename': row['filename'],
                         'series_id': row['series_id'],
                         'episode_id': row['episode_id'],
                         'frame_num': row['frame_num'],
                         'face_num': row['face_num']}}
        batch.append(d)
        if len(batch) >= batch_size:
            CLIENT.upsert(
                collection_name='FacialEmbeddings',
                points=[
                    PointStruct(
                        id=b['point']['idx'],
                        vector=b['point']['vector'].tolist(),
                        payload=b['payload']
                    ) for b in batch 
                ]
            )
            batch = []


def verify_columns(df, conn):
    r = conn.execute(db.text("""
                                    SELECT COLUMN_NAME 
                                    FROM INFORMATION_SCHEMA.COLUMNS
                                    WHERE TABLE_SCHEMA = 'CineFace' AND TABLE_NAME = 'faces';
                                    """))
    columns = [x[0] for x in r.fetchall() if x[0] != 'uid']
    for column in columns:
        if column not in df.columns.tolist():
            return False 
    return True 
    

def add_to_server(df,
                  table,
                  username='amos',
                  host='192.168.0.131',
                  password='M0$hicat',
                  port='3306',
                  database='CineFace'):
    connection_string = f'mysql+pymysql://{username}:{urllib.parse.quote(password)}@{host}:{port}/{database}'
    engine = db.create_engine(connection_string)
    with engine.connect() as conn:
        # df = df.reset_index().rename({'index': 'uid'}, axis=1)
        if verify_columns(df, conn):
            # r = conn.execute(db.text('SELECT COUNT(*) FROM faces;'))
            # cnt = r.fetchone()[0]
            # df['uid'] = df['uid'].map(lambda x: x + cnt)
            sql = df.drop(['encoding', 'filepath'], axis=1)
            sql.to_sql(table, conn, if_exists='append', index=False)
            inject_encodings(df)
            conn.commit()



def main(args):
    CLIENT.recreate_collection(collection_name='FacialEmbeddings',
                           vectors_config=VectorParams(size=128, distance=Distance.COSINE))
    
    df = pd.read_csv(args.src, index_col=0)
    add_to_server(df, 
                  args.table,
                  username=args.username,
                  host=args.host,
                  port=args.port,
                  database=args.database)
     



if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--table', default='faces')
    ap.add_argument('--host', default='192.168.0.131', type=str)
    ap.add_argument('--username', default='amos')
    ap.add_argument('--password', default='M0$hicat')
    ap.add_argument('--port', default='3306')
    ap.add_argument('--database', default='CineFace')
    args = ap.parse_args()
    main(args)
