from pathlib import Path
from argparse import ArgumentParser

import cv2
import dlib
import numpy as np
import pandas as pd
import sqlalchemy as db
from tqdm import tqdm


username = 'amos'
password = 'M0$hicat'
host = '192.168.0.131'
port = '3306'
database = 'CineFace'
connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}' 
engine = db.create_engine(connection_string)
conn = engine.connect()

model_path = './data/dlib_face_recognition_resnet_model_v1.dat'
encoder = dlib.face_recognition_model_v1(str(model_path))

# CLIENT = QdrantClient(host='192.168.0.131', port=6333)   
# collections = [x.name for x in CLIENT.get_collections().collections]
# if 'FacialEmbeddings' not in collections:
#     CLIENT.recreate_collection(collection_name='FacialEmbeddings',
#                                 vectors_config=VectorParams(size=128, distance=Distance.COSINE))


def process_image(face):
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA)
    return np.array(resized, dtype=np.uint8)


def encode_from_filepath(df):
    # face_df = pd.read_sql_query(f"""
    #         SELECT faces.*
    #         FROM faces
    #         LEFT JOIN queue 
    #             ON queue.filename = faces.filename
    #         WHERE queue.filepath = {filepath}
    #     """, conn)
    filepath = df.at[0, 'filepath']
    cap = cv2.VideoCapture(filepath)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_num'])
        ret, frame = cap.read()
        x1 = int(row['x1'] * row['img_width'])
        y1 = int(row['y1'] * row['img_height'])
        x2 = int(row['x2'] * row['img_width'])
        y2 = int(row['y2'] * row['img_height'])
        face = frame[y1:y2, x1:x2]
        img = process_image(face)
        encoding = encoder.compute_face_descriptor(img)
        df.at[idx, 'encoding'] = encoding
    df.to_csv('./data/fixed.csv')
        # CLIENT.upsert(
        #     collection_name='FacialEmbeddings',
        #     points=PointStruct(id=row['encoding_id'],
        #                        payload={},
        #                        vector=np.array(encoding).tolist())
        # )
        



def main(args):
    df = pd.read_csv('/home/amos/programs/CineFace/data/faces/seinfeld_1989_98904/Seinfeld.S01E04.1080p.WEBRip.x265-RARBG.csv', index_col=0)
    encode_from_filepath(df)
    # face_dirs = [x for x in Path('./data/faces').iterdir()]
    # for face_dir in face_dirs:
    #     files = [x for x in face_dir.iterdir()]
    #     for file in files:
    #         df = pd.read_csv(str(file), index_col=0)
    #         encode_from_filepath(df)

    # df = pd.read_sql_query("""
    #     SELECT DISTINCT(queue.filepath)
    #     FROM faces  
    #     LEFT JOIN queue 
    #         ON queue.filename = faces.filename
    #     """, conn)
    # for filepath in df['filepath']:
    #     encode_from_filepath(filepath)



if __name__ == '__main__':
    ap = ArgumentParser()
    args = ap.parse_args()
    main(args)
