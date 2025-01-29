import time
import threading
from pathlib import Path
from argparse import ArgumentParser

import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace


# def encode_from_filepath(df):
#     filepath = df.at[0, 'filepath']
#     cap = cv2.VideoCapture(filepath)
#     for frame_num in tqdm(df['frame_num'].unique().tolist(), leave=False):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#         ret, frame = cap.read()
#         temp = df[df['frame_num'] == frame_num]
#         for idx, row in temp.iterrows():
#             x1 = int(row['x1'] * row['img_width'])
#             y1 = int(row['y1'] * row['img_height'])
#             x2 = int(row['x2'] * row['img_width'])
#             y2 = int(row['y2'] * row['img_height'])
#             face = frame[y1:y2, x1:x2]
#             encoding = DeepFace.represent(face, 
#                                         model_name='Facenet512', 
#                                         detector_backend='skip', 
#                                         enforce_detection=False,
#                                         normalization='Facenet2018',
#                                         max_faces=1,
#                                         align=True)
#             df.at[idx, 'encoding'] = encoding[0]['embedding']
#     return df       


# def encode_from_filepath(df):
#     filepath = df.at[0, 'filepath']
#     cap = cv2.VideoCapture(filepath)
#     framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     for frame_num in tqdm(range(framecount)):
#         ret, frame = cap.read()
#         if frame_num in df['frame_num']:
#             temp = df[df['frame_num'] == frame_num]
#             for idx, row in temp.iterrows():
#                 x1 = int(row['x1'] * row['img_width'])
#                 y1 = int(row['y1'] * row['img_height'])
#                 x2 = int(row['x2'] * row['img_width'])
#                 y2 = int(row['y2'] * row['img_height'])
#                 face = frame[y1:y2, x1:x2]
#                 encoding = DeepFace.represent(face, 
#                                             model_name='Facenet512', 
#                                             detector_backend='skip', 
#                                             enforce_detection=False,
#                                             normalization='Facenet2018',
#                                             max_faces=1,
#                                             align=False)
#                 df.at[idx, 'encoding'] = encoding[0]['embedding']
#     return df      


def extract_faces(df):
    filepath = df.at[0, 'filepath']
    cap = cv2.VideoCapture(filepath)
    faces = []
    for frame_num in tqdm(df['frame_num'].unique().tolist(), leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        temp = df[df['frame_num'] == frame_num]
        for idx, row in temp.iterrows():
            x1 = int(row['x1'] * row['img_width'])
            y1 = int(row['y1'] * row['img_height'])
            x2 = int(row['x2'] * row['img_width'])
            y2 = int(row['y2'] * row['img_height'])
            face = frame[y1:y2, x1:x2]
            faces.append(face)

            # encoding = DeepFace.represent(face, 
            #                             model_name='Facenet512', 
            #                             detector_backend='skip', 
            #                             enforce_detection=False,
            #                             normalization='Facenet2018',
            #                             max_faces=1,
            #                             align=True)
            # df.at[idx, 'encoding'] = encoding[0]['embedding']
        
    return df     


class Encoder(object):
    def __init__(self, src):
        self.df = pd.read_csv(src, index_col=0)

        self.done = False
        self.faces = []
        self.encodings = []
        self.pb = tqdm(total=self.df.shape[0])

    def extract_faces(self):
        filepath = self.df.at[0, 'filepath']
        cap = cv2.VideoCapture(filepath)
        for frame_num in self.df['frame_num'].unique().tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            temp = self.df[self.df['frame_num'] == frame_num]
            for idx, row in temp.iterrows():
                x1 = int(row['x1'] * row['img_width'])
                y1 = int(row['y1'] * row['img_height'])
                x2 = int(row['x2'] * row['img_width'])
                y2 = int(row['y2'] * row['img_height'])
                face = frame[y1:y2, x1:x2]
                self.faces.append((face, idx, row))
        self.done = True

    def encode_faces(self):
        while not self.done or self.faces:
            if not self.faces:
                time.sleep(0.1)
                continue
            face, idx, datum = self.faces.pop(0)
            encoding = DeepFace.represent(face, 
                                        model_name='VGG-Face', 
                                        detector_backend='skip', 
                                        enforce_detection=False,
                                        normalization='Facenet2018',
                                        max_faces=1,
                                        align=True)
            self.df.at[idx, 'encoding'] = encoding[0]['embedding']
            self.pb.update()
    
    def encode(self):
        extract = threading.Thread(target=self.extract_faces)
        extract.start()
        encoders = [threading.Thread(target=self.encode_faces) for x in range(1)]
        for encoder in encoders:
            encoder.start()
            encoder.join()
        return self.df


def main(args):
    dst = Path('./data/faces_reencoded')
    face_dirs = [x for x in Path('./data/faces').iterdir()]
    for face_dir in tqdm(face_dirs):
        files = [x for x in face_dir.iterdir()]
        for file in tqdm(files, leave=False):
            fp = dst.joinpath(file.name)
            if fp.exists():
                continue

            df = Encoder(str(file)).encode()
            df.to_csv(str(fp))

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
