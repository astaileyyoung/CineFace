import logging
import traceback
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
import face_recognition
from tqdm import tqdm
from videotools.extract_faces import extract_face

# def encode_face(src):
#     img = cv2.imread(str(src))
#     if img is None or img.shape[0] == 0:
#         print(f'{Path(src).name} is invalid.')
#         return 
        
#     # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # h, w = rgb.shape[:2]
#     face_encodings = DeepFace.represent(img, model_name='Facenet', enforce_detection=False)
#     return face_encodings[0] if face_encodings else None


def create_target_directory(df,
                            dst):
    series_id = df.at[0, 'series_id']
    dst_dir = Path(dst).joinpath(str(series_id))
    try:
        Path.mkdir(dst_dir)
    except:
        pass
    return dst_dir


def encode_face(face):
    h, w = face.shape[:2]
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        logging.error(e) 
        return None
    
    encoding = face_recognition.face_encodings(face,
                                    known_face_locations=[(0, h, w, 0)])
    return encoding


def encode_faces(file,
                 dst):
    try:
        df = pd.read_csv(str(file), index_col=0)
    except Exception as e:
        logging.error(e)
        exit()
    df = df.reset_index(drop=True)
    if 'encoding_path' in df.columns:
        df['encoding_path'] = ''
    dst_dir = create_target_directory(df, dst)
    cap = cv2.VideoCapture(df.at[0, 'video_src'])
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        face = extract_face(row, cap)
        encoding = encode_face(face)
        if encoding is None:
            logging.error(row['video_src'])
            return None
        frame_num = row['frame_num']
        face_num = row['face_num']
        name = f'{file.stem}_{frame_num}_{face_num}.npy'
        fp = Path(dst_dir).joinpath(name)
        np.save(str(fp), encoding)
        df.at[idx, 'encoding_path'] = str(fp)
    df.to_csv(str(file))


def encode_from_directory(src,
                          dst):
    files = [x for x in Path(src).iterdir()]
    for file in tqdm(files):
        encode_faces(file, dst)
        # name = f'{file.stem}.npy'
        # fp = Path(dst).joinpath(name)
        # if fp.exists():
        #     continue
        # encoding = encode_face(file,
        #                         dst)
        # if encoding is not None:
        #     np.save(str(fp), encoding)
        

def main(args):
    try:
        Path.mkdir(Path(args.dst))
    except:
        pass
    encode_from_directory(args.src,
                          args.dst)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--verbosity', '-v', default=10, type=int)
    ap.add_argument('--log', default='./logs/encode_faces.log')
    args = ap.parse_args()

    logging.basicConfig(level=args.verbosity,
                    filename=args.log,
                    format='%(levelname)s %(asctime)s: %(message)s',
                    filemode='a')
    
    main(args)
