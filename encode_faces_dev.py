import logging
from pathlib import Path
from argparse import ArgumentParser

import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from videotools.extract_faces import extract_face


encoder = dlib.face_recognition_model_v1('data/dlib_face_recognition_resnet_model_v1.dat')

# def encode_face(src):
#     img = cv2.imread(str(src))
#     if img is None or img.shape[0] == 0:
#         print(f'{Path(src).name} is invalid.')
#         return 
        
#     # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # h, w = rgb.shape[:2]
#     face_encodings = DeepFace.represent(img, model_name='Facenet', enforce_detection=False)
#     return face_encodings[0] if face_encodings else None


# def create_target_directory(df,
#                             dst):
#     series_id = df.at[0, 'series_id']
#     dst_dir = Path(dst).joinpath(str(series_id))
#     try:
#         Path.mkdir(dst_dir)
#     except:
#         pass
#     return dst_dir


# def encode_face(face):
#     h, w = face.shape[:2]
#     try:
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#     except cv2.error as e:
#         logging.error(e) 
#         return None
    
#     encoding = face_recognition.face_encodings(face,
#                                     known_face_locations=[(0, h, w, 0)])
#     return encoding


def batch_encode_faces(cropped_faces):
    encodings = []
    for face in cropped_faces:
        if face.sum() > 0:
            encoding = encoder.compute_face_descriptor(face)
        else:
            encoding = None
        encodings.append(encoding)
    return encodings


def encode_faces(file):
    try:
        df = pd.read_csv(str(file), index_col=0)
    except Exception as e:
        logging.error(e)
        exit()
    df = df.reset_index(drop=True)
    df['encoding'] = ''
    cap = cv2.VideoCapture(df.at[0, 'video_src'])
    faces = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        face = extract_face(row, cap)
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        faces.append(cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA))

    encodings = encoder.compute_face_descriptor(np.array(faces))
    # encodings = batch_encode_faces(faces)
    for num, encoding in enumerate(encodings):
        e = np.array(encoding)
        df.at[num, 'encoding'] = e

        # encoding = encode_face(face)
        # if encoding is None:
        #     logging.error(row['video_src'])
        #     return None
        # frame_num = row['frame_num']
        # face_num = row['face_num']
        # name = f'{file.stem}_{frame_num}_{face_num}.npy'
        # fp = Path(dst_dir).joinpath(name)
        # np.save(str(fp), encoding)
        # df.at[idx, 'encoding_path'] = str(fp)
    return df


# def encode_from_directory(src,
#                           dst):
#     files = [x for x in Path(src).iterdir()]
#     for file in tqdm(files):
#         encode_faces(file, dst)
#         # name = f'{file.stem}.npy'
#         # fp = Path(dst).joinpath(name)
#         # if fp.exists():
#         #     continue
#         # encoding = encode_face(file,
#         #                         dst)
#         # if encoding is not None:
#         #     np.save(str(fp), encoding)
        

def main(args):
    df = encode_faces(args.src)
    if args.dst:
        dst = args.dst 
    else:
        dst = args.src
    if 'video_src' in df.columns:
        df = df.drop('video_src', axis=1)
    df.to_csv(dst)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--dst', default='data/encodings.csv')
    ap.add_argument('--verbosity', '-v', default=10, type=int)
    ap.add_argument('--log', default='./logs/encode_faces.log')
    args = ap.parse_args()

    logging.basicConfig(level=args.verbosity,
                    filename=args.log,
                    format='%(levelname)s %(asctime)s: %(message)s',
                    filemode='a')
    
    main(args)
