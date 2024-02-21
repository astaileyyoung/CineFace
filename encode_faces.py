from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
import face_recognition
from tqdm import tqdm
from videotools.extract_faces import extract_faces

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
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    encoding = face_recognition.face_encodings(face,
                                    known_face_locations=[(0, h, w, 0)])
    return encoding


def encode_faces(file,
                 dst):
    try:
        df = pd.read_csv(str(file), index_col=0)
    except Exception as e:
        print(str(file), e)
        exit()
    df = df.reset_index(drop=True)
    dst_dir = create_target_directory(df, dst)
    faces = extract_faces(df,
                          df.iloc[0]['video_src'])
    for num, face in enumerate(faces):
        name = f'{file.stem}_{frame_num}_{face_num}.npy'
        fp = Path(dst_dir).joinpath(name)
        if fp.exists():
            continue
        encoding = encode_face(face)
        frame_num = df.at[num, 'frame_num']
        face_num = df.at[num, 'face_num']
        np.save(str(fp), encoding)
        df.at[num, 'encoding_path'] = str(fp)
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
    args = ap.parse_args()
    main(args)
