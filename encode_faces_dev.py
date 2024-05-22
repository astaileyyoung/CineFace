import logging
from pathlib import Path
from argparse import ArgumentParser

import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import extract_face


model_path = Path(__file__).parent.joinpath('data/dlib_face_recognition_resnet_model_v1.dat').absolute().resolve()
encoder = dlib.face_recognition_model_v1(str(model_path))


def process_image(face):
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA)


def encode_faces(src):
    if isinstance(src, str):
        try:
            df = pd.read_csv(str(src), index_col=0)
            df = df.reset_index(drop=True)
        except Exception as e:
            logging.error(e)
            exit()
    else:
        df = src 

    cap = cv2.VideoCapture(df.at[0, 'filepath'])
    frame_nums = df['frame_num'].unique().tolist()
    faces = []
    for frame_num in tqdm(frame_nums):
        ret, frame = cap.read()
        temp = df[df['frame_num'] == frame_num]
        for idx, row in temp.iterrows():
            face = extract_face(row, frame)
            resized = process_image(face)
            faces.append(resized)

    encodings = encoder.compute_face_descriptor(np.array(faces))
    df = df.assign(encoding=list(encodings))
    # for num, encoding in enumerate(encodings):
    #     e = np.array(encoding)
    #     df.at[num, 'encoding'] = encoding
    return df


def main(args):
    df = encode_faces(args.src)
    if args.dst:
        dst = args.dst 
    else:
        dst = args.src
    
    for col in args.to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    df.to_csv(dst)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--dst', default=None)
    ap.add_argument('--to_drop', default=(), nargs='+')
    ap.add_argument('--verbosity', '-v', default=10, type=int)
    ap.add_argument('--log', default='./logs/encode_faces.log')
    args = ap.parse_args()

    log_path = Path(__file__).parent.joinpath(args.log)
    logging.basicConfig(level=args.verbosity,
                    filename=log_path,
                    format='%(levelname)s %(asctime)s: %(message)s',
                    filemode='a')
    
    main(args)
