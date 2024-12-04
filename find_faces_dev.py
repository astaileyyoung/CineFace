import os 

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging 
import traceback
from pathlib import Path
from argparse import ArgumentParser 

import cv2 
import dlib 
import numpy as np
import pandas as pd
from tqdm import tqdm 
from retinaface import RetinaFace

from utils import gather_files


logging.getLogger("h5py").setLevel(logging.ERROR)
logging.getLogger("github").setLevel(logging.ERROR)


model_path = Path(__file__).parent.joinpath('data/dlib_face_recognition_resnet_model_v1.dat').absolute().resolve()
encoder = dlib.face_recognition_model_v1(str(model_path))


def distance_from_center(row):
    x = int((row['x2'] - row['x1'])/2)
    y = int((row['y2'] - row['y1'])/2)
    
    xx = int(row['img_width']/2)
    yy = int(row['img_height']/2)
    a = abs(yy - y) 
    b = abs(xx - x)
    c = np.sqrt(a*a + b*b)
    return round(c, 2) 


def pct_of_frame(row):
    x = int((row['x2'] - row['x1'])/2)
    y = int((row['y2'] - row['y1'])/2)

    xx = int(row['img_width']/2)
    yy = int(row['img_height']/2)

    pct_of_frame = (x * y)/(xx * yy)
    return round(pct_of_frame, 4) 


def calc(df):
    df['distance_from_center'] = df.apply(distance_from_center, axis=1)
    df['pct_of_frame'] = df.apply(pct_of_frame, axis=1)
    return df


def format_predictions(predictions, frame_num, encodings):
    data = []
    for face_num, (_, prediction) in enumerate(predictions.items()):
        x1, y1, x2, y2 = [int(x) for x in prediction['facial_area']]
        datum = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        for k, v in prediction['landmarks'].items():
            x, y = v 
            datum[f'{k}_x'] = int(x) 
            datum[f'{k}_y'] = int(y)
        datum['confidence'] = round(prediction['score'], 3)
        datum['frame_num'] = frame_num
        datum['face_num'] = face_num
        datum['encoding'] = np.array(encodings[face_num])
        data.append(datum)
    return data 


def process_image(face):
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (150, 150), interpolation=cv2.INTER_AREA)


def extract_face(box, frame):
    x1, y1, x2, y2 = box 
    return frame[y1:y2, x1:x2]


def encode(faces, frame):
    f = [process_image(extract_face(v['facial_area'], frame)) for k, v in faces.items()]
    encodings = encoder.compute_face_descriptor(np.array(f)) 
    return encodings


def detect_faces(src, frameskip=24):
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    data = []
    for frame_num in tqdm(range(framecount), desc=Path(src).name, leave=False):
        ret, frame = cap.read()
        if not ret or frame is None:
            if (framecount - frame_num) <= frameskip:
                break 
            else:
                return None
        elif frame_num % frameskip == 0:
            faces = RetinaFace.detect_faces(frame)
            encodings = encode(faces, frame)
            d = format_predictions(faces, frame_num, encodings)
            data.extend(d)
    df = pd.DataFrame(data)
    df['img_width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    df['img_height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return df


def main(args):
    if not Path(args.src).exists():
        logging.error(f'{args.src} does not exist. Exiting.')
        
    dst = Path(args.dst)
    if dst.is_dir() and not dst.exists():
        Path.mkdir(dst)
        logging.debug(f'Created destination directory at {args.dst}')

    if Path(args.src).is_dir():
        files = gather_files(args.src, ext=args.ext)
        if Path(args.dst).is_dir() and not Path(args.dst).exists():
            Path.mkdir(args.dst)
            logging.debug(f'Created destination directory at {args.dst}')
        dst = [Path(args.dst).joinpath(x.name) for x in files]
    else:
        files = [Path(args.src)]
        dst = [args.dst]

    for num, file in enumerate(files):
        fp = dst[num]
        try:
            df = detect_faces(str(file))
        except:
            e = traceback.format_exc()
            logging.error(e)
            exit()
        if args.imdb_id:
            df['series_id'] = args.imdb_id
        if args.episode_id:
            df['episode_id'] = args.episode_id
        df['filepath'] = str(file)
        df.to_csv(str(fp))
        logging.debug(f'Saved detected faces to {str(fp)}')


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--ext', default=('.mp4', '.avi', '.m4v', '.mkv'))
    ap.add_argument('--imdb_id', default=None)
    ap.add_argument('--episode_id', default=None)
    ap.add_argument('--log_path', default='./find_faces.log')
    ap.add_argument('--verbosity', '-v', default=10, type=int)
    args = ap.parse_args()

    sh = logging.StreamHandler()
    sh.setLevel(40)
    fh = logging.FileHandler(args.log_path,
                             mode='a')
    fh.setLevel(args.verbosity)
    logging.basicConfig(handlers=[fh, sh],
                        format='%(levelname)s  %(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d_%H:%M:%S')
    
    main(args)
