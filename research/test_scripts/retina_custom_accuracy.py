import os 
import time 
from pathlib import Path
from argparse import ArgumentParser

import cv2
import pandas as pd 
from tqdm import tqdm

from videotools.Detectors import RetinaFaceCustom


det = RetinaFaceCustom()


def get_box(faces):
    data = []
    face_num = 0
    for _, face in faces.items():
        x1, y1, x2, y2 = face['facial_area']
        width = x2 - x1
        height = y2 - y1
        area = width * height 
        datum = {'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'width': width,
                'height': height,
                'area': area,
                'confidence': face['score'],
                'face_num': face_num}
        face_num += 1
        data.append(datum)
    return data


def format_data(img, data, name):
    new = []
    for datum in data:
        h, w = img.shape[:2]
        pct = datum['area']/(h * w)
        datum.update({'img_width': w,
                      'img_height': h,
                      'pct_of_frame': pct,
                      'name': name})
        new.append(datum)
    return new


def resize_image(img, max_size=720):
    h, w = img.shape[:2]
    scale = h/max_size
    rw = int(w / scale)
    resized = cv2.resize(img, (rw, max_size))
    return resized


def detect_image(src, max_size=720):
    img = cv2.imread(str(src))
    resized = resize_image(img, max_size=max_size)
    data = det.predict_image(resized)
    data = get_box(data)
    data = format_data(img, data, Path(src).name)
    return data 


def main(args):
    images = pd.read_csv(args.src)
    names = images['name'].unique().tolist()
    data = []
    for name in tqdm(names):
        fp = Path('/home/amos/programs/CineFace/research/test_images').joinpath(name)
        d = detect_image(str(fp), max_size=args.max_size)
        data.extend(d)
    df = pd.DataFrame(data)
    df.to_csv(args.dst)
        
        
    # for idx, row in tqdm(images.iterrows(), total=images.shape[0]):
    #     fp = Path('/home/amos/programs/CineFace/research/test_images').joinpath(row['name'])
    #     d = detect_image(str(fp))
    #     for i in preds:
    #         i['name'] = name
    #         i['duration'] = round(d, 3)
    #     data.extend(preds)
    # df = pd.DataFrame(data)
    # df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--src', default='./images.csv')
    ap.add_argument('--dst', default='../test_results/retina_custom.csv')
    ap.add_argument('--max_size', default=480, type=int)
    args = ap.parse_args()
    main(args)
