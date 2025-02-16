import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time 
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd 
from tqdm import tqdm 
from insightface.app import FaceAnalysis


def get_box(faces):
    data = []
    for face_num, face in enumerate(faces):
        x1, y1, x2, y2 = [int(x) for x in face['bbox']]
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
                'confidence': face['det_score'],
                'face_num': face_num}
        data.append(datum)
    return data


def format_data(img, data):
    new = []
    for datum in data:
        h, w = img.shape[:2]
        pct = datum['area']/(h * w)
        datum.update({'img_width': w,
                      'img_height': h,
                      'pct_of_frame': pct})
        new.append(datum)
    return new


def detect_image(fp, app):
    img = cv2.imread(str(fp))
    faces = app.get(img)
    if not faces:
        data = [{'x1': np.nan,
                 'y1': np.nan,
                 'x2': np.nan,
                 'y2': np.nan,
                 'width': np.nan,
                 'height': np.nan,
                 'area': np.nan,
                 'confidence': np.nan,
                 'face_num': np.nan,
                 'img_width': np.nan,
                 'pct_of_frame': np.nan,
                 'name': Path(fp).name,
                 'duration': np.nan}]
    else:
        data = get_box(faces)
        data = format_data(img, data)
    return data
    

def main(args):
    app = FaceAnalysis(allowed_modules=['detection'], name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))

    images = [x for x in Path(args.image_dir).iterdir()]
    data = []
    for image in tqdm(images):
        t = time.time()
        preds = detect_image(str(image), app)
        d = time.time() - t
        for i in preds:
            i['name'] = image.name
            i['duration'] = round(d, 3)
        data.extend(preds)
    df = pd.DataFrame(data)
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('dst')
    ap.add_argument('--image_dir', default='/home/amos/programs/CineFace/research/test_images')
    args = ap.parse_args()
    main(args)
