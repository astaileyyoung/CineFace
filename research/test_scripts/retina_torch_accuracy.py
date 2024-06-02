import os 
import time 
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd 
from tqdm import tqdm

from retinaface.pre_trained_models import get_model


model = get_model("resnet50_2020-07-20", max_size=2048, device='cuda')
model.eval()


def get_box(faces):
    data = []
    face_num = 0
    for face in faces:
        if not face['bbox']:
            continue
        x1, y1, x2, y2 = face['bbox']
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
    faces = model.predict_jsons(resized)
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
                 'name': Path(src).name,
                 'duration': np.nan}]
    else:
        data = get_box(faces)
        data = format_data(img, data, src)
    return data


def main(args):
    images = [x for x in Path(args.image_dir).iterdir()]
    data = []
    for image in tqdm(images):
        t = time.time()
        preds = detect_image(str(image))
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
    ap.add_argument('--image_dir', default='../test_images')
    args = ap.parse_args()
    main(args)
