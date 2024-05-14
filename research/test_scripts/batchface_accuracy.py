import os 
import time 
from pathlib import Path
from argparse import ArgumentParser

import cv2
from batch_face import RetinaFace
import pandas as pd 
from tqdm import tqdm 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

det = RetinaFace(gpu_id=0, network='resnet50')


def get_box(faces):
    data = []
    face_num = 0
    for box, landmarks, confidence in faces:
        if confidence >= 0.9:
            x1, y1, x2, y2 = [int(x) for x in box]
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
                    'confidence': confidence,
                    'face_num': face_num}
            face_num += 1
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


def detect_image(fp):
    img = cv2.imread(str(fp))
    faces = det(img, cv=True)
    data = get_box(faces)
    data = format_data(img, data)
    return data
    

def main(args):
    images = pd.read_csv(args.src)
    names = images['name'].unique().tolist()
    data = []
    for name in tqdm(names):
        fp = Path('/home/amos/programs/CineFace/research/test_images').joinpath(name)
        t = time.time()
        preds = detect_image(str(fp))
        d = time.time() - t
    # for idx, row in tqdm(images.iterrows(), total=images.shape[0]):
    #     fp = Path('/home/amos/programs/CineFace/research/test_images').joinpath(row['name'])
    #     d = detect_image(str(fp))
        for i in preds:
            i['name'] = name
            i['duration'] = round(d, 3)
        data.extend(preds)
    df = pd.DataFrame(data)
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
