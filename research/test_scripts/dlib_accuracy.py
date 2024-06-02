import time 
from pathlib import Path 
from argparse import ArgumentParser

import cv2
import dlib 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 


det = dlib.cnn_face_detection_model_v1('/home/amos/programs/CineFace/research/data/mmod_human_face_detector.dat')

def get_box(face, face_num):
    x1, y1, x2, y2 = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
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
             'confidence': face.confidence,
             'face_num': face_num}
    return datum


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


def remove_bad_faces(faces):
    new_faces = []
    for face in faces:
        for f in faces:
            x, y, w, h = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
            x2, y2, w2, h2 = f.rect.left(), f.rect.top(), f.rect.right(), f.rect.bottom()
            if x2 > x and w2 < w:
                continue
            new_faces.append(face)
    return new_faces


def detect_image(fp):
    img = cv2.imread(str(fp))
    faces = det(img, 1)
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
        # faces = remove_bad_faces(faces)
        data = [get_box(x, num) for num, x in enumerate(faces)]
        data = format_data(img, data)
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
