import time 
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd 
from tqdm import tqdm 


net = cv2.dnn.readNetFromCaffe('/home/amos/programs/CineFace/research/data/deploy.prototxt',
                                '/home/amos/programs/CineFace/research/data/res10_300x300_ssd_iter_140000.caffemodel')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


def get_box(faces, img):
    data = []
    face_num = 0
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            (h, w) = img.shape[:2]
            x1, y1, x2, y2 = [int(x) for x in (faces[0, 0, i, 3:7] * np.array([w, h, w, h]))]
            width = x2 - x1
            height = y2 - y1
            area = width * height 
            datum = {'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'img_width': w,
                    'img_height': h,
                    'width': width,
                    'height': height,
                    'area': area,
                    'pct_of_frame': area/(w * h),
                    'confidence': confidence,
                    'face_num': face_num}
            face_num += 1
            data.append(datum)
    return data


def detect_image(fp):
    img = cv2.imread(str(fp))
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    data = get_box(faces, img)
    if not data:
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

