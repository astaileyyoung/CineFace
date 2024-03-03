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
    return data


def main(args):
    images = pd.read_csv(args.src)
    data = []
    for idx, row in tqdm(images.iterrows(), total=images.shape[0]):
        fp = Path('/home/amos/programs/CineFace/research/test_images').joinpath(row['name'])
        d = detect_image(str(fp))
        for i in d:
            i['id'] = idx
            i['name'] = row['name']
        data.extend(d)
    df = pd.DataFrame(data)
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
