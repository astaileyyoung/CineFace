import time
from pathlib import Path
from argparse import ArgumentParser

import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import pandas as pd 
from tqdm import tqdm 


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)


def get_box(face, face_num):
    x1, y1, x2, y2 = [int(x) for x in face.tolist()]
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
             'confidence': face[-1],
             'face_num': face_num}
    return datum


def format_data(img, fp, data):
    new = []
    for datum in data:
        h, w = img.shape[:2]
        pct = datum['area']/(h * w)
        datum.update({'img_width': w,
                      'img_height': h,
                      'pct_of_frame': pct,
                      'name': Path(fp).name})
        new.append(datum)
    return new


def detect_image(fp):
    img = cv2.imread(str(fp))
    faces = mtcnn.detect(img)
    if faces[0] is not None:
        data = [get_box(x, num) for num, x in enumerate(faces[0])]
        data = format_data(img, fp, data)
        return data
    else:
        return [{'x1': np.nan,
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