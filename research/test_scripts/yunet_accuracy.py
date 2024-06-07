from pathlib import Path 
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd 
from tqdm import tqdm 

from videotools.detectors import FaceDetectorYunet


def get_box(faces):
    data = []
    for face in faces[1]:
        confidence = face[-1]
        x1, y1, x2, y2 = list(map(int, face[:4]))
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
                'confidence': confidence}
        data.append(datum)
    return data


def format_data(img, data):
    new = []
    for datum in data:
        h, w = img.shape[:2]
        pct = datum['area']/(h * w)
        datum.update({'img_width': w,
                      'img_width': h,
                      'pct_of_frame': pct})
        new.append(datum)
    return new

    
def main(args):    
    images = [x for x in Path(args.image_dir).iterdir()]
    data = []
    for fp in tqdm(images):
        d = fd.detect(str(fp))
        if d is not None:
            for i in d:
                i['name'] = fp.name
                data.append(i)
        else:
            d = {'x1': np.nan,
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
                 'name': fp.name,
                 'duration': np.nan}
            data.append(d)
    df = pd.DataFrame(data)
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('dst')
    ap.add_argument('--image_dir', default='../test_images')
    args = ap.parse_args()
    
    fd = FaceDetectorYunet()
    
    main(args)

