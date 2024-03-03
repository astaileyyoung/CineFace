from pathlib import Path 
from argparse import ArgumentParser

import cv2
import dlib 
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


def detect_image(fp):
    img = cv2.imread(str(fp))
    faces = det(img)
    data = [get_box(x, num) for num, x in enumerate(faces)]
    data = format_data(img, data)
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
