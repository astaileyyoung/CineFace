from pathlib import Path 
from argparse import ArgumentParser

import cv2
import pandas as pd 
from tqdm import tqdm 

from videotools.Detectors import FaceDetectorYunet


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


# def detect_image(src):
#     org = cv2.imread(src)
#     img = cv2.cvtColor(org, cv2.COLOR_BGRA2BGR)
#     img = cv2.resize(img, (SIZE, SIZE))
#     fd.setInputSize((SIZE, SIZE))
#     faces = fd.detect(img)
#     if faces[1] is None:
#         return [] 
    
#     data = get_box(faces)
#     data = format_data(img, data)
#     return data     

    
def main(args):    
    images = pd.read_csv(args.src, index_col=0)
    names = images['name'].unique().tolist()
    data = []
    for name in tqdm(names):
        fp = Path('/home/amos/programs/CineFace/research/test_images').joinpath(name)
        d = fd.predict_image(str(fp))
        if d is not None:
            for i in d:
                i['name'] = name
                data.append(i)
            data.extend(d)
    df = pd.DataFrame(data)
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    
    fd = FaceDetectorYunet()
    
    main(args)
