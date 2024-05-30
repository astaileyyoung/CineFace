import time 
from pathlib import Path 
from argparse import ArgumentParser

import cv2
import pandas as pd
from tqdm import tqdm 
from retinaface import RetinaFace

from utils import resize_image


def format_predictions(predictions, frame_num):
    data = []
    for face_num, (_, prediction) in enumerate(predictions.items()):
        x1, y1, x2, y2 = [int(x) for x in prediction['facial_area']]
        datum = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        for k, v in prediction['landmarks'].items():
            x, y = v 
            datum[f'{k}_x'] = int(x) 
            datum[f'{k}_y'] = int(y)
        datum['confidence'] = round(prediction['score'], 3)
        datum['frame_num'] = frame_num
        datum['face_num'] = face_num
        data.append(datum)
    return data 

    
def main(args):
    t = time.time()
    # src = '/home/amos/datasets/test_videos/shining_bat.mp4'
    cap = cv2.VideoCapture(args.src, cv2.CAP_DSHOW)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_num in tqdm(range(framecount), leave=False):
        ret, frame = cap.read()
        if frame_num % 24 == 0:
            # frame = resize_image(frame)
            faces = RetinaFace.detect_faces(frame)
            format_predictions(faces, frame_num)
    print(time.time() - t)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    args = ap.parse_args()
    main(args)
