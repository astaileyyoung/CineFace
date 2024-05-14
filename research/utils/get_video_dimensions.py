import os 
from pathlib import Path 
from argparse import ArgumentParser

import cv2
import pandas as pd
from tqdm import tqdm 


def gather_files(d):
    paths = []
    for root, dirs, files in os.walk(d):
        for name in files:
            path = Path(root).joinpath(name)
            if path.suffix in ['.mkv', '.avi', '.m4v', '.mp4']:
                paths.append(path)
    return paths


def get_dimensions(fp):
    cap = cv2.VideoCapture(str(fp))
    _, frame = cap.read()
    h, w = frame.shape[:2]
    # w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return (int(i) for i in [w, h])


def get_resolution(height):
    if height <= 480:
        return '480p'
    elif height <= 720:
        return '720p'
    elif height <= 1080:
        return '1080p'
    else:
        return '4k'
    

def get_aspect_ratio(w, h):
    ratio = w/h 
    if ratio <= 1.33:
        return 1.33
    elif ratio <= 1.66:
        return 1.66
    elif ratio <= 1.85:
        return 1.85
    else:
        return 2.35


def main(args):
    paths = gather_files(args.src)

    data = []
    for path in tqdm(paths):
        try:
            w, h = get_dimensions(str(path))
        except AttributeError:
            continue
        datum = {'fp': str(path),
                 'width': w,
                 'height': h,
                 'resolution': get_resolution(h),
                 'aspect_ratio': w/h,
                 'aspect_ratio_coded': get_aspect_ratio(w, h)}
        data.append(datum)
    df = pd.DataFrame(data)
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
