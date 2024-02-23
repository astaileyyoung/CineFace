import os 
import traceback
from pathlib import Path
from argparse import ArgumentParser

import cv2
import pandas as pd
from tqdm import tqdm

from videotools.Detectors import FaceDetectorYunet


def analyze_video(src,
                  frameskip=24,
                  target_size=(300, 300),
                  encode=False,
                  encoder='DeepID'):
    d = FaceDetectorYunet(img_size=target_size)
    data = d.predict(src,
                     frameskip=frameskip,
                     encode=encode,
                     encoder=encoder)
    return data     


def main(args):
    if Path(args.src).is_dir():
        files = [x for x in Path(args.src).iterdir() if x.suffix in ('.mp4', '.mkv', '.avi', '.m4v')]
        dsts = [str(Path(args.dst).joinpath(f'{file.stem}.csv')) for file in files]
    else:
        files = [args.src]
        dsts = [args.dst]
    
    for num, file in enumerate(tqdm(files)):
        fp = dsts[num]
        if fp.exists():
            continue

        data = analyze_video(file,
                             frameskip=args.framekskip,
                             target_size=tuple(args.target_size),
                             encode=args.encode,
                             encoder=args.encoder)
        df = pd.DataFrame(data)
        df.to_csv(str(fp))



if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--frameskip', default=24, type=int)
    ap.add_argument('--target_size', '-ts', default=(300, 300), nargs='+', type=int)
    ap.add_argument('--encode', default=False)
    ap.add_argument('--encoder', default='DeepID')
    args = ap.parse_args()
    main(args)
