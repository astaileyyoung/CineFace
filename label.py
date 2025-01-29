from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def main(args):
    actors = {48: 'Hugh Laurie', 
              49: 'Robert Sean Leonard',
              50: 'Lisa Edelstein',
              51: 'Omar Epps',
              52: 'Jennifer Morrison',
              53: 'Jesse Spencer',
              32: None}
    df = pd.read_csv(args.src, index_col=0)
    df = df.assign(label=None)
    cap = cv2.VideoCapture(df.at[0, 'filepath'])
    frame_nums = df['frame_num'].unique().tolist()
    cnt = 0
    pb = tqdm(total=args.n)
    while cnt < args.n:
        try:
            frame_num = frame_nums.pop(np.random.randint(0, len(frame_nums)) - 1)
            temp = df[df['frame_num'] == frame_num]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            for idx, row in temp.iterrows():
                # x1 = int(row['x1'] * row['img_width'])
                # y1 = int(row['y1'] * row['img_height'])
                # x2 = int(row['x2'] * row['img_width'])
                # y2 = int(row['y2'] * row['img_height'])
                x1 = row['x1']
                y1 = row['y1']
                x2 = row['x2']
                y2 = row['y2']
                face = frame[y1:y2, x1:x2]
                cv2.imshow('frame', face)
                k = cv2.waitKey(0)
                name = actors[k]
                if not name:
                    continue

                
                df.at[idx, 'label'] = name
                df.to_csv(args.dst)
                cnt += 1
                pb.update()
        except KeyboardInterrupt:
            exit()

            
if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('n', type=int)
    args = ap.parse_args()
    main(args)
