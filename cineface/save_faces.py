from pathlib import Path
from argparse import ArgumentParser

import cv2
import pandas as pd
from tqdm import tqdm 


def save_faces(src, dst, label=None):
    df = pd.read_csv(src)
    cap = cv2.VideoCapture(df.iloc[0]['filepath'])
    df = df.set_index('frame_num')
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        _, frame = cap.read()

        x1 = row['x1']
        y1 = row['y1']
        x2 = row['x2']
        y2 = row['y2']
        face = frame[y1:y2, x1:x2]
        name = f'{idx}_{row["face_num"]}.png'
        if label is not None:
            dst_dir = Path(dst).joinpath(str(row[label]))
        else:
            dst_dir = Path(dst)

        if not dst_dir.exists():
            Path.mkdir(dst_dir, parents=True)

        fp = dst_dir.joinpath(name)
        cv2.imwrite(str(fp), face)


def main(args):
    save_faces(args.src, args.dst, label=args.label) 


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--label', default=None)
    args = ap.parse_args()
    main(args)
