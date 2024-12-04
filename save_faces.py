from pathlib import Path
from argparse import ArgumentParser

import cv2
import pandas as pd
from tqdm import tqdm 


def main(args):
    dst_dir = Path(args.dst)
    df = pd.read_csv(args.src)
    cap = cv2.VideoCapture(df.iloc[0]['filepath'])
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame_num'])
        ret, frame = cap.read()
        x1 = int(row['x1'] * row['img_width'])
        y1 = int(row['y1'] * row['img_height'])
        x2 = int(row['x2'] * row['img_width'])
        y2 = int(row['y2'] * row['img_height'])
        face = frame[y1:y2, x1:x2]
        dst = dst_dir.joinpath(str(row['label']))
        if not dst.exists():
            Path.mkdir(dst, parents=True)
        name = f'{row["frame_num"]}_{row["face_num"]}.png'
        fp = dst.joinpath(name)
        cv2.imwrite(str(fp), face)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
