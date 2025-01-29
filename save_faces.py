from pathlib import Path
from argparse import ArgumentParser

import cv2
import pandas as pd
from tqdm import tqdm 


def main(args):
    dst_dir = Path(args.dst)
    df = pd.read_csv(args.src)
    cap = cv2.VideoCapture(df.iloc[0]['filepath'])
    frame_nums = df['frame_num'].unique().tolist()
    for frame_num in tqdm(frame_nums):
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
            name = f'{row["frame_num"]}_{row["face_num"]}.png'
            if args.label is not None:
                dst = Path(args.dst).joinpath(str(row[args.label]))
            else:
                dst = Path(args.dst)

            if not dst.exists():
                Path.mkdir(dst, parents=True)

            fp = dst.joinpath(name)
            cv2.imwrite(str(fp), face)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--label', default=None)
    args = ap.parse_args()
    main(args)
