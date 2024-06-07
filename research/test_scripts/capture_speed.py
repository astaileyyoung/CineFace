from argparse import ArgumentParser 

import cv2 
from tqdm import tqdm 


def main(args):
    cap = cv2.VideoCapture(args.src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for n in tqdm(range(framecount)):
        ret, frame = cap.read()
        if n % 24 == 0:
            frames.append(frame)
    print(len(frames))


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    args = ap.parse_args()
    main(args)
      