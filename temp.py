from argparse import ArgumentParser

import cv2
from tqdm import tqdm
from deepface import DeepFace

from videotools.detectors import Yunet


def main(args):
    fd = Yunet()
    cap = cv2.VideoCapture(args.src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_num in tqdm(range(framecount)):
        ret, frame = cap.read()
        # if frame_num % 24 == 0:
        #     try:
        #         faces = fd.predict(frame)
        #     except ValueError:
        #         continue


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
