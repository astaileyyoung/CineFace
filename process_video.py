import traceback
from argparse import ArgumentParser

import cv2 
import pandas as pd
import videotools.Detectors as dt


def detect_faces(src,
                 frameskip=24,
                 target_size=None,
                 precision=3):
    detector = dt.FaceDetectorYunet()
    try:
        data = detector.predict(src,
                                frameskip=frameskip,
                                target_size=target_size)
    except Exception as e:
        traceback.print_exc()
        exit() 
    df = pd.DataFrame(data)
    return df


def main(args):
    df = detect_faces(args.src,
                      )
    df.to_csv(args.dst)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
