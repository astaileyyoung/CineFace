import time 
from argparse import ArgumentParser

import cv2
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace 


def find_faces(src,
               frameskip=24,
               detector_backend='yolov11m',
               recognition_model='Facenet'):
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pb = tqdm(total=int(framecount / frameskip))
    data = []
    for frame_num in range(framecount):
        ret, frame = cap.read()
        if not ret:
            break
        elif frame_num % frameskip != 0:
            continue

        pb.update(1)
        try:
            faces = DeepFace.represent(frame,
                                       detector_backend=detector_backend,
                                       model_name=recognition_model,
                                       align=True,
                                       normalization='Facenet')
        except ValueError:
            continue
 
        for num, face in enumerate(faces):
            x1, y1, w, h = list(face['facial_area'].values())[:4]
            x2 = x1 + w
            y2 = y1 + h
            datum = {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'frame_num': frame_num,
                'face_num': num,
                'embedding': face['embedding'],
                'confidence': face['face_confidence']
            }
            data.append(datum)

    return pd.DataFrame(data)


def main(args):
    t = time.time()
    df = find_faces(args.src, frameskip=args.frameskip)
    if args.dst is not None:
        df.to_csv(args.dst, index=False) 
    print(time.time() - t)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--dst')
    ap.add_argument('--frameskip', default=24, type=int)
    args = ap.parse_args()
    main(args)
