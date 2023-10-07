from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import face_recognition
from tqdm import tqdm
from deepface import DeepFace


def main(args):
    dst = Path(args.dst)
    if not dst.exists():
        Path.mkdir(dst)

    files = [x for x in Path(args.src).iterdir()]
    for file in tqdm(files):
        name = f'{file.stem}.npy'
        fp = dst.joinpath(name)
        if fp.exists():
            continue
            
        if args.package == 'dlib':
            img = cv2.imread(str(file))
            if img is None or img.shape[0] == 0:
                print(f'{file.name} is invalid.')
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            face_encodings = face_recognition.face_encodings(rgb, known_face_locations=[[0, w, h, 0]])
        else:
            embedding = DeepFace.represent(img_path=str(file), enforce_detection=False)
            face_encodings = [np.array(embedding[0]['embedding'])]
            
        if face_encodings:
            face_encoding = face_encodings[0]
            np.save(str(fp), face_encoding)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--package', default='dlib')
    args = ap.parse_args()
    main(args)
