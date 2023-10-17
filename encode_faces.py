from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import face_recognition
from tqdm import tqdm
from deepface import DeepFace


def encode_face(src,
                dst):
    img = cv2.imread(str(src))
    if img is None or img.shape[0] == 0:
        print(f'{Path(src).name} is invalid.')
        return 
        
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    face_encodings = face_recognition.face_encodings(rgb, known_face_locations=[[0, w, h, 0]])
    return face_encodings[0] if face_encodings else None


def encode_faces(d,
                 dst):
    files = [x for x in Path(d).iterdir()]
    for file in tqdm(files):
        name = f'{file.stem}.npy'
        fp = dst.joinpath(name)
        if fp.exists():
            continue
        encoding = encode_face(file)
        if encoding:
            np.save(str(fp), encoding)
        

def main(args):
    dst = Path(args.encoding_dir)
    if not dst.exists():
        Path.mkdir(dst)

    encode_faces(args.src,
                 args.dst)
    

    

    # files = [x for x in Path(args.src).iterdir()]
    # for file in tqdm(files):
    #     name = f'{file.stem}.npy'
    #     fp = dst.joinpath(name)
    #     if fp.exists():
    #         continue
    #     encoding = encode_face(file)
    #     if encoding:
    #         np.save(str(fp), encoding)
            
            
        # if args.package == 'dlib':
        #     img = cv2.imread(str(file))
        #     if img is None or img.shape[0] == 0:
        #         print(f'{file.name} is invalid.')
        #         continue
        #     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     h, w = rgb.shape[:2]
        #     face_encodings = face_recognition.face_encodings(rgb, known_face_locations=[[0, w, h, 0]])
        # else:
        #     embedding = DeepFace.represent(img_path=str(file), enforce_detection=False)
        #     face_encodings = [np.array(embedding[0]['embedding'])]
            
        # if face_encodings:
        #     face_encoding = face_encodings[0]
        #     np.save(str(fp), face_encoding)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    main(args)
